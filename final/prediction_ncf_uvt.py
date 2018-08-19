# coding: utf-8
################################## 
# EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 20		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
TRAIN_PERCENTAGE = 0.8			##
##################################

from utils import *
import random

data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit = load_interaction()
text_data, nb_words = load_text()
userep = load_userep()
train_visual, test_visual, train_photo_to_index, test_photo_to_index = load_visual()

nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
len_test = test_interaction.shape[0]
len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]

def padding(a, N):
	return (a + N * [0])[:N]

# min_photo_per_user = np.min([len(x) for x in photo_doc])
# min_photo_per_user = 20
# photo_doc = np.array([x[:min_photo_per_user] for x in photo_doc]).astype(int)
# min_user_per_photo = 12
# user_doc = np.array([padding(x, min_user_per_photo) for x in user_doc]).astype(int)

for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


shuffled_index = range(len_train)
random.shuffle(shuffled_index)
# shuffled_index = train_interaction['time'].argsort().tolist()
boundary = int(len_train * TRAIN_PERCENTAGE)
ind_tr = np.array(shuffled_index[:boundary])
ind_te = np.array(shuffled_index[boundary:])
ind_res = np.array(range(len_test))

train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]

batch_size = 1024
def batch_generator(data, visual, user_doc, photo_doc, batch_size, batch_index, item_num, photo_to_index, is_train):
	if is_train or (item_num % batch_size == 0):
		total = item_num//batch_size
	else:
		total = item_num//batch_size + 1
	####################################
	for i in range(total):
		base = i*batch_size
		end = base + batch_size
		##########################################################
		user_batch = data['user_id'].values[batch_index[base:end]]
		photo_batch = data['photo_id'].values[batch_index[base:end]]
		#############################################################
		if not is_train:
			if len(user_batch) < batch_size and len(user_batch) > 0:
				user_batch = user_batch.tolist() + [user_batch[0]]*(batch_size-len(user_batch))
				photo_batch = photo_batch.tolist() + [photo_batch[0]]*(batch_size-len(photo_batch))
			if i%100==0 or i > total-5:
				print("item: %d/%d %d"%(i, total, len(user_batch)))
		##### V&T batch #####		
		visual_batch = []
		text_batch = np.zeros((batch_size, MAX_SENTENCE_LENGTH))
		# visual_batch = np.zeros((batch_size, 2048))
		for batch_it in xrange(batch_size):
			photo_id = photo_batch[batch_it]
			text_batch[batch_it] = text_data[photo_reverse_fit[photo_id]]
			# visual_batch[batch_it] = visual[photo_to_index[photo_reverse_fit[photo_id]]]
		###################### doc_batch ########################
		# user_doc_batch = []
		user_doc_batch =  user_doc[photo_batch]
		photo_doc_batch = []
		# photo_doc_batch =  photo_doc[user_batch]
		# user_rep_batch = []
		user_rep_batch = userep[user_batch]
		#########################################################
		label_batch = data['click'].values[batch_index[base:end]]
		# if i%100==0 or i > total-5:
		# 	print("user_batch", np.array(user_batch).shape, 
		# 		  "text_batch", np.array(text_batch).shape, 
		# 		  "user_rep_batch", np.array(user_rep_batch).shape, 
		# 		  # "visual_batch", np.array(visual_batch).shape, 
		# 		  # "photo_doc_batch", np.array(photo_doc_batch).shape, 
		# 		  "user_doc_batch", np.array(user_doc_batch).shape, 
		# 		  "label_batch", np.array(label_batch).shape)
		yield user_batch, user_rep_batch, photo_batch, text_batch, user_doc_batch, photo_doc_batch, visual_batch, label_batch


from model import *

gpu_num = 2
ctx = mx.gpu(gpu_num)
myncf = ncf_uit(nb_users, nb_photos, nb_words)
myncf.initialize(ctx=ctx)
# lr_sch = mx.lr_scheduler.FactorScheduler(step=20000, factor=0.65)
loss_func = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
# trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.001, 'lr_scheduler': lr_sch})
trainer = mx.gluon.Trainer(myncf.collect_params(), 'sgd', {'learning_rate': 0.1})
acc = mx.metric.Accuracy()

total_epoch = 500
IS_TRAIN = True
model_name = 'ncf_uit_1'
print("******************INFO********************")
print("USER   NUM: %d"%nb_users)
print("PHOTO  NUM: %d"%nb_photos)
print("WORD   NUM: %d"%nb_words)
print("GPU DEVICE: %d"%gpu_num)
print("MODEL NAME: %s"%model_name)
print("IS   TRAIN: %s"%IS_TRAIN)
print("EPOCH  NUM: %d"%total_epoch)
print("BATCH SIZE: %d"%batch_size)
print('TRAIN  NUM: %d'%train_item_num)
print('TEST   NUM: %d'%test_item_num)
print("TRAIN PART: %f"%TRAIN_PERCENTAGE)
print("******************************************")

max_auc = 0
if IS_TRAIN:
	for epoch in range(total_epoch):
		it = 0
		for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_tr, train_item_num, {}, IS_TRAIN):
			with autograd.record():
				pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
				loss = loss_func(pred, nd.array(label,ctx=ctx))
				if it%100 == 0:
					acc.update(preds = [pred],labels = [nd.array(label,ctx=ctx)])
					print('epoch %d it %d loss %f, max_au %f'%(epoch, it, nd.mean(loss).asnumpy()[0], max_auc),'acc',acc.get())
			it += 1
			loss.backward()
			trainer.step(batch_size)
		labels = []
		preds = []
		for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_te, test_item_num, {}, IS_TRAIN):
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
			preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
			labels += np.array(label).reshape((1, -1)).tolist()[0]
		score = scoreAUC(labels, preds)	
		print('epoch %d auc score %f'%(epoch, score))
		if score > max_auc:
			max_auc = score
			myncf.collect_params().save('model/sncf/%s.params'%model_name)
			print('Model saved')
	print("Max AUC Score", max_auc)
	IS_TRAIN = False
	myncf.collect_params().load('model/sncf/%s.params'%model_name, ctx=ctx)
	preds = []
	for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(test, [], user_doc, photo_doc, batch_size, ind_res, len_test, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/%s_submission.txt'%model_name,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds
	preds = []
	index = np.array(range(len_train))
	for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, index, test_item_num+train_item_num, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	np.save('res/%s_on_train.npy'%model_name, np.array(preds)[:len_train])
else:
	myncf.collect_params().load('model/sncf/%s.params'%model_name, ctx=ctx)
	preds = []
	for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(test, [], user_doc, photo_doc, batch_size, ind_res, len_test, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/%s_submission.txt'%model_name,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds
	preds = []
	index = np.array(range(len_train))
	for user_batch, user_rep, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, index, test_item_num+train_item_num, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_rep, ctx=ctx), nd.array(photo_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	np.save('res/%s_on_train.npy'%model_name, np.array(preds)[:len_train])

