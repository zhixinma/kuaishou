# coding: utf-8
################################## 
EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 20		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
TRAIN_PERCENTAGE = 0.8			##
MAX_INTERACTION_PER_UT = 32		##
# UT means <user, time>			##
##################################

from utils import *
import random

ut_ut, ut_p, ut_c, user_doc, photo_doc, train_num, test_num, user_reverse_fit, photo_reverse_fit = load_interaction_UT(True)
data = np.concatenate((np.reshape(ut_ut[:,0],(-1,1)), ut_p, ut_c),axis=1)
text_data, nb_words = load_text()

nb_users = np.max(ut_ut[:,0]) + 1
nb_photos = np.max(ut_p) + 1
len_train = train_num
len_test = test_num
train = data[:len_train]
test = data[len_train:]


def padding(a, N):
	return (a + N * [0])[:N]

min_user_per_photo = 12
user_doc = np.array([padding(x, min_user_per_photo) for x in user_doc]).astype(int)

for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


text_matrix = np.zeros((nb_photos, MAX_SENTENCE_LENGTH), dtype=int)
for i in photo_reverse_fit:
	text_matrix[i] = np.array(text_data[photo_reverse_fit[i]])


shuffled_index = range(len_train)
random.shuffle(shuffled_index)
boundary = int(len_train * TRAIN_PERCENTAGE)
ind_tr = np.array(shuffled_index[:boundary])
ind_te = np.array(shuffled_index[boundary:])
ind_res = np.array(range(len_test))

train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]

batch_size = 512
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
		user_batch = data[batch_index[base:end], 0]
		photo_batch = data[batch_index[base:end], 1:33]
		#############################################################
		if not is_train:
			if len(user_batch) < batch_size and len(user_batch) > 0:
				user_batch = user_batch.tolist() + [user_batch[0]]*(batch_size-len(user_batch))
				photo_batch = photo_batch.tolist() + [photo_batch[0]]*(batch_size-len(photo_batch))
			if i%100==0 or i > total-5:
				print("item: %d/%d %d"%(i, total, len(user_batch)))
		##### V&T batch #####		
		visual_batch = []
		text_batch = np.zeros((batch_size, MAX_INTERACTION_PER_UT, MAX_SENTENCE_LENGTH))
		# visual_batch = np.zeros((batch_size, 2048))
		for batch_it in xrange(batch_size):
			photo_id_list = photo_batch[batch_it]
			text_batch[batch_it] = text_matrix[photo_id_list]
			# visual_batch[batch_it] = visual[photo_to_index[photo_reverse_fit[photo_id]]]
		###################### doc_batch ########################
		# user_doc_batch = []
		user_doc_batch =  user_doc[photo_batch]
		photo_doc_batch = []
		# photo_doc_batch =  photo_doc[user_batch]
		#########################################################
		label_batch = data[batch_index[base:end], 33:]
		# if i%100==0 or i > total-5:
		# 	print("user_batch", np.array(user_batch).shape, 
		# 			"text_batch", np.array(text_batch).shape, 
		# 			"visual_batch", np.array(visual_batch).shape, 
		# 			"photo_doc_batch", np.array(photo_doc_batch).shape, 
		# 			"user_doc_batch", np.array(user_doc_batch).shape, 
		# 			"label_batch", np.array(label_batch).shape)
		yield user_batch, photo_batch, text_batch, user_doc_batch, photo_doc_batch, visual_batch, label_batch


from model import *

gpu_num = 0
ctx = mx.gpu(gpu_num)
myncf = ncf_itt(nb_users, nb_photos, nb_words)
myncf.initialize(ctx=ctx)
sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.001})
# trainer = mx.gluon.Trainer(myncf.collect_params(), 'sgd', {'learning_rate': 0.1})
acc = mx.metric.Accuracy()

total_epoch = 10
IS_TRAIN = False
model_name = 'ncf_itt'
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
		for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_tr, train_item_num, {}, IS_TRAIN):
			with autograd.record():
				pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
				loss = sigmoid_binary_cross_entropy_loss(pred, nd.array(label,ctx=ctx))
				pred = nd.array(pred.asnumpy().reshape((1,-1)))
				label = np.array(label).reshape((1, -1))
				if it%100 == 0:
					acc.update(preds = [pred],labels = [nd.array(label,ctx=ctx)])
					print('epoch %d it %d loss %f'%(epoch, it, nd.mean(loss).asnumpy()[0]),'acc',acc.get())
			it += 1
			loss.backward()
			trainer.step(batch_size)
		labels = []
		preds = []
		for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_te, test_item_num, {}, IS_TRAIN):
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
			preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
			labels += np.array(label).reshape((1, -1)).tolist()[0]
		score = scoreAUC(labels, preds)
		print('epoch %d auc score %f'%(epoch, score))
		if score > max_auc:
			max_auc = score
			myncf.collect_params().save('model/sncf/%s.params'%model_name)
			print('Model saved')
	print("MAX AUC SCORE IS %f"%max_auc)
else:
	myncf.collect_params().load('model/sncf/%s.params'%model_name, ctx=ctx)
	preds = np.zeros((1,32))
	for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(test, [], user_doc, photo_doc, batch_size, ind_res, len_test, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
		# preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
		preds = np.concatenate((np.array(preds), pred.asnumpy()),axis=0)
	preds = preds[1:]
	with open('res/%s_submission.txt'%model_name,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds

# index = np.array(range(len_train))
# preds = np.zeros((1,32))
# for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, label in batch_generator(train, [], user_doc, photo_doc, batch_size, index, test_item_num+train_item_num, {}, IS_TRAIN):
# 	pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(user_doc_batch,ctx=ctx), nd.array(text, ctx=ctx))
# 	preds = np.concatenate((np.array(preds), pred.asnumpy()),axis=0)


# preds = preds[1:len_train + 1]

preds = np.load("%spred_itt.npy"%PATH)
lens = np.zeros((len_train, 1), dtype=int)
for row_it in xrange(len_train):
	row = data[row_it, 1:33]
	count = 1
	first = row[0]
	for pid in row[1:]:
		if pid == first:
			break
		else:
			count += 1
	lens[row_it] = count


res_on_train = []
for row_it in xrange(len_train):
	if row_it%5000==0:
		print("%d/%d"%(row_it, len_train))
	row = (np.reshape(preds[row_it, 1:1+lens[row_it][0]], (-1,1))).tolist()
	res_on_train += row


res_on_train = np.array(res_on_train).reshape((1,-1))
sort = np.argsort(-res_on_train)

# 0.799
# 0.16318601369857788
click_pred = (preds+0.29391497373580933).astype(int)
statistic = np.zeros((len_train, 2), dtype=int)
for row_it in xrange(len_train):
	if row_it%5000==0:
		print("%d/%d"%(row_it, len_train))
	row = np.reshape(preds[row_it, 1:1+lens[row_it][0]], (-1,1))
	statistic[row_it][0] = np.sum(row)
	statistic[row_it][1] = lens[row_it][0] - np.sum(row)

statistic[:,0]+1 /(statistic[:,0]+statistic[:,1]+1)
np.save("%sstatistics"%PATH, statistic)
