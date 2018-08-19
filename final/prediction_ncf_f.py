# coding: utf-8
from utils import *

######################################## Interaction ###########################################################################################
data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit, le_user, le_photo = load_interaction(True)

nb_words = 233243
nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
len_test = test_interaction.shape[0]
len_train = train_interaction.shape[0]

train = data[:len_train]
test = data[len_train:]

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


######################################## Face Data ############################################
# gender, age, gender
face_path = "%sface_data.npy"%PATH
if path.isfile(face_path):
	face_data = np.load(face_path)
else:
	face = load_face()
	photo_id_set = le_photo.transform(face.keys())
	face_data = np.zeros((nb_photos,3), dtype=int)
	face_data[0]
	it = 0
	size = len(face)
	for photo_id in photo_id_set:
		if it%100000==0:
			print("process %d/%d"%(it, size))
		it += 1
		face_data[photo_id] = np.array(face[photo_reverse_fit[photo_id]][1:])
	np.save(face_path, face_data)

age_size = np.max(face_data[:,1]) + 1
score_size = np.max(face_data[:,2]) + 1
face_data[face_data[:,1] == 0] = np.array([2,0,0])

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

##################################
# EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 20		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
TRAIN_PERCENTAGE = 0.8			##
##################################

import random
shuffled_index = range(len_train)
random.shuffle(shuffled_index)
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
		text_batch = []
		# text_batch = np.zeros((batch_size, MAX_SENTENCE_LENGTH))
		# visual_batch = np.zeros((batch_size, 2048))
		# for batch_it in xrange(batch_size):
			# photo_id = photo_batch[batch_it]
			# text_batch[batch_it] = text_data[photo_reverse_fit[photo_id]]
			# visual_batch[batch_it] = visual[photo_to_index[photo_reverse_fit[photo_id]]]
		###################### doc_batch ########################
		user_doc_batch = []
		# user_doc_batch =  user_doc[photo_batch]
		photo_doc_batch = []
		# photo_doc_batch =  photo_doc[user_batch]
		# face_batch = []
		face_batch = face_data[photo_batch]
		#########################################################
		label_batch = data['click'].values[batch_index[base:end]]
		# if i%100==0 or i > total-5:
		# 	print("user_batch", np.array(user_batch).shape, 
		# 			# "text_batch", np.array(text_batch).shape, 
		# 			# "visual_batch", np.array(visual_batch).shape, 
		# 			# "photo_doc_batch", np.array(photo_doc_batch).shape, 
		# 			# "user_doc_batch", np.array(user_doc_batch).shape, 
		# 			"face_batch", np.array(face_batch).shape, 
		# 			"label_batch", np.array(label_batch).shape)
		yield user_batch, photo_batch, text_batch, user_doc_batch, photo_doc_batch, visual_batch, face_batch, label_batch


from model import *

gpu_num = 0
ctx = mx.gpu(gpu_num)
myncf = ncf_f(nb_users, age_size, score_size)
myncf.initialize(ctx=ctx)
loss_func = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.001 })
acc = mx.metric.Accuracy()

total_epoch = 5
IS_TRAIN = False
model_name = 'ncf_f'
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
		for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_tr, train_item_num, {}, IS_TRAIN):
			with autograd.record():
				pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
				loss = loss_func(pred, nd.array(label,ctx=ctx))
				if it%100 == 0:
					acc.update(preds = [pred],labels = [nd.array(label,ctx=ctx)])
					print('epoch %d it %d loss %f'%(epoch, it, nd.mean(loss).asnumpy()[0]),'acc',acc.get())
			it += 1
			loss.backward()
			trainer.step(batch_size)
		labels = []
		preds = []
		for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(train, [], user_doc, photo_doc, batch_size, ind_te, test_item_num, {}, IS_TRAIN):
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
			preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
			labels += np.array(label).reshape((1, -1)).tolist()[0]
		score = scoreAUC(labels, preds)	
		print('epoch %d auc score %f'%(epoch, score))
		if score > max_auc:
			max_auc = score
			myncf.collect_params().save('model/sncf/%s.params'%model_name)
			print('Model saved')
	print("Max AUC Score", max_auc)
	######################################################################
	############################### OUTPUT ###############################
	######################################################################
	IS_TRAIN = False
	myncf.collect_params().load('model/sncf/%s.params'%model_name, ctx=ctx)
	preds = []
	for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(test, [], user_doc, photo_doc, batch_size, ind_res, len_test, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/%s_submission.txt'%model_name,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds
	preds = []
	index = np.array(range(len_train))
	for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(train, [], user_doc, photo_doc, batch_size, index, test_item_num+train_item_num, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	np.save('res/%s_on_train.npy'%model_name, np.array(preds)[:len_train])
else:
	myncf.collect_params().load('model/sncf/%s.params'%model_name, ctx=ctx)
	preds = []
	for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(test, [], user_doc, photo_doc, batch_size, ind_res, len_test, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/%s_submission.txt'%model_name,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds
	preds = []
	index = np.array(range(len_train))
	for user_batch, photo_batch, text, user_doc_batch, photo_doc_batch, visual, face_batch, label in batch_generator(train, [], user_doc, photo_doc, batch_size, index, test_item_num+train_item_num, {}, IS_TRAIN):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(face_batch,ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	np.save('res/%s_on_train.npy'%model_name, np.array(preds)[:len_train])

