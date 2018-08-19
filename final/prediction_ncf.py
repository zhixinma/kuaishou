# coding: utf-8
from utils import *

################################## 
# Documents number  2532604 	##
################################## 
# EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 20		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
TRAIN_PERCENTAGE = 0.8			##
##################################

data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit = load_interaction()
text_data = load_text()
# profile_matrix = generate_prof()
# face_data = load_face()
# train_visual, test_visual, train_photo_to_index, test_photo_to_index = load_visual()
# user_text = get_user_text(profile_matrix, text_data)


nb_words = 233243
nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
len_train = train_interaction.shape[0]
len_test = test_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]

# user_doc['photo_id'] = user_doc['photo_id'].astype(int)
# user_doc['user_id_doc'] = user_doc['user_id_doc'].apply(lambda x:[int(s) for s in x.split(' ')])
# train = pd.merge(train,user_doc,on='photo_id',how='left')
# test = pd.merge(test,user_doc,on='photo_id',how='left')
# y = train_interaction['click'].values

import random
shuffled_index = range(len_train)
random.shuffle(shuffled_index)
boundary = int(len_train * TRAIN_PERCENTAGE)
ind_tr = np.array(shuffled_index[:boundary])
ind_te = np.array(shuffled_index[boundary:])

train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]

def padding(a, N):
	return (a + N * [0])[:N]

for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


batch_size = 1178
def batch_generator(batch_size, batch_index, item_num):
	for i in range(item_num//batch_size):
		base = i*batch_size
		end = base + batch_size
		# 
		user_batch = train['user_id'].values[batch_index[base:end]]
		photo_batch = train['photo_id'].values[batch_index[base:end]]
		# 
		visual_batch = []
		text_batch = []
		for photo_id in photo_batch:
			text_batch.append(text_data[photo_reverse_fit[photo_id]])
			# visual_batch.append(train_visual[train_photo_to_index[photo_reverse_fit[photo_id]]])
		label_batch = train['click'].values[batch_index[base:end]]
		yield user_batch, text_batch, visual_batch, label_batch


from model import *

layer_num = 2
gpu_num = 0
ctx = mx.gpu(gpu_num)
# myncf = sncf(nb_users)
# myncf.initialize(ctx=ctx)
# myrnn = rnn_net('gru', nb_words, EMBEDDING_DIM_WORD, 512, layer_num)
# myrnn.initialize(ctx=ctx)
sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.01})
acc = mx.metric.Accuracy()

total_epoch = 60
IS_TRAIN = True
print("******************INFO********************")
print("GPU DEVICE: %d"%gpu_num)
print("IS   TRAIN: %s"%IS_TRAIN)
print("EPOCH  NUM: %d"%total_epoch)
print("BATCH SIZE: %d"%batch_size)
print('TRAIN  NUM: %d'%train_item_num)
print('TEST   NUM: %d'%test_item_num)
print("TRAIN PART: %f"%TRAIN_PERCENTAGE)
print("******************************************")

if IS_TRAIN:
	for epoch in range(total_epoch):
		it = 0
		for user_batch, text, visual, label in batch_generator(batch_size, ind_tr, train_item_num):
			with autograd.record():
				# states, final_state = myrnn(nd.array(text,ctx=ctx), nd.zeros((layer_num, batch_size, 512),ctx=ctx))
				# pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(states, ctx=ctx), nd.array(visual,ctx=ctx))
				pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx))
				loss = sigmoid_binary_cross_entropy_loss(pred, nd.array(label,ctx=ctx))
				if it%100 == 0:
					acc.update(preds = [pred],labels = [nd.array(label,ctx=ctx)])
					print('epoch %d it %d loss %f'%(epoch, it, nd.mean(loss).asnumpy()[0]),'acc',acc.get())
			it += 1
			loss.backward()
			trainer.step(batch_size)
		labels = []
		preds = []
		for user_batch, text, visual, label in batch_generator(batch_size, ind_te, test_item_num):
			# states, final_state = myrnn(nd.array(text,ctx=ctx), nd.zeros((layer_num, batch_size, 512),ctx=ctx))
			# pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(states, ctx=ctx), nd.array(visual,ctx=ctx))
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx))
			preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
			labels += np.array(label).reshape((1, -1)).tolist()[0]
		score = scoreAUC(labels, preds)	
		print('epoch %d auc score %f'%(epoch, score))
		myncf.collect_params().save('model/sncf/text_only')
		# myrnn.collect_params().save('model/sncf/rnn_1')
else:
	epoch = 19
	myncf.load_params("model/sncf/ncf_vt_sgd_E%d.params"%epoch, ctx=ctx)
	preds = []
	for user_batch, text, visual, label in batch_generator(batch_size, ind_res, len_test):
		# states, final_state = myrnn(nd.array(text,ctx=ctx), nd.zeros((layer_num, batch_size, 512),ctx=ctx))
		# pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(states, ctx=ctx), nd.array(visual,ctx=ctx))
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/ncf_lda_nt_E%d.txt'%epoch,'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
