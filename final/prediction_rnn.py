# coding: utf-8
from utils import *

data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit = load_interaction()
text_data = load_text()

nb_words = 233243
nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
len_train = train_interaction.shape[0]

train = data[:len_train]
test = data[len_train:]

################################## 
# EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 128		##
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

train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]
print('BATCH GENERATED')
print('ind_tr num: %d'%train_item_num)
print('ind_te num: %d'%test_item_num)

def padding(a, N):
	return (a + N * [0])[:N]

for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


batch_size = 1024
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
		label_batch = train['click'].values[batch_index[base:end]]
		yield user_batch, text_batch, visual_batch, label_batch


from model import *

layer_num = 2
num_hidden = 256
ctx = mx.gpu(3)
myncf = ncf_rnn(nb_users)
myncf.initialize(ctx=ctx)
myrnn = rnn_net(mode='gru', vocab_size=nb_words, num_embed=EMBEDDING_DIM_WORD, num_hidden=num_hidden, num_layers=layer_num)
myrnn.initialize(ctx=ctx)
sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.001})
acc = mx.metric.Accuracy()

for epoch in range(60):
	it = 0
	for user_batch, text, visual, label in batch_generator(batch_size, ind_tr, train_item_num):
		with autograd.record():
			states, final_state = myrnn(nd.array(text,ctx=ctx), nd.zeros((layer_num, batch_size, num_hidden),ctx=ctx))
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(states, ctx=ctx))
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
		states, final_state = myrnn(nd.array(text,ctx=ctx), nd.zeros((layer_num, batch_size, num_hidden),ctx=ctx))
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(states, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
		labels += np.array(label).reshape((1, -1)).tolist()[0]
	score = scoreAUC(labels, preds)	
	print('epoch %d auc score %f'%(epoch, score))
	# myncf.collect_params().save('model/sncf/ncf_rnn_1')
	# myrnn.collect_params().save('model/sncf/rnn_1')
	myncf.save_params("model/rnn/ncf_gru_512_E%d.params"%epoch)
	myrnn.save_params("model/rnn/gru_512_E%d.params"%epoch)
