# coding: utf-8
from utils import *

################################## 
# Documents number  2532604 	##
##################################
EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 10		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
TRAIN_PERCENTAGE = 0.8			##
TOPIC_NUM = 100					##
ITER_NUM = 1500					##
##################################

data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit = load_interaction()
text_data, nb_words = load_text()
def padding(a, N):
	return (a + N * [0])[:N]


for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


# fitted word
# perp_list = []
# for i in range(11)[2:]:
# 	TOPIC_NUM = i*50
# 	topic_word, doc_topic = load_topic(text_data, TOPIC_NUM, ITER_NUM)
# 	perplexcity = compute_preplexity(topic_word, doc_topic)
# 	print("TOPIC_NUM: %d PERPLEXCITY:%f"%(TOPIC_NUM, perplexcity))
# 	perp_list.append(perplexcity)
# 	break
##############################################
# topic_word: topic_num*vocabulary_size		##
# doc_topic : document_num*topic_num		##
##############################################

topic_word, doc_topic = load_topic(text_data, TOPIC_NUM, ITER_NUM)
doc_topic = doc_topic[0]
topic_sort = doc_topic.argsort()
words = topic_word[topic_sort].argsort()[:,-1]
topics = list(set(words))
topics_num = len(topics)
 
nb_words = 233243
nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
len_test = test_interaction.shape[0]
len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]

# shuffle
import random
shuffled_index = range(len_train)
random.shuffle(shuffled_index)
boundary = int(len_train * TRAIN_PERCENTAGE)
ind_tr = np.array(shuffled_index[:boundary])
ind_te = np.array(shuffled_index[boundary:])
ind_res = np.array(range(len_test))

train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]

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
		label_batch = train['click'].values[batch_index[base:end]]
		yield user_batch, text_batch, visual_batch, label_batch


from model import *

gpu_num = 2
ctx = mx.gpu(gpu_num)
myncf = ncf_tt(nb_users, topics_num, MAX_SENTENCE_LENGTH)
myncf.initialize(ctx=ctx)
sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.01})
acc = mx.metric.Accuracy()

total_epoch = 60
IS_TRAIN = True
model_name = 'lda'
print("******************INFO********************")
print("GPU DEVICE: %d"%gpu_num)
print("MODEL NAME: %d"%model_name)
print("TOPIC  NUM: %d"%topics_num)
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
				pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx), nd.array(topics, ctx=ctx))
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
			pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx), nd.array(topics, ctx=ctx))
			preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
			labels += np.array(label).reshape((1, -1)).tolist()[0]
		score = scoreAUC(labels, preds)	
		print('epoch %d auc score %f'%(epoch, score))
		myncf.collect_params().save('model/lda/%s_E%d.params'%(model_name, epoch))
else:
	epoch = 2
	myncf.load_params('model/lda/%s_E%d.params'%(model_name, epoch), ctx=ctx)
	preds = []
	for user_batch, text, visual, label in batch_generator(batch_size, ind_res, len_test):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx), nd.array(topics, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	with open('res/ncf_%s_E%d.txt'%(model_name, epoch),'wb') as submission:
		for user_id, photo_id, propability in zip(test['user_id'].values, test['photo_id'].values, preds):
			submission.write("%d\t%d\t%f\n"%(user_reverse_fit[user_id], photo_reverse_fit[photo_id], propability))
	del preds
	epoch = 2
	myncf.load_params('model/lda/%s_E%d.params'%(model_name, epoch), ctx=ctx)
	preds = []
	index = np.array(range(len_train))
	it = 0
	for user_batch, text, visual, label in batch_generator(batch_size, index, len_train):
		pred = myncf(nd.array(user_batch,ctx=ctx), nd.array(text, ctx=ctx), nd.array(topics, ctx=ctx))
		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
		it += 1
	np.save('res/%s_E%d.npy'%(model_name, epoch), np.array(preds)[:len_train])

