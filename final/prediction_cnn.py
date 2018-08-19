# coding: utf-8
from utils import *

data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit = load_interaction()
text_data = load_text()
profile_matrix = generate_prof()
face_data = load_face()
train_visual, test_visual, train_photo_to_index, test_photo_to_index = load_visual()
user_text = get_user_text(profile_matrix, text_data)
user_visual = get_user_visual(profile_matrix, train_photo_to_index)

nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
nb_words = 233243

len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]

user_doc['photo_id'] = user_doc['photo_id'].astype(int)
user_doc['user_id_doc'] = user_doc['user_id_doc'].apply(lambda x:[int(s) for s in x.split(' ')])
train = pd.merge(train,user_doc,on='photo_id',how='left')
test = pd.merge(test,user_doc,on='photo_id',how='left')
y = train_interaction['click'].values

from sklearn.model_selection import StratifiedKFold

################################## 
# EMBEDDING_DIM_USER = 512		##
EMBEDDING_DIM_WORD = 512		##
EMBEDDING_DIM_PHOTO = 64		##
MAX_SENTENCE_LENGTH = 20		##
MAX_TEXT_PER_USER = 1024		##
MAX_INTERACTION_ITEM = 100		##
##################################

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_interaction['user_id'],y)
for ind_tr, ind_te in skf:
    break


train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]
print('BATCH GENERATED')
print('ind_tr num: %d'%train_item_num)
print('ind_te num: %d'%test_item_num)
print('test num: %d'%test['user_id_doc'].values.shape[0])

def padding(a, N):
	return (a + N * [0])[:N]

# test_user_mean = np.array([padding(v, MAX_SENTENCE_LENGTH) for v in test['user_id_doc'].values])
# train_user_mean =  np.array([padding(v, MAX_SENTENCE_LENGTH) for v in train['user_id_doc'].values])

for user_id in user_text:
	user_text[user_id] = padding(user_text[user_id], MAX_TEXT_PER_USER)


for photo_id in text_data:
	text_data[photo_id] = padding(text_data[photo_id], MAX_SENTENCE_LENGTH)


batch_size = 30
def batch_generator(batch_size, batch_index, item_num):
	for i in range(item_num//batch_size):
		base = i*batch_size
		end = base + batch_size
		# get transformed user_id and photo_id
		user_batch = train['user_id'].values[batch_index[base:end]]
		photo_batch = train['photo_id'].values[batch_index[base:end]]
		# get text & vis batch
		visual_batch = []
		text_batch = []
		for photo_id in photo_batch:
			text_batch.append(train_text[photo_reverse_fit[photo_id]][:MAX_SENTENCE_LENGTH])
			visual_batch.append(train_visual[train_photo_to_index[photo_reverse_fit[photo_id]]])
		# get emb_vis
		user_vis_batch = []
		for user_id in user_batch:
			user_vis_batch.append(train_visual[user_visual[user_reverse_fit[user_id]]])
		# get emb_text
		user_text_batch = []
		for user_id in user_batch:
			user_text_batch.append(user_text[user_reverse_fit[user_id]])
		# get label_batch
		label_batch = y[batch_index[base:end]]
		# del user_batch, photo_batch 
		yield user_text_batch, user_vis_batch, text_batch, visual_batch, label_batch

from model import *

ctx = mx.gpu(0)
myncf = ncf()
myncf.initialize(ctx=ctx)
mycnn = cnn()
mycnn.initialize(ctx=ctx)
myrnn_1 = rnn_net('gru', nb_words, EMBEDDING_DIM_WORD, 512, 10)
myrnn_1.initialize(ctx=ctx)
myrnn_2 = rnn_net('gru', nb_words, EMBEDDING_DIM_WORD, 512, 10)
myrnn_2.initialize(ctx=ctx)

sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = mx.gluon.Trainer(myncf.collect_params() and mycnn.collect_params() and myrnn_1.collect_params() and myrnn_2.collect_params(), 'adam', {'learning_rate': 0.001})
acc = mx.metric.Accuracy()

for epoch in range(30):
	it = 0
	for user_text_batch, user_vis_batch, text, visual, label in batch_generator(batch_size, ind_tr, train_item_num):
		with autograd.record():
			user_text_emb, _ = myrnn_1(nd.array(text,ctx=ctx), nd.zeros((10, batch_size, 512),ctx=ctx))
			text_emb, _ = myrnn_2(nd.array(user_text_batch,ctx=ctx), nd.zeros((10, batch_size, 512),ctx=ctx))
			vis_emb = mycnn(nd.array(user_vis_batch, ctx=ctx))
			pred = myncf(nd.array(user_text_emb), nd.array(vis_emb), nd.array(text_emb,ctx=ctx), nd.array(visual,ctx=ctx))
			label = nd.array(label,ctx=ctx)
			loss = sigmoid_binary_cross_entropy_loss(pred, label)
			if it%100 == 1:
				print('epoch %d it %d loss %f'%(epoch, it, nd.mean(loss).asnumpy()[0]))
		loss.backward()
		trainer.step(batch_size)
		if(it%100 == 0):
			print("judging...")
			labels = []
			preds = []
			for user_text_batch, user_vis_batch, text, visual, label in batch_generator(batch_size, ind_te, test_item_num):
				text_emb, hidden = myrnn(nd.array(user_text_batch,ctx=ctx), nd.array(hidden,ctx=ctx))
				vis_emb = mycnn(nd.array(user_vis_batch, ctx=ctx))
				pred = myncf(text_emb, vis_emb, nd.array(text,ctx=ctx), nd.array(visual,ctx=ctx))
				preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
				labels += np.array(label).reshape((1, -1)).tolist()[0]
			score = scoreAUC(labels, preds)	
			print('epoch %d auc score %f'%(epoch, score))
			myncf.collect_params().save('model/ncf_cnn/ncf_1')
			mycnn.collect_params().save('model/ncf_cnn/cnn_1')
			myrnn_1.collect_params().save('model/ncf_cnn/rnn1_1')
			myrnn_2.collect_params().save('model/ncf_cnn/rnn2_1')
		it += 1
