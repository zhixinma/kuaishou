# coding: utf-8
import scipy as sp
from scipy import sparse as ssp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from utils import *
import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet.gluon import nn
from mxnet import autograd

data, user_doc, photo_doc, train_interaction, test_interaction = load_interaction()

ctx = mx.gpu(1)
print('Preparing embedding matrices')

MAX_SENTENCE_LENGTH = 30
EMBEDDING_DIM_PHOTO = 64
EMBEDDING_DIM_USER = 64
EMBEDDING_DIM_WORD = 64
nb_users = data['user_id'].nunique()
nb_photos = data['photo_id'].nunique()
nb_words = 100000

len_train = train_interaction.shape[0]
train = data[:len_train]
test = data[len_train:]

user_doc['photo_id'] = user_doc['photo_id'].astype(int)
user_doc['user_id_doc'] = user_doc['user_id_doc'].apply(lambda x:[int(s) for s in x.split(' ')])

train = pd.merge(train,user_doc,on='photo_id',how='left')
test = pd.merge(test,user_doc,on='photo_id',how='left')
y = train_interaction['click'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_interaction['user_id'],y)
for ind_tr, ind_te in skf:
    break


print('ind_tr num: %d'%ind_tr.shape[0])
print('ind_te num: %d'%ind_te.shape[0])
train_item_num = ind_tr.shape[0]
test_item_num = ind_te.shape[0]
print('Test.value.length: %d'%test['user_id_doc'].values.shape[0])

def padding(a, N):
	return (a + N * [0])[:N]


print("Padding...")
test_user_mean = nd.array([padding(v, MAX_SENTENCE_LENGTH) for v in test['user_id_doc'].values],ctx = ctx)
train_user_mean =  nd.array([padding(v, MAX_SENTENCE_LENGTH) for v in train['user_id_doc'].values],ctx = ctx)

def scoreAUC(labels, probs):
    i_sorted = sorted(range(len(probs)),key=lambda i: probs[i],
                      reverse=True)
    auc_temp = 0.0
    TP = 0.0
    TP_pre = 0.0
    FP = 0.0
    FP_pre = 0.0
    P = 0
    N = 0
    last_prob = probs[i_sorted[0]] + 1.0
    for i in range(len(probs)):
        if last_prob != probs[i_sorted[i]]:
            auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0
            TP_pre = TP
            FP_pre = FP
            last_prob = probs[i_sorted[i]]
        if labels[i_sorted[i]] == 1:
            TP = TP + 1
        else:
            FP = FP + 1
    auc_temp += (TP+TP_pre) * (FP-FP_pre) / 2.0
    auc = auc_temp / (TP * FP)
    return auc


user_valid = train['user_id'].values[ind_te]
photo_valid = train['photo_id'].values[ind_te]
user2_valid = train_user_mean[ind_te]
label_valid = y[ind_te]
batch_size = 10240

def batch_generator(batch_size):
	for i in range(train_item_num//batch_size):
		base = i*batch_size
		end = base + batch_size
		photo_batch = train['photo_id'].values[ind_tr[base:end]]
		user_batch = train['user_id'].values[ind_tr[base:end]]
		mean_batch = train_user_mean[ind_tr[base:end]]
		label_batch = y[ind_tr[base:end]]
		yield user_batch, photo_batch, mean_batch, label_batch


def validation_generator(batch_size):
	for i in range(test_item_num//batch_size):
		base = i*batch_size
		end = base+batch_size
		photo_batch = train['photo_id'].values[ind_te[base:end]]
		user_batch = train['user_id'].values[ind_te[base:end]]
		mean_batch = train_user_mean[ind_te[base:end]]
		label_batch = y[ind_te[base:end]]
		yield user_batch, photo_batch, mean_batch, label_batch


class ncf(nn.Block):
    def __init__(self, **kwargs):
        super(ncf, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_u = nn.Embedding(nb_users,EMBEDDING_DIM_USER)#,weight_initializer=)
            self.emb_u.weight.set_data(nd.normal(shape=(nb_users,EMBEDDING_DIM_USER)))
            self.emb_u.grad_req = 'null'
            
            self.emb_u2 = nn.Embedding(nb_users,EMBEDDING_DIM_USER)#,weight_initializer=)
            
            self.emb_p = nn.Embedding(nb_photos,EMBEDDING_DIM_PHOTO)#,weight_initializer=)
            self.emb_p.weight.set_data(nd.normal(shape=(nb_photos,EMBEDDING_DIM_USER)))
            self.emb_p.grad_req = 'null'
            
            self.bn = nn.BatchNorm()
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(rate=0.25)
            self.dense1 = nn.Dense(units=128, activation='relu')
            self.dense2 = nn.Dense(units=1, activation='sigmoid')

    def forward(self, user_id, photo_id, user_id2):
        user_id = self.emb_u(user_id)
        user_id2 = self.emb_u2(user_id2)
        photo_id = self.emb_p(photo_id)
        user_id2 = self.flatten(user_id2)
        x = nd.concat(user_id, user_id2, photo_id, dim=1)
        x_1 = self.dense1(x)
        x_2 = self.bn(x_1)
        x_3 = self.dropout(x_2)
        res = self.dense2(x_3)
        res = res.reshape((1,-1))[0]
        return res


myncf = ncf()
myncf.initialize(ctx = ctx)
sigmoid_binary_cross_entropy_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = mx.gluon.Trainer(myncf.collect_params(), 'adam', {'learning_rate': 0.001})
# myncf.collect_params().reset_ctx(ctx)

for epoch in range(30):
	it = 0
	for user_id, photo_id, user_id2, label in batch_generator(batch_size):
		it += 1
		with autograd.record():
			pred = myncf(nd.array(user_id,ctx=ctx), nd.array(photo_id,ctx=ctx), nd.array(user_id2,ctx=ctx))
			label = nd.array(label).copyto(ctx)
			loss = sigmoid_binary_cross_entropy_loss(pred, label)
			if it%10 == 1:
				print('epoch %d it %d loss %f'%(epoch, it, nd.mean(loss).asnumpy()[0]))
		
		loss.backward()
		trainer.step(batch_size)
		if(it%100 == 0):
			print("judging...")
			scores = []
			accs = []
			for user_validation, photo_validation, user2_validation, label_validation in validation_generator(batch_size):
				pred_validation = myncf(nd.array(user_validation,ctx=ctx), nd.array(photo_validation,ctx=ctx), nd.array(user2_validation,ctx=ctx))
				score = scoreAUC(label_validation, pred_validation.asnumpy().tolist())
				scores.append(score)
				
			print('epoch %d auc score %f'%(epoch, np.mean(np.array(scores))))
			myncf.collect_params().save(filename='model/ncf_1/te')
			# myncf.collect_params().load(filename='model/ncf_1',ctx = ctx)

# user_test = test['user_id'].values
# photo_test = test['photo_id'].values
# user2_test = test_user_mean

