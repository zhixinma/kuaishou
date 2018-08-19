from utils import *

def grid_search_3():

	############################################ LOAD DATA #######################################
	tr_i = np.load("%schen_it_on_train.npy"%PATH)
	tr_v = np.load("res/ncf_it_interaction_on_train.npy")
	tr_vf = np.load("%sgrid_on_train.npy"%PATH)
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	click = np.array(train_interaction["click"])
	##############################################################################################

	################# Partition #################
	import random
	train_size = train_interaction.shape[0]
	shuffled_index = range(train_size)
	random.shuffle(shuffled_index)
	boundary = int(0.9*train_size)
	tr_index = shuffled_index[:boundary]
	val_index = shuffled_index[boundary:]
	val_i = tr_i[val_index]
	val_v = tr_v[val_index]
	val_vf = tr_vf[val_index]
	label = click[val_index]
	#############################################

	################# Search ####################
	# 0.1:54+1
	# 0.01:5049+1
	grid = 0.1
	grid_num = int(1/grid)
	model_num = 3

	total_it = 0
	for x in xrange(grid_num+1):
		total_it += x
	grid_res = np.zeros((total_it,model_num+1))

	it = 0
	for i in xrange(grid_num):
		w_1 = i*grid
		for j in xrange(grid_num-i):
			w_2 = j*grid
			k = grid_num-i-j
			w_3 = k*grid
			res = w_1*val_i+w_2*val_v+w_3*val_vf
			score = scoreAUC(label.tolist(), res.tolist())
			grid_res[it] = np.array([w_1, w_2, w_3, score])
			print("Searching %d/%d: %f %f %f -> %f"%(it, total_it, w_1, w_2, w_3, score))
			it += 1
	np.save("%sgrid_res_%d_%f.npy"%(PATH, model_num, grid), grid_res)
	#############################################

def grid_search_7():

	############################################ LOAD DATA #######################################
	tr_i = np.load("%sres/ncf_i_on_train.npy"%PATH)
	tr_v = np.load("%sres/ncf_v_on_train.npy"%PATH)
	tr_f = np.load("%sres/ncf_f_on_train.npy"%PATH)
	tr_t = np.load("%sres/ncf_t_on_train.npy"%PATH)
	tr_it = np.load("%sres/ncf_it_on_train.npy"%PATH)
	tr_vt = np.load("%sres/ncf_vt_on_train.npy"%PATH)
	tr_vf = np.load("%sres/ncf_vf_on_train.npy"%PATH)
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	click = np.array(train_interaction["click"])
	##############################################################################################

	################# Partition #################
	import random
	train_size = train_interaction.shape[0]
	shuffled_index = range(train_size)
	random.shuffle(shuffled_index)
	boundary = int(0.9*train_size)
	tr_index = shuffled_index[:boundary]
	val_index = shuffled_index[boundary:]
	val_i = tr_i[val_index]
	val_v = tr_v[val_index]
	val_f = tr_f[val_index]
	val_t = tr_t[val_index]
	val_it = tr_it[val_index]
	val_vt = tr_vt[val_index]
	val_vf = tr_vf[val_index]

	label = click[val_index]
	#############################################

	################# Search ####################
	# 0.3 0.2 0.0 0.0 0.1 0.3 0.1

	grid = 0.1
	grid_num = int(1/grid)
	model_num = 7

	grid_res = np.zeros((5004,model_num+1))

	it = 0
	for i in xrange(grid_num):
		w_1 = i*grid
		for j in xrange(grid_num-i):
			w_2 = j*grid
			for k in xrange(grid_num-i-j):
				w_3 = k*grid
				for m in xrange(grid_num-i-j-k):
					w_4 = m*grid
					for n in xrange(grid_num-i-j-k-m):
						w_5 = n*grid
						for p in xrange(grid_num-i-j-k-m-n):
							w_6 = p*grid
							q = grid_num-i-j-k-m-n-p
							w_7 = q*grid
							res = w_1*val_i+w_2*val_v+w_3*val_f+w_4*val_t+w_5*val_it+w_6*val_vt+w_7*val_vf
							score = scoreAUC(label, res)
							grid_res[it] = np.array([w_1, w_2, w_3, w_4, w_5, w_6, w_7, score])
							print("Searching %d/5004: %.04f %.04f %.04f %.04f %.04f %.04f %.04f -> %f"%(it, w_1, w_2, w_3, w_4, w_5, w_6, w_7, score))
							it += 1
		np.save("%sgrid_res_%d_%f.npy"%(PATH, model_num, grid), grid_res)
	#############################################

def fusion_res():
	columns = ["user_id", "photo_id", "p"]
	i = pd.read_table('res/ncf_i_submission.txt',header=None)
	v = pd.read_table('res/ncf_v_submission.txt',header=None)
	f = pd.read_table('res/ncf_f_submission.txt',header=None)
	t = pd.read_table('res/ncf_t_submission.txt',header=None)
	it = pd.read_table('res/ncf_it_submission.txt',header=None)
	vt = pd.read_table('res/ncf_vt_submission.txt',header=None)
	vf = pd.read_table('res/ncf_vf_submission.txt',header=None)
	grid = pd.read_table('res/grid_submission.txt',header=None)
	i.columns = columns
	v.columns = columns
	f.columns = columns
	t.columns = columns
	it.columns = columns
	vt.columns = columns
	vf.columns = columns
	grid.columns = columns

	# 0.3 0.2 0.0 0.0 0.1 0.3 0.1

	res = pd.concat([i[["user_id", "photo_id"]], (0.3*i["p"] + 0.2*v["p"] + 0.1*it["p"] + 0.3*vt["p"] + 0.1*vf["p"])], axis=1)
	res.columns = columns

	with open('res/grid_submission.txt','wb') as submission:
		for a,b,c in zip(res['user_id'], res["photo_id"], res["p"]):
			submission.write("%d\t%d\t%f\n"%(a,b,c))

	it = pd.read_table('res/ncf_it_submission.txt',header=None)
	vt = pd.read_table('res/ncf_vt_submission.txt',header=None)
	vf = pd.read_table('res/ncf_vf_submission.txt',header=None)
	it.columns = columns
	vt.columns = columns

	# grid = 0.3*tr_i + 0.2*tr_v + 0.1*tr_it + 0.3*tr_vt + 0.1*tr_vf
	
	# size = len(i)
	# var = np.concatenate((np.reshape(i["p"], (1,size))[0], np.reshape(v["p"], (1,size))[0])).reshape((2, size))
	# var = np.concatenate((var,np.array(t["p"]).reshape(1,size)))
	# var = np.concatenate((var,np.array(it["p"]).reshape(1,size)))
	# var = np.concatenate((var,np.array(vt["p"]).reshape(1,size)))
	# var = np.concatenate((var,np.array(f["p"]).reshape(1,size)))
	# 
	# cor = np.corrcoef(var)
	# print(cor)

def fusion_weight_training():
	
	import mxnet as mx
	from mxnet import nd
	import numpy as np
	from mxnet.gluon import nn, rnn
	from mxnet import autograd

	class fusion(nn.Block):
	    def __init__(self, model_num, **kwargs):
	        super(fusion, self).__init__(**kwargs)
	        with self.name_scope():
	        	self.W = self.params.get('param_test',shape=(model_num,))
	    def forward(self, models):
	    	res = nd.dot(models, nd.softmax(self.W.data()).reshape((model_num, 1)))
	        return res

	def load():
		print("Loading Data")
		columns = ["user_id", "photo_id", "p"]
		i = pd.read_table('res/ncf_i_submission.txt',header=None)
		i.columns = columns
		v = pd.read_table('res/ncf_v_submission.txt',header=None)
		v.columns = columns
		# t = pd.read_table('res/ncf_t_submission.txt',header=None)
		# t.columns = columns
		# f = pd.read_table('res/ncf_f_submission.txt',header=None)
		# f.columns = columns
		it = pd.read_table('res/ncf_it_submission.txt',header=None)
		it.columns = columns
		vt = pd.read_table('res/ncf_vt_submission.txt',header=None)
		vt.columns = columns
		vf = pd.read_table('res/ncf_vf_submission.txt',header=None)
		vf.columns = columns
		size = i.shape[0]
		test = np.concatenate((np.reshape(i["p"], (1,size))[0], np.reshape(v["p"], (1,size))[0])).reshape((2, size))
		# test = np.concatenate((test,np.array(t["p"]).reshape(1,size)))
		# test = np.concatenate((test,np.array(f["p"]).reshape(1,size)))
		test = np.concatenate((test,np.array(it["p"]).reshape(1,size)))
		test = np.concatenate((test,np.array(vt["p"]).reshape(1,size)))
		test = np.concatenate((test,np.array(vf["p"]).reshape(1,size)))
		tr_i = np.load("%sres/ncf_i_on_train.npy"%PATH)
		tr_v = np.load("%sres/ncf_v_on_train.npy"%PATH)
		# tr_t = np.load("%sres/ncf_t_on_train.npy"%PATH)
		# tr_f = np.load("%sres/ncf_f_on_train.npy"%PATH)
		tr_it = np.load("%sres/ncf_it_on_train.npy"%PATH)
		tr_vt = np.load("%sres/ncf_vt_on_train.npy"%PATH)
		tr_vf = np.load("%sres/ncf_vf_on_train.npy"%PATH)
		tr_size = tr_i.shape[0]
		train = np.concatenate((tr_i.reshape((1, tr_size)), tr_v.reshape((1, tr_size))))
		# train = np.concatenate((train, tr_t.reshape((1, tr_size))))
		# train = np.concatenate((train, tr_f.reshape((1, tr_size))))
		train = np.concatenate((train, tr_it.reshape((1, tr_size))))
		train = np.concatenate((train, tr_vt.reshape((1, tr_size))))
		train = np.concatenate((train, tr_vf.reshape((1, tr_size))))
		columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
		train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
		train_interaction.columns = columns
		click = np.array(train_interaction["click"])
		test_columns = ['user_id','photo_id','time','duration_time']
		test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
		test_interaction.columns = test_columns
		info = test_interaction[["user_id", "photo_id"]]
		return train, test, click, info

	############# Load Data #############
	train, test, click, info = load()
	model_num = train.shape[0]
	test_size = test.shape[1]
	train_size = train.shape[1]
	#####################################

	############## Index ################
	import random
	shuffled_index = range(train_size)
	random.shuffle(shuffled_index)
	boundary = int(0.9*train_size)
	tr_index = shuffled_index[:boundary]
	val_index = shuffled_index[boundary:]
	te_index = range(test_size)
	#####################################

	#####################################
	batch_size = 128
	def batch_generator(data, index):
		total_it = len(index) // batch_size + 1
		for i in xrange(total_it):
			start = i*batch_size
			end = start + batch_size
			index_batch = index[start:end]
			data_batch = data[:,index_batch].transpose()
			label_batch = click[index_batch]
			# print("data:", data_batch.shape,
			# 	  "label", label_batch.shape)
			yield data_batch, label_batch


	############### Train ###############

	gpu_num = 0
	ctx = mx.gpu(gpu_num)
	net = fusion(model_num)
	net.initialize(ctx=ctx)
	loss_func = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
	trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

	total_epoch = 20
	IS_TRAIN = True
	model_name = 'fusion_5'

	max_auc = 0
	if IS_TRAIN:
		for epoch in range(total_epoch):
			it = 0
			total_it = len(tr_index) / batch_size + 1
			for data, label in batch_generator(train, tr_index):
				with autograd.record():
					pred = net(nd.array(data, ctx=ctx))
					loss = loss_func(pred, nd.array(label,ctx=ctx))
					if it%100 == 0:
						print('epoch %d it %d/%d loss %f max_auc %f'%(epoch, it, total_it, nd.mean(loss).asnumpy()[0], max_auc))
				it += 1
				loss.backward()
				trainer.step(batch_size)
			labels = []
			preds = []
			it = 0
			total_it = len(val_index) / batch_size + 1
			for data, label in batch_generator(train, val_index):
				pred = net(nd.array(data, ctx=ctx))
				preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
				labels += np.array(label).reshape((1, -1)).tolist()[0]
				it += 1
				if it%100 == 0:
					print('Calculate: it %d/%d'%(it, total_it))
			score = scoreAUC(labels, preds)	
			print('epoch %d auc score %f'%(epoch, score))
			if score > max_auc:
				max_auc = score
				net.collect_params().save('model/%s.params'%model_name)
				print('Model saved')
	# else:
	# 	print("Max AUC Score", max_auc)
	# 	net.collect_params().load('model/%s.params'%model_name, ctx=ctx)
	# 	preds = []
	# 	for data, label in batch_generator(test, te_index):
	# 		pred = net(nd.array(data, ctx=ctx))
	# 		preds += pred.asnumpy().reshape((1,-1)).tolist()[0]
	# 	with open('res/%s_submission.txt'%model_name,'wb') as submission:
	# 		for user_id, photo_id, propability in zip(info['user_id'], info['photo_id'], preds):
	# 			submission.write("%d\t%d\t%f\n"%(user_id, photo_id, propability))

grid_search_3()