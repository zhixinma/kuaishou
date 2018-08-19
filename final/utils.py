# coding: utf-8
import os
from os import path
import csv
import numpy as np
import pandas as pd
##########################################################
PATH = "/home/share/mazhixin/kuaishou/final/"			##
# user_doc.csv 				photo_doc.csv				##
# train_face.txt 			test_face.txt				##
# train_text.txt 			test_text.txt				##
# train_interaction.txt 	test_interaction.txt		##
# preliminary_visual_train 	preliminary_visual_test		##
# train_photo_index_dic.npy test_photo_index_dic.npy	##
# VISUAL_PATH = "/home/share/zhaozheng/kuaishou/data/"	##
IS_TRAIN = True											##
##########################################################

def load_interaction(return_encoder=False):
	print("Loading Interaction Items ... ")
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns

	# double the last day data
	# max_time = np.max(train_interaction["time"])
	# min_time = np.min(train_interaction["time"])
	# boundary = min_time + (max_time - min_time) * 2 // 3
	# train_interaction = pd.concat([train_interaction, train_interaction[train_interaction['time'] >= boundary]])

	test_columns = ['user_id','photo_id','time','duration_time']
	test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
	test_interaction.columns = test_columns
	data = pd.concat([train_interaction,test_interaction])
	from sklearn.preprocessing import LabelEncoder

	le_user = LabelEncoder()
	if path.isfile(PATH+'user_reverse_fit.npy'):
		data['user_id'] = le_user.fit_transform(data['user_id'])
		user_reverse_fit = np.load(PATH+'user_reverse_fit.npy').item()
	else:
		keys = le_user.fit_transform(data['user_id'])
		user_reverse_fit = {}
		for key, value in zip(keys, data['user_id']):
			user_reverse_fit[key] = value
		data['user_id'] = keys
		np.save(PATH+'user_reverse_fit.npy', user_reverse_fit)

	le_photo = LabelEncoder()
	if path.isfile(PATH+'photo_reverse_fit.npy'):
		data['photo_id'] = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = np.load(PATH+'photo_reverse_fit.npy').item()
	else:
		keys = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = {}
		for key, value in zip(keys, data['photo_id']):
			photo_reverse_fit[key] = value
		data['photo_id'] = keys
		np.save(PATH+'photo_reverse_fit.npy', photo_reverse_fit)
 
	def generate_doc(df,name,concat_name):
		res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
		res.columns = [name,'%s_doc'%concat_name]
		return res
 	
	def load_doc(name,concat_name):
		dic = pd.read_csv("%s%s_doc.csv"%(PATH, name))
		dic['%s_id_doc'%name] = dic['%s_id_doc'%name].apply((lambda x : x.split(" ")))
		dic = dic.sort_values(['%s_id'%concat_name]).reset_index()
		return np.array(dic['%s_id_doc'%name])

	if path.isfile(PATH + 'user_doc.csv'):
		user_doc = load_doc('user','photo')
		photo_doc = load_doc('photo','user')
	else:
		user_doc = generate_doc(data,'photo_id','user_id')
		photo_doc = generate_doc(data,'user_id','photo_id')
		user_doc.to_csv(PATH+'user_doc.csv',index=False)
		photo_doc.to_csv(PATH+'photo_doc.csv',index=False)
		user_doc['user_id_doc'].to_csv(PATH+'user.adjlist',index=False)
		photo_doc['photo_id_doc'].to_csv(PATH+'photo.adjlist',index=False)

	if return_encoder:
		return data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit, le_user, le_photo

	return data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit

# Optimize on return value and user_doc format
def load_interaction_optimized():
	print("Loading Interaction Items ... ")
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns

	# # double the last day data
	# max_time = np.max(train_interaction["time"])
	# min_time = np.min(train_interaction["time"])
	# boundary = min_time + (max_time - min_time) * 2 // 3
	# train_interaction = pd.concat([train_interaction, train_interaction[train_interaction['time'] >= boundary]])

	test_columns = ['user_id','photo_id','time','duration_time']
	test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
	# train_interaction = train_interaction[train_interaction['follow'] == 0]
	test_interaction.columns = test_columns
	data = pd.concat([train_interaction,test_interaction])
	train_num = train_interaction.shape[0]
	test_num = test_interaction.shape[0]
	from sklearn.preprocessing import LabelEncoder
	
	le_user = LabelEncoder()
	if path.isfile(PATH+'user_reverse_fit.npy'):
		data['user_id'] = le_user.fit_transform(data['user_id'])
		user_reverse_fit = np.load(PATH+'user_reverse_fit.npy').item()
	else:
		keys = le_user.fit_transform(data['user_id'])
		user_reverse_fit = {}
		for key, value in zip(keys, data['user_id']):
			user_reverse_fit[key] = value
		data['user_id'] = keys
		np.save(PATH+'user_reverse_fit.npy', user_reverse_fit)

	le_photo = LabelEncoder()
	if path.isfile(PATH+'photo_reverse_fit.npy'):
		data['photo_id'] = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = np.load(PATH+'photo_reverse_fit.npy').item()
	else:
		keys = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = {}
		for key, value in zip(keys, data['photo_id']):
			photo_reverse_fit[key] = value
		data['photo_id'] = keys
		np.save(PATH+'photo_reverse_fit.npy', photo_reverse_fit)
 
	def generate_doc(df,name,concat_name):
		res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
		res.columns = [name,'%s_doc'%concat_name]
		return res
 	
	def load_doc(name,concat_name):
		dic = pd.read_csv("%s%s_doc.csv"%(PATH, name))
		dic['%s_id_doc'%name] = dic['%s_id_doc'%name].apply((lambda x : x.split(" ")))
		dic = dic.sort_values(['%s_id'%concat_name]).reset_index()
		return np.array(dic['%s_id_doc'%name])	

	if path.isfile(PATH + 'user_doc.csv'):
		user_doc = load_doc('user','photo')
		photo_doc = load_doc('photo','user')
	else:
		user_doc = generate_doc(data,'photo_id','user_id')
		photo_doc = generate_doc(data,'user_id','photo_id')
		user_doc.to_csv(PATH+'user_doc.csv',index=False)
		photo_doc.to_csv(PATH+'photo_doc.csv',index=False)
		user_doc['user_id_doc'].to_csv(PATH+'user.adjlist',index=False)
		photo_doc['photo_id_doc'].to_csv(PATH+'photo.adjlist',index=False)

	return data, user_doc, photo_doc, train_num, test_num, user_reverse_fit, photo_reverse_fit

# Optimized
def load_interaction_UT(ut_pad=False):
	print("Loading UT Interaction Items ... ")
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	test_columns = ['user_id','photo_id','time','duration_time']
	test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
	test_interaction.columns = test_columns
	data = pd.concat([train_interaction,test_interaction])

	from sklearn.preprocessing import LabelEncoder
	le_user = LabelEncoder()
	if path.isfile(PATH+'user_reverse_fit.npy'):
		data['user_id'] = le_user.fit_transform(data['user_id'])
		train_interaction['user_id'] = le_user.transform(train_interaction['user_id'])
		test_interaction['user_id']  = le_user.transform(test_interaction['user_id'])
		user_reverse_fit = np.load(PATH+'user_reverse_fit.npy').item()
	else:
		keys = le_user.fit_transform(data['user_id'])
		user_reverse_fit = {}
		for key, value in zip(keys, data['user_id']):
			user_reverse_fit[key] = value
		data['user_id'] = keys
		np.save(PATH+'user_reverse_fit.npy', user_reverse_fit)

	le_photo = LabelEncoder()
	if path.isfile(PATH+'photo_reverse_fit.npy'):
		data['photo_id'] = le_photo.fit_transform(data['photo_id'])
		train_interaction['photo_id'] = le_photo.transform(train_interaction['photo_id'])
		test_interaction['photo_id']  = le_photo.transform(test_interaction['photo_id'])
		photo_reverse_fit = np.load(PATH+'photo_reverse_fit.npy').item()
	else:
		keys = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = {}
		for key, value in zip(keys, data['photo_id']):
			photo_reverse_fit[key] = value
		data['photo_id'] = keys
		np.save(PATH+'photo_reverse_fit.npy', photo_reverse_fit)

	def generate_doc(df,name,concat_name):
	    res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
	    res.columns = [name,'%s_doc'%concat_name]
	    return res

	def load_doc(name,concat_name):
		dic = pd.read_csv("%s%s_doc.csv"%(PATH, name))
		dic['%s_id_doc'%name] = dic['%s_id_doc'%name].apply((lambda x : x.split(" ")))
		dic = dic.sort_values(['%s_id'%concat_name]).reset_index()
		return np.array(dic['%s_id_doc'%name])

	if path.isfile(PATH + 'user_doc.csv'):
		user_doc = load_doc('user','photo')
		photo_doc = load_doc('photo','user')
	else:
		user_doc = generate_doc(data,'photo_id','user_id')
		photo_doc = generate_doc(data,'user_id','photo_id')
		user_doc.to_csv(PATH+'user_doc.csv',index=False)
		photo_doc.to_csv(PATH+'photo_doc.csv',index=False)
		user_doc['user_id_doc'].to_csv(PATH+'user.adjlist',index=False)
		photo_doc['photo_id_doc'].to_csv(PATH+'photo.adjlist',index=False)

	def ut_padding(x):
		times = 32 // len(x) + 1
		x = (times * list(x))[:32]
		res = ' '.join([str(y) for y in x])
		return res

	if ut_pad:
		if path.isfile("%sut_ut_pad.npy"%PATH):
			ut_ut = np.load("%sut_ut_pad.npy"%PATH)
			ut_p = np.load("%sut_p_pad.npy"%PATH)
			ut_c = np.load("%sut_c_pad.npy"%PATH)
			train_num = 3700532
			test_num = 997412
		else:
			tr_ut = train_interaction.groupby(["user_id", "time"])
			te_ut = test_interaction.groupby(["user_id", "time"])
			train_UT_P = tr_ut["photo_id"].apply((lambda x :ut_padding(x))).reset_index()
			train_UT_C = tr_ut["click"].apply((lambda x :ut_padding(x))).reset_index()
			test_UT = te_ut["photo_id"].apply((lambda x :ut_padding(x))).reset_index()
			train_UT = pd.concat([train_UT_P, train_UT_C["click"]], axis=1)
			data_UT = pd.concat([train_UT, test_UT])
			train_num = train_UT.shape[0]
			test_num = test_UT.shape[0]
			ut_ut = np.array(data_UT[["user_id", "time"]])
			ut_photo = np.array(data_UT["photo_id"])
			ut_click = np.array(data_UT["click"])
			ut_click[train_num:] = '-1'
			ut_p = np.zeros((train_num+test_num, 32)).astype(int)
			ut_c = np.zeros((train_num+test_num, 32)).astype(int)
			for ut_it in xrange(train_num+test_num):
				ut_p[ut_it] = np.array(ut_photo[ut_it].split(" "))
				ut_c[ut_it] = np.array(ut_click[ut_it].split(" "))
			np.save("%sut_ut_pad.npy"%PATH,ut_ut)
			np.save("%sut_p_pad.npy"%PATH,ut_p)
			np.save("%sut_c_pad.npy"%PATH,ut_c)
	else:
		if path.isfile("%sut_ut.npy"%PATH):
			ut_ut = np.load("%sut_ut.npy"%PATH)
			ut_p = np.load("%sut_p.npy"%PATH)
			ut_c = np.load("%sut_c.npy"%PATH)
			train_num = 3700532
			test_num = 997412
		else:
			tr_ut = train_interaction.groupby(["user_id", "time"])
			te_ut = test_interaction.groupby(["user_id", "time"])
			train_UT_P = tr_ut["photo_id"].apply((lambda x :' '.join([str(y) for y in x]))).reset_index()
			train_UT_C = tr_ut["click"].apply((lambda x :' '.join([str(y) for y in x]))).reset_index()
			test_UT = te_ut["photo_id"].apply((lambda x :' '.join([str(y) for y in x]))).reset_index()
			train_UT = pd.concat([train_UT_P, train_UT_C["click"]], axis=1)
			data_UT = pd.concat([train_UT, test_UT])
			train_num = train_UT.shape[0]
			test_num = test_UT.shape[0]
			ut_ut = np.array(data_UT[["user_id", "time"]])
			ut_p = np.array(data_UT["photo_id"])
			ut_c = np.array(data_UT["click"])
			np.save("%sut_ut.npy"%PATH,ut_ut)
			np.save("%sut_p.npy"%PATH,ut_p)
			np.save("%sut_c.npy"%PATH,ut_c)

	return ut_ut, ut_p, ut_c, user_doc, photo_doc, train_num, test_num, user_reverse_fit, photo_reverse_fit

# RAW (rewrite when use)
def load_interaction_with_text(text_data):
	print("Loading Interaction Items with text ... ")
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	test_columns = ['user_id','photo_id','time','duration_time']
	test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
	test_interaction.columns = test_columns
	data = pd.concat([train_interaction,test_interaction])
	from sklearn.preprocessing import LabelEncoder
	
	le_user = LabelEncoder()
	if path.isfile(PATH+'user_reverse_fit.npy'):
		data['user_id'] = le_user.fit_transform(data['user_id'])
		user_reverse_fit = np.load(PATH+'user_reverse_fit.npy').item()
	else:
		keys = le_user.fit_transform(data['user_id'])
		user_reverse_fit = {}
		for key, value in zip(keys, data['user_id']):
			user_reverse_fit[key] = value
		data['user_id'] = keys
		np.save(PATH+'user_reverse_fit.npy', user_reverse_fit)
	
	le_photo = LabelEncoder()
	if path.isfile(PATH+'photo_reverse_fit.npy'):
		data['photo_id'] = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = np.load(PATH+'photo_reverse_fit.npy').item()
	else:
		keys = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = {}
		for key, value in zip(keys, data['photo_id']):
			photo_reverse_fit[key] = value
		data['photo_id'] = keys
		np.save(PATH+'photo_reverse_fit.npy', photo_reverse_fit)
 	
	def generate_doc(df,name,concat_name):
		res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
		res.columns = [name,'%s_doc'%concat_name]
		return res
 	
	def load_doc(name,concat_name):
		dic = pd.read_csv("%s%s_doc.csv"%(PATH, name))
		dic['%s_id_doc'%name] = dic['%s_id_doc'%name].apply((lambda x : x.split(" ")))
		dic = dic.sort_values(['%s_id'%concat_name]).reset_index()
		return np.array(dic['%s_id_doc'%name])
	
	#################################################################################################
	photo_with_text = []
	for photo_id in text_data:
		if text_data[photo_id] != [1]:
			photo_with_text.append(photo_id)
	train_interaction = train_interaction.loc[train_interaction['photo_id'].isin(photo_with_text)]
	test_interaction = test_interaction.loc[test_interaction['photo_id'].isin(photo_with_text)]
	photo_with_text = le_photo.transform(photo_with_text)
	data = data.loc[data['photo_id'].isin(photo_with_text)]
	#################################################################################################
	
	
	if path.isfile(PATH + 'user_doc.csv'):
		user_doc = load_doc('user','photo')
		photo_doc = load_doc('photo','user')
	else:
		user_doc = generate_doc(data,'photo_id','user_id')
		photo_doc = generate_doc(data,'user_id','photo_id')
		user_doc.to_csv(PATH+'user_doc.csv',index=False)
		photo_doc.to_csv(PATH+'photo_doc.csv',index=False)
		user_doc['user_id_doc'].to_csv(PATH+'user.adjlist',index=False)
		photo_doc['photo_id_doc'].to_csv(PATH+'photo.adjlist',index=False)
	
	return data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit

# RAW (rewrite when use)
def load_interaction_no_follow():
	print("Loading Interaction Items unfollowedw ... ")
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	train_interaction = train_interaction[train_interaction['follow'] == 0]
	test_columns = ['user_id','photo_id','time','duration_time']
	test_interaction = pd.read_table(PATH + 'test_interaction.txt',header=None)
	test_interaction.columns = test_columns

	data = pd.concat([train_interaction,test_interaction])
	from sklearn.preprocessing import LabelEncoder
	
	le_user = LabelEncoder()
	if path.isfile(PATH+'user_reverse_fit.npy'):
		data['user_id'] = le_user.fit_transform(data['user_id'])
		user_reverse_fit = np.load(PATH+'user_reverse_fit.npy').item()
	else:
		keys = le_user.fit_transform(data['user_id'])
		user_reverse_fit = {}
		for key, value in zip(keys, data['user_id']):
			user_reverse_fit[key] = value
		data['user_id'] = keys
		np.save(PATH+'user_reverse_fit.npy', user_reverse_fit)

	le_photo = LabelEncoder()
	if path.isfile(PATH+'photo_reverse_fit.npy'):
		data['photo_id'] = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = np.load(PATH+'photo_reverse_fit.npy').item()
	else:
		keys = le_photo.fit_transform(data['photo_id'])
		photo_reverse_fit = {}
		for key, value in zip(keys, data['photo_id']):
			photo_reverse_fit[key] = value
		data['photo_id'] = keys
		np.save(PATH+'photo_reverse_fit.npy', photo_reverse_fit)
 
	def generate_doc(df,name,concat_name):
		res = df.astype(str).groupby(name)[concat_name].apply((lambda x :' '.join(x))).reset_index()
		res.columns = [name,'%s_doc'%concat_name]
		return res
 	
	def load_doc(name,concat_name):
		dic = pd.read_csv("%s%s_doc.csv"%(PATH, name))
		dic['%s_id_doc'%name] = dic['%s_id_doc'%name].apply((lambda x : x.split(" ")))
		dic = dic.sort_values(['%s_id'%concat_name]).reset_index()
		return np.array(dic['%s_id_doc'%name])

	if path.isfile(PATH + 'user_doc.csv'):
		user_doc = load_doc('user','photo')
		photo_doc = load_doc('photo','user')
	else:
		user_doc = generate_doc(data,'photo_id','user_id')
		photo_doc = generate_doc(data,'user_id','photo_id')
		user_doc.to_csv(PATH+'user_doc.csv',index=False)
		photo_doc.to_csv(PATH+'photo_doc.csv',index=False)
		user_doc['user_id_doc'].to_csv(PATH+'user.adjlist',index=False)
		photo_doc['photo_id_doc'].to_csv(PATH+'photo.adjlist',index=False)

	return data, user_doc, photo_doc, train_interaction, test_interaction, user_reverse_fit, photo_reverse_fit

def load_text():
	print("Loading Text ... ")
	train_text = pd.read_table(PATH+"train_text.txt", header=None)
	test_text = pd.read_table(PATH+"test_text.txt", header=None)
	columns = ['photo_id', 'text_list']
	text_data = pd.concat([train_text, test_text])
	text_data.columns = columns
	text_data['text_list'] = text_data['text_list'].apply(lambda x:[int(s) for s in x.split(',')])

	vocabulary = []
	for word_list in text_data['text_list']:
	 	for word in word_list:
			vocabulary.append(word)			
	vocabulary = list(set(vocabulary))

	# zero means unknown
	word_index = {}
	for i in xrange(len(vocabulary)):
		word_index[vocabulary[i]] = i + 1
	text_data['text_list'] = text_data['text_list'].apply(lambda x:[word_index[s] for s in x])

	text = {}
	for photo_id, word_list in zip(text_data['photo_id'], text_data['text_list']):
		text[photo_id] = word_list

	return text, len(vocabulary)

def load_userep():
	tar = "%suserep.npy"%PATH
	if path.isfile(tar):
		userep = np.load(tar)
	else:
		raw_tar = "%suserep.txt"%PATH
		userep = np.zeros((37821, 2048))
		it = 0
		with open(raw_tar, "rb") as userep_file:
			for row in userep_file.readlines():
				if it%1000 == 0:
					print("%d/%d"%(it, 37820))
				comps = row.split(":")
				userep[int(comps[0])-1] = np.array(comps[1].split(" "))
				it += 1
		np.save(tar, userep)
	return userep

# get the largest one per photo
def load_face():
	print("Loading Face ... ")
	tar = "%sface_dic.npy"%PATH
	if path.isfile(tar):
		face = np.load(tar).item()
	else:
		import ast

		train_face = pd.read_table(PATH+"train_face.txt", header=None)
		test_face = pd.read_table(PATH+"test_face.txt", header=None)
		columns = ['photo_id', 'face_info']
		face_data = pd.concat([train_face, test_face])
		face_data.columns = columns
		face_data['face_info'] = face_data['face_info'].apply(lambda x:ast.literal_eval(x))

		# 人脸占整个图片的比例、人脸性别（0:女性，1:男性）、人脸年龄、相貌属性
		max_score = 0
		min_score = 100000
		max_age = 0
		min_age = 100
		face = {}
		for photo_id, face_info in zip(face_data['photo_id'], face_data['face_info']):
			max_age = np.max([np.max(np.array(face_info)[:,2]), max_age])
			min_age = np.min([np.min(np.array(face_info)[:,2]), min_age])
			max_score = np.max([np.max(np.array(face_info)[:,3]), max_score])
			min_score = np.min([np.min(np.array(face_info)[:,3]), min_score])
			face_info = np.array(face_info)
			i_max = np.argsort(-face_info[:,0])[0]
			face[photo_id] = face_info[i_max].tolist()
		print("MAX")
		np.save(tar ,face)


	return face

def load_visual():
	print("Loading Visual ... ")
	if path.isfile(PATH+'test.npy'):
		test_visual = np.load(PATH+'test.npy')
		train_visual = np.load(PATH+'train.npy')
	else:
		print("Error, Visual file not found")
		# import gc
		# test_num = 1643866
		# train_num = 7560366
		# train_photo_list = os.listdir(PATH+'final_visual_train')
		# test_photo_list = os.listdir(PATH+'final_visual_test')
		# ###################################################################################
		# i = 0
		# test_visual = np.zeros((1,2048))
		# for batch_num in xrange(1644):
		# 	del test_visual
		# 	gc.collect()
		# 	test_visual = np.zeros((1,2048))
		# 	start = batch_num*1000
		# 	end = start + 1000
		# 	tmp_list = test_photo_list[start:end]
		# 	for photo_id in tmp_list:
		# 		test_visual = np.concatenate((test_visual, np.load('%sfinal_visual_test/%s'%(PATH, photo_id))))
		# 		if i%100 == 0:
		# 			print("It: %d of Test Batch 1"%i)
		# 		i += 1
		# 	np.save('%stest_visual/test_B1_%d.npy'%(PATH, batch_num), test_visual[1:])
		# test_visual = np.zeros((1,2048))
		# for batch_num in xrange(1644):
		# 	test_visual = np.concatenate((test_visual, np.load('%stest_visual/test_B1_%d.npy'%(PATH, batch_num))))
		# 	if i%100 == 0:
		# 		print("It: %d of Test Batch 1"%i)
		# 		i += 1
		# 	np.save('%stest.npy'%(PATH), test_visual[1:])
		# ###################################################################################
		# i = 0
		# train_visual = np.zeros((1,2048))
		# for batch_num in xrange(1644):
		# 	del train_visual
		# 	gc.collect()
		# 	train_visual = np.zeros((1,2048))
		# 	start = batch_num*1000
		# 	end = start + 1000
		# 	tmp_list = train_photo_list[start:end]
		# 	for photo_id in tmp_list:
		# 		train_visual = np.concatenate((train_visual, np.load('%sfinal_visual_train/%s'%(PATH, photo_id))))
		# 		if i%100 == 0:
		# 			print("It: %d of Train Batch 1"%i)
		# 		i += 1
		# 	np.save('%strain_visual/train_B1_%d.npy'%(PATH, batch_num), train_visual[1:])
		# train_visual = np.zeros((1,2048))	
		# for batch_num in xrange(1644):
		# 	train_visual = np.concatenate((train_visual, np.load('%strain_visual/train_B1_%d.npy'%(PATH, batch_num))))
		# 	if i%100 == 0:
		# 		print("It: %d of Train Batch 1"%i)
		# 		i += 1
		# 	np.save('%strain.npy'%(PATH), train_visual[1:])

	if path.isfile(PATH+'train_photo_index_dic.npy'):
		train_photo_to_index = np.load(PATH+'train_photo_index_dic.npy').item()
		test_photo_to_index = np.load(PATH+'test_photo_index_dic.npy').item()
	else:
		train_photo_list = os.listdir(PATH+'final_visual_train')
		test_photo_list = os.listdir(PATH+'final_visual_test')

		train_photo_to_index = {}
		for i in xrange(len(train_photo_list)):
			train_photo_to_index[int(train_photo_list[i])] = i

		test_photo_to_index = {}
		for i in xrange(len(test_photo_list)):
			test_photo_to_index[int(test_photo_list[i])] = i

		np.save(PATH+'train_photo_index_dic.npy', train_photo_to_index)
		np.save(PATH+'test_photo_index_dic.npy', test_photo_to_index)

	return train_visual, test_visual, train_photo_to_index, test_photo_to_index

def load_topic(text_data, TOPIC_NUM, ITER_NUM):
	if path.isfile("topics/topic_word_%d_%d.npy"%(TOPIC_NUM, ITER_NUM)):
		topic_word = np.load("topics/topic_word_%d_%d.npy"%(TOPIC_NUM, ITER_NUM))
		doc_topic = np.load("topics/doc_topic_%d_%d.npy"%(TOPIC_NUM, ITER_NUM))
	else:
		
		import lda
		
		lda_model = lda.LDA(n_topics=TOPIC_NUM, n_iter=ITER_NUM, random_state=1)
		docs = list(text_data.values())
		docs = np.array(docs)
		indces = []
		for i in xrange(len(docs)):
			if (docs[i] == 0).all():
				indces.append(i)

		docs = np.delete(docs, indces, 0)
		lda_data = np.zeros((1,233243)).astype(int)
		for doc in docs:
			lda_data[0][doc] += 1

		lda_data[0][0] = 0
		lda_model.fit_transform(lda_data)

		topic_word = lda_model.topic_word_
		doc_topic = lda_model.doc_topic_

		np.save("topics/topic_word_%d_%d.npy"%(TOPIC_NUM, ITER_NUM), topic_word)
		np.save("topics/doc_topic_%d_%d.npy"%(TOPIC_NUM, ITER_NUM), doc_topic)

	return topic_word, doc_topic

def generate_prof():
	if path.isfile(PATH+"prof.npy"):
		profile_matrix = np.load(PATH+'prof.npy').item()
	else:
		click_list = {}
		unclick_list = {}
		profile_matrix = {}
		length = []
		columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
		train_interaction = pd.read_table(PATH+'train_interaction.txt',header=None)
		train_interaction.columns = columns
		user_list = train_interaction['user_id'].unique()
		for user_id in user_list:
			click_list[user_id] = []
			unclick_list[user_id] = []
			profile_matrix[user_id] = []
		for user_id, photo_id, click in zip(train_interaction['user_id'], train_interaction['photo_id'], train_interaction['click']):
			if click == 1:
				click_list[user_id].append(photo_id)
			else:
				unclick_list[user_id].append(photo_id)
		for user_id in user_list:
			profile_matrix[int(user_id)] = click_list[user_id]+unclick_list[user_id]
			length.append(len(click_list[user_id]))
		print('max:%d'%np.max(length))
		print('min:%d'%np.min(length))
		print('mean:%d'%np.mean(length))
		np.save(PATH+'prof.npy', profile_matrix) 
	return profile_matrix

def get_user_text(profile_matrix, text_data):
	if path.isfile(PATH+'user_text.npy'):
		user_text = np.load(PATH+'user_text.npy').item()
	else:
		user_text = {}
		for user_id in profile_matrix:
			user_text[user_id] = []
		for user_id in profile_matrix:
			for photo_id in profile_matrix[user_id]:
				user_text[user_id] += text_data[photo_id]
		np.save(PATH+'user_text.npy', user_text)
	return user_text

def get_user_visual(profile_matrix, train_photo_to_index):
	if path.isfile(PATH+'visual_profile.npy'):
		user_visual = np.load(PATH+'visual_profile.npy').item()
	else:
		user_visual = {}
		for user_id in profile_matrix:
			profile = []
			for photo_id in profile_matrix[user_id]:			
				profile.append(train_photo_to_index[str(photo_id)])
			user_visual[user_id] = np.array(profile[:100]).astype(int)
		np.save(PATH+'visual_profile.npy', user_visual)
	return user_visual

def scoreAUC(labels, probs):
    i_sorted = sorted(range(len(probs)),key=lambda i: probs[i], reverse=True)
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

def compute_preplexity(topic_word ,doc_topic):
	doc = np.load(PATH+"doc_bag_of_words.npy")
	prob = 0
	word_num = doc.shape[0]
	topic_num = doc_topic.shape[1]
	vocab_size = topic_word.shape[1]
	# Build probobility matrix
	topic_word = topic_word.transpose()
	print("***************************************")
	print("TOPIC  WORD: %d*%d"%(topic_word.shape[0], topic_word.shape[1]))
	print("DOC   TOPIC: %d*%d"%(doc_topic.shape[0], doc_topic.shape[1]))
	prob_matrix = np.multiply(topic_word, doc_topic)
	print("    ↑      : MULTIPLY (element wise)")
	print("PROB MATRIX: %d*%d"%(prob_matrix.shape[0], prob_matrix.shape[1]))
	prob_matrix = prob_matrix.transpose()
	print("PROB MATRIX: TRANSPOSE")
	print("PROB MATRIX: %d*%d"%(prob_matrix.shape[0], prob_matrix.shape[1]))
	doc = doc.reshape((-1,1))
	print("        DOC: %d*%d"%(doc.shape[0], doc.shape[1]))
	topic_sum_prob = np.matmul(prob_matrix, doc)
	print("    ↑      : MATMUL (dot)")
	print("TOPIC SUM P: %d*%d"%(topic_sum_prob.shape[0], topic_sum_prob.shape[1]))
	prob = topic_sum_prob.sum()
	print("SUM        : %f"%prob)
	prob = np.log(prob)
	print("LOG        : %f"%prob)
	prob = -1 * prob / word_num
	print("-prob / N  : %f"%prob)
	prob = np.exp(prob)
	print("EXP        : %f"%prob)
	print("***************************************")
	return prob
