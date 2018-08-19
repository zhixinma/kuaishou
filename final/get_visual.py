# coding: utf-8
import os
from os import path
import numpy as np

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

import gc
test_num = 1643866
train_num = 7560366

if not IS_TRAIN:
	test_photo_list = os.listdir(PATH+'final_visual_test')

	i = 0
	test_visual = np.zeros((1,2048))
	for batch_num in xrange(1644):
		del test_visual
		gc.collect()
		test_visual = np.zeros((1,2048))
		start = batch_num*1000
		end = start + 1000
		tmp_list = test_photo_list[start:end]
		for photo_id in tmp_list:
			test_visual = np.concatenate((test_visual, np.load('%sfinal_visual_test/%s'%(PATH, photo_id))))
			if i%100 == 0:
				print("It: %d of Test Batch 1"%i)
			i += 1
		np.save('%stest_visual/test_B1_%d.npy'%(PATH, batch_num), test_visual[1:])

	test_visual = np.zeros((1,2048))
	for batch_num in xrange(1644):
		test_visual = np.concatenate((test_visual, np.load('%stest_visual/test_B1_%d.npy'%(PATH, batch_num))))
		if i%100 == 0:
			print("It: %d of Test Batch 2"%i)
		i += 1
	np.save('%stest.npy'%(PATH), test_visual[1:])

else:
	train_photo_list = os.listdir(PATH+'final_visual_train')
	
	train_visual = np.zeros((1,2048))
	for batch_num in xrange(7561):
		tar = '%strain_visual/train_B1_%d.npy'%(PATH, batch_num)
	
		if path.isfile(tar):
			print(tar)
			continue
		else:
			del train_visual
	
		gc.collect()
		train_visual = np.zeros((1,2048))
		start = batch_num*1000
		end = start + 1000
		tmp_list = train_photo_list[start:end]
		for photo_id in tmp_list:
			train_visual = np.concatenate((train_visual, np.load('%sfinal_visual_train/%s'%(PATH, photo_id))))
		np.save(tar, train_visual[1:])
		print(tar)
	
	
	train_visual = np.zeros((1,2048))	
	for batch_num in xrange(7561):
		tar = '%strain_visual/train_B1_%d.npy'%(PATH, batch_num)
		train_visual = np.concatenate((train_visual, np.load(tar)))
		print("Batch 2: %04d/7561"%(batch_num))
		if batch_num%200 == 0:
			np.save('%strain.npy'%(PATH), train_visual[1:])
			print("%d Saved"%batch_num)
	np.save('%strain.npy'%(PATH), train_visual[1:])

	# part1 = np.load("%strain_part1.npy"%PATH)
	# print("PART 1 finished")
	# part2 = np.load("%strain_part2.npy"%PATH)
	# print("PART 2 finished")
	# full = np.concatenate((part1, part2))
	# del part1
	# del part2
	# gc.collect()
	# print("Concatenated")
	# print(full.shape)
	# np.save('%strain_visual.npy'%(PATH), full)

full = np.load('%strain_visual.npy'%(PATH))
train_photo_list = np.array(os.listdir(PATH+'final_visual_train')).astype(int)
for i in xrange(len(full)):
	tar = '%sfinal_visual_train/%d'%(PATH, train_photo_list[i])
	row = np.load(tar)
	if i%100 == 0:
		print("processing %dth item"%i)
	if (full[i] == row).all():
		del row
		p = gc.collect()
		continue
	else:
		print("The %dth is wrong"%i)
		break
