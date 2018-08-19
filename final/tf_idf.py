from utils import *
import numpy as np

def count_word(train_text, test_text):
	words = []
	count = 0
	for u_id in train_text:
		for word in train_text[u_id]:
			if word not in words:
				words.append(word)
				count += 1
	for u_id in test_text:
		for word in test_text[u_id]:
			if word not in words:
				words.append(word)
				count += 1

	user_dic = {}
	for word in words:
		user_dic[word] = 0

	for u_id in train_text:
		for word in train_text[u_id]:
			user_dic[word] += 1
	for u_id in test_text:
		for word in test_text[u_id]:
			user_dic[word] += 1

	length = []
	for word in user_dic:
		length.append(user_dic[word])

	return count, np.array(length)


def main():
	length_test = get_info(test_text)
	length_train = get_info(train_text)
	count, length = count_word(train_text, test_text)

	print('******************* Train **********************')
	print('max word number per user:  %d'%np.max(length_test))
	print('min word number per user:  %d'%np.min(length_test))
	print('mean word number per user: %d'%np.mean(length_test))
	print('******************* Test  **********************')
	print('max word number per user:  %d'%np.max(length_train))
	print('min word number per user:  %d'%np.min(length_train))
	print('mean word number per user: %d'%np.mean(length_train))
	print('************************************************')
	print('total number of words:     %d'%count)
	print('max user number per word:  %d'%np.max(length))
	print('min user number per word:  %d'%np.min(length))
	print('mean user number per word: %d'%np.mean(length))
	print('************************************************')


# Interaction
# data, user_doc, photo_doc, train_interaction, test_interaction = load_interaction()
# train_interaction['photo_id'].nunique()
# test_interaction['photo_id'].nunique()
# 
# Text
# train_text, test_text, train_text_length, test_text_length = load_text()
# np.max(train_text_length)