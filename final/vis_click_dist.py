from utils import *

# def load_res(model, epoch):
mdoel = "text_only"
epoch = 2
ptr = "%s_E%d.npy"%(mdoel, epoch)
pte = "ncf_%s_E%d.txt"%(mdoel, epoch)

columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
train_interaction.columns = columns
total_click = np.sum(train_interaction["click"])

train = np.load("res/%s"%ptr)
index = (-train).argsort()

time = np.array((train_interaction['time'] - np.min(train_interaction['time']))).astype(int)
click_bool = np.array(train_interaction["click"] == 1)
unclick_bool = np.logical_not(click_bool)

def click_dist_per_user(user_id):
	user_index = set(np.where(train_interaction["user_id"] == user_id)[0])
	clicked_per_user = list(set(index[:total_click]) & user_index)
	unclicked_per_user = list(set(index[total_click:]) & user_index)
	clicked = time[clicked_per_user].tolist()
	unclicked = time[unclicked_per_user].tolist()
	print("Total Interaction :%d\
		\nClicked           :%d\
		\nUnclicked         :%d"%(len(user_index), len(clicked), len(unclicked)))
	np.save("clicked.npy", clicked)
	np.save("unclicked.npy", unclicked)


def gt_dist(user_id):
	user_bool = train_interaction["user_id"] == user_id
	clicked_per_user = np.where(np.logical_and(user_bool, click_bool))[0]
	unclicked_per_user = np.where(np.logical_and(user_bool, unclick_bool))[0]
	clicked = time[clicked_per_user].tolist()
	unclicked = time[unclicked_per_user].tolist()
	print("Total Interaction :%d\
		\nClicked           :%d\
		\nUnclicked         :%d"%(np.sum(user_bool), len(clicked), len(unclicked)))
	np.save("clicked.npy", clicked)
	np.save("unclicked.npy", unclicked)

gt_dist(6666)
click_dist_per_user(1)

# import matplotlib.pyplot as plt
# from matplotlib import colors
# 
# pyplot.hist(clicked, alpha=0.5, label='clicked')
# pyplot.hist(unclicked, alpha=0.5, label='unclicked')
plt.legend()
# plt.show()
