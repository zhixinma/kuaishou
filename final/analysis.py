from utils import *

def load_train_interaction():
	columns = ['user_id','photo_id','click','like','follow','time','playing_time','duration_time']
	train_interaction = pd.read_table(PATH + 'train_interaction.txt',header=None)
	train_interaction.columns = columns
	return train_interaction


train_interaction = load_train_interaction()
# train_interaction["time"] = (train_interaction["time"] - np.min(train_interaction["time"])) // 60000
user_time_slice = train_interaction.groupby(["user_id","time"])["click"]
user_time_rate_slice = user_time_slice.sum() / user_time_slice.count()
user_rate_slice = user_time_rate_slice.groupby(['user_id'])
user_rate_std = np.array(user_rate_slice.std())
np.save("std.npy", user_rate_std)
np.mean(user_rate_std)
# count = np.array(user_time_slice.count())
# _sum = np.array(user_time_slice.sum())
# rate = 1.0 * _sum / count
