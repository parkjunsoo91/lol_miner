import json
import sqlite3
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy

from experiments import *
from datetime import datetime, date
import time
import calendar
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
import sqlite3
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import pandas as pd

def userdata_opgg():
	#build data of the form
	#[delta(month), monthly diversity, month]
	

	histories = fetch_all_user_history(seasons=[9, 10, 11])
	#histories = [{'aid', 'tier', 'rank', 'matchlist':{'matches', totalGames}, 'region'}]
	history_with_opgg = []
	for u in histories:
		recent_history = get_recent_history(u['aid'], region=u['region'])
		
		#pick only those with opgg history
		if recent_history == None:
			continue
		
		u_valid = u
		numgames = u['matchlist']['totalGames']
		mlist = u['matchlist']['matches']
		for m in recent_history:
			mstring = str(m['month'])
			if mstring != "2018.02":
				mstring = mstring + '.30'
			else:
				mstring = mstring + '.28'
			m['month'] = mstring

		u_valid['recent'] = recent_history
		history_with_opgg.append(u_valid)

	print("people:", len(history_with_opgg))
	return history_with_opgg

def get_tierwise_userdata():
	history = userdata_opgg()
	tierwise_data = [[],[],[],[],[],[],[]]
	for h in history:
		opgg_record = h['recent']
		starting_tier = opgg_record[0]['tier'] - 1
		tierwise_data[starting_tier].append(h)
	return tierwise_data

def lasso_analysis():
	whole_data = userdata_opgg()
	print("for everyone:")
	X = extract_features(whole_data)
	corr_matrix = X.corr()
	corr_matrix.to_csv('whole.csv')
	'''
	train, test = train_test_split(X, test_size=0.2, random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(X[:,2:], X[:,0], test_size=0.2, random_state=0)
	regressor = MLPRegressor()
	regressor.fit(X_train, y_train)
	s1 = regressor.score(X_test, y_test)
	print(s1)

	X_train, X_test, y_train, y_test = train_test_split(X[:,2:], X[:,1], test_size=0.2, random_state=0)
	regressor = MLPRegressor()
	regressor.fit(X_train, y_train)
	s1 = regressor.score(X_test, y_test)
	print(s1)
	'''
	tierwise_data = get_tierwise_userdata()
	for i in range(7):
		print("for rank :", i)
		X = extract_features(tierwise_data[i])
		corr_matrix = X.corr()
		corr_matrix.to_csv('rank'+str(i)+'.csv')
		#feature_analysis(X)
		'''
		scaler = StandardScaler()
		scaler.fit(X)
		X = scaler.transform(X)

		X_train, X_test, y_train, y_test = train_test_split(X[:,2:], X[:,0], test_size=0.2, random_state=0)
		regressor = MLPRegressor()
		regressor.fit(X_train, y_train)
		s1 = regressor.score(X_test, y_test)
		print(s1)

		X_train, X_test, y_train, y_test = train_test_split(X[:,2:], X[:,1], test_size=0.2, random_state=0)
		regressor = MLPRegressor()
		regressor.fit(X_train, y_train)
		s1 = regressor.score(X_test, y_test)
		print(s1)
		'''

def extract_features(userdata):
	#userdata: [{aid, ...}]
	X = []
	for u in userdata:
		opgg_record = u['recent']
		matches = u['matchlist']['matches']
		first_month_data = opgg_record[0]
		first_month = int(first_month_data['month'][5:7])

		last_month_data = {}
		for r in opgg_record:
			if r['month'] <= "2018.01.30":
				last_month_data = r;
		last_month = int(last_month_data['month'][5:7])
		if last_month == 1:
			last_month = 13
		delta_month = last_month - first_month
		ts_start = month_to_timestamp(first_month_data['month'])
		ts_end = month_to_timestamp(last_month_data['month'])
		games_within_period = [m for m in matches if ts_start < m['timestamp']//1000 and m['timestamp']//1000 < ts_end]
		#play patterns
		num_games = len(games_within_period)
		if num_games == 0:
			continue
		monthly_games = num_games/delta_month;

		#this part is for checking switchers
		monthly_mosts = [0,0,0,0,0] #aggregate of monthly mosts 
		for i in range(len(opgg_record)-1):
			ts_start = month_to_timestamp(opgg_record[i]['month'])
			ts_end = month_to_timestamp(opgg_record[i+1]['month'])
			games_in_month = [m for m in matches if ts_start < m['timestamp']//1000 and m['timestamp']//1000 < ts_end]
			rh = np.array([get_role_vector(m) for m in games_in_month])
			rfv = np.sum(rh, axis=0)
			#print(rfv)
			#unfocus = 10 - int(np.max(rfv) / np.sum(rfv) * 10)
			mostrole = np.argmax(rfv)
			monthly_mosts[mostrole] += 1
		monthly_mosts_entropy = entropy(monthly_mosts)

		#end

		#performance metric
		initial_score = calculate_score(first_month_data)
		final_score = calculate_score(last_month_data)
		total_score_change = final_score - initial_score
		monthly_score_change = total_score_change/delta_month;
		gamewise_score_change = total_score_change/num_games;
		
		#choice metric
		role_history = np.array([get_role_vector(m) for m in games_within_period])
		role_frequency_vector = np.sum(role_history, axis=0)
		#role_proportion_vector = role_frequency_vector/num_games
		#secondary_role_proportion = 1 - (max(role_frequency_vector) / num_games)

		role_entropy = entropy(role_frequency_vector);
		#print(role_frequency_vector, role_entropy)
		features = [monthly_score_change, gamewise_score_change, initial_score, monthly_games, role_entropy, monthly_mosts_entropy]
		X.append(features)
	X = np.array(X)

	df = pd.DataFrame(X, columns=['D_skill/month', 'D_skill/game', 'init_skill', '#games/month', 'roles_entropy', 'monthly_mosts_entropy'])
	print (df)
	df.plot(kind="scatter", x='D_skill/month', y='roles_entropy', alpha=0.1)
	plt.show()
	df.plot(kind="scatter", x='D_skill/game', y='roles_entropy', alpha=0.1)
	plt.show()
	df.plot(kind="scatter", x='init_skill', y='roles_entropy', alpha=0.1)
	plt.show()
	df.plot(kind="scatter", x='D_skill/month', y='monthly_mosts_entropy', alpha=0.1)
	plt.show()
	df.plot(kind="scatter", x='D_skill/game', y='monthly_mosts_entropy', alpha=0.1)
	plt.show()
	df.plot(kind="scatter", x='init_skill', y='monthly_mosts_entropy', alpha=0.1)
	plt.show()
	return df


def propensity_score_matching():
	whole_data = userdata_opgg()
	X = []
	for u in whole_data:
		opgg_record = u['recent']
		matches = u['matchlist']['matches']

		# entropy of monthly main role
		monthly_mosts = [0,0,0,0,0] #aggregate of monthly mosts 
		for i in range(len(opgg_record)-1):
			ts_start = month_to_timestamp(opgg_record[i]['month'])
			ts_end = month_to_timestamp(opgg_record[i+1]['month'])
			monthly_games = [m for m in matches if ts_start < m['timestamp']//1000 and m['timestamp']//1000 < ts_end]
			rh = np.array([get_role_vector(m) for m in monthly_games])
			rfv = np.sum(rh, axis=0)
			#print(rfv)
			#unfocus = 10 - int(np.max(rfv) / np.sum(rfv) * 10)
			mostrole = np.argmax(rfv)
			monthly_mosts[mostrole] += 1
		monthly_mosts_entropy = entropy(monthly_mosts)
		if monthly_mosts_entropy >= 1:
			return
		features = []




def feature_analysis(dataframe):
	print(dataframe)
	#clf = LassoCV
	#sfm = SelectFromModel(clf, threshold=0.25)
	#sfm.fit(X, y)

def switchers():
	tierwise_data = get_tierwise_userdata()
	for i in len(tierwise_data):
		tierwise_data[i]


def EI_analysis():
	#doing with progress-diversity
	history_with_opgg = userdata_opgg()
	dataframe = []
	for h in history_with_opgg:
		opgg_record = h['recent']
		matches = h['matchlist']['matches']

		for i in range(len(opgg_record)-1):
			r_start = opgg_record[i]
			r_end = opgg_record[i+1]
			score_start = calculate_score(r_start)
			score_end = calculate_score(r_end)
			delta_score = score_end - score_start
			
			if r_start['month'] == '2017.01.01':
				continue
			month_ts_start = month_to_timestamp(r_start['month'])
			month_ts_end = month_to_timestamp(r_end['month'])
			matches_in_month = [m for m in matches if month_ts_start < m['timestamp']//1000 and m['timestamp']//1000 < month_ts_end]
			num_games = len(matches_in_month)
			if num_games == 0:
				continue

			monthly_role_history = np.array([get_role_vector(m) for m in matches_in_month])
			role_frequency_vector = np.sum(monthly_role_history, axis=0)
			role_proportion_vector = role_frequency_vector/num_games

			#TODO:maybe exclude one of the above 2
			monthly_entropy = entropy(np.sum(monthly_role_history, axis=0))


			monthly_features = [delta_score, score_start, num_games, monthly_entropy]
			monthly_features = np.append(monthly_features, role_frequency_vector)
			#monthly_features = np.append(monthly_features, role_proportion_vector)
		
			dataframe.append(monthly_features)
			role_sequence.append(monthly_role_history)

		return


def has_won(region, aid, gameId):
	connection = sqlite3.connect(region + '.db')
	cur = connection.cursor()
	cur.execute("SELECT match FROM matches where gameId=?",(gameId,))
	result = cur.fetchone()
	if result == None:
		return 0.5
	match_dto = json.loads(result[0])
	pid = 0
	for p in match_dto['participantIdentities']:
		if aid == p['player']['currentAccountId']:
			pid = p['participantId']
	if pid == 0:
		print("victory error")
	win = 0.5
	for p in match_dto['participants']:
		if p["participantId"] == pid:
			if p['stats']['win'] == True:
				win = 1
			else:
				win = 0
	return win

def monthly_regression():
	history_with_opgg = userdata_opgg()
	dataframe = []
	role_sequence = []
	for h in history_with_opgg:
		opgg_record = h['recent']
		matches = h['matchlist']['matches']

		months_sequence = []
		for i in range(len(opgg_record)-1):
			r_start = opgg_record[i]
			r_end = opgg_record[i+1]
			score_start = calculate_score(r_start)
			score_end = calculate_score(r_end)
			delta_score = score_end - score_start
			
			if r_start['month'] in ['2017.01.01', '2017.02.01','2017.03.01']:
				continue
			month_ts_start = month_to_timestamp(r_start['month'])
			month_ts_end = month_to_timestamp(r_end['month'])
			matches_in_month = [m for m in matches if month_ts_start < m['timestamp']//1000 and m['timestamp']//1000 < month_ts_end]
			num_games = len(matches_in_month)
			if num_games == 0:
				continue

			monthly_role_history = np.array([get_role_vector(m) for m in matches_in_month])
			role_frequency_vector = np.sum(monthly_role_history, axis=0)
			#role_proportion_vector = role_frequency_vector/num_games

			main_role_proportion = 1 - max(role_frequency_vector) / num_games

			#TODO:maybe exclude one of the above 2
			monthly_entropy = entropy(role_frequency_vector)


			monthly_features = [delta_score, score_start, num_games, entropy]
			#monthly_features = np.append(monthly_features, role_frequency_vector)
			#monthly_features = np.append(monthly_features, role_proportion_vector)
			
			months_sequence.append(monthly_features)
			#dataframe.append(monthly_features)
			#role_sequence.append(monthly_role_history)
		for i in range(len(months_sequence)-2):
			#print(months_sequence)
			past_features = np.append(months_sequence[i+2], months_sequence[i+1])
			past_features = np.append(past_features, months_sequence[i])
			#print(np.shape(past_features))
			dataframe.append(past_features)
	#include past 3 months


	with open('dataframe.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(dataframe)
	dataframe = np.array(dataframe)
	role_sequence = dataframe[:,-1]

	num_data = len(dataframe)
	num_train = int(num_data*(0.8))
	num_test = num_data - num_train
	print("total, tain, test: ", num_data, num_train, num_test)

	X_train = dataframe[:num_train, :]
	X_test = dataframe[num_train:, :]
	Seq_train = role_sequence[:num_train]
	Seq_test = role_sequence[num_train:]

	print("shapes", np.shape(X_train), np.shape(X_test), len(Seq_train), len(Seq_test))


	#rnn_regression(X_train, X_test, Seq_train, Seq_test)
	for h in [2, 5, 10, 20, 50, 100, 200, 300]:
		simple_regression(X_train, X_test, h)

def plot_EI():
	history_with_opgg = userdata_opgg()
	dataframe = []
	role_sequence = []
	for h in history_with_opgg:
		opgg_record = h['recent']
		matches = h['matchlist']['matches']

		months_sequence = []
		for i in range(len(opgg_record)-1):
			r_start = opgg_record[i]
			r_end = opgg_record[i+1]
			score_start = calculate_score(r_start)
			score_end = calculate_score(r_end)
			delta_score = score_end - score_start
			
			if r_start['month'] in ['2017.01.01', '2017.02.01','2017.03.01']:
				continue
			month_ts_start = month_to_timestamp(r_start['month'])
			month_ts_end = month_to_timestamp(r_end['month'])
			matches_in_month = [m for m in matches if month_ts_start < m['timestamp']//1000 and m['timestamp']//1000 < month_ts_end]
			num_games = len(matches_in_month)
			if num_games == 0:
				continue

			monthly_role_history = np.array([get_role_vector(m) for m in matches_in_month])
			role_frequency_vector = np.sum(monthly_role_history, axis=0)

			main_role_proportion = 1 - max(role_frequency_vector) / num_games
			#TODO:maybe exclude one of the above 2
			monthly_entropy = entropy(role_frequency_vector)

			monthly_features = [delta_score, score_start, num_games, main_role_proportion]
			
			months_sequence.append(monthly_features)
			dataframe.append(monthly_features)
		if len(months_sequence) == 0:
			continue
		months_sequence = np.array(months_sequence)
		#plt.figure(1, figsize=(10,10))
		#plt.plot(months_sequence[:,1], months_sequence[:,3])
		#plt.plot([months_sequence[0,1]], [months_sequence[0,3]], 'ro')
		#axes = plt.gca()
		#axes.set_xlim([0, 3000])
		#axes.set_ylim([0,1])
		#plt.show()
	dataframe = np.array(dataframe)
	plt.figure(1, figsize=(10,10))
	plt.plot(dataframe[:,0], dataframe[:,3], 'r.', alpha=0.1)
	plt.show()
	plt.figure(1, figsize=(10,10))
	plt.plot(dataframe[:,0], dataframe[:,2], 'r.', alpha=0.1)
	plt.show()
	plt.figure(1, figsize=(10,10))
	plt.plot(dataframe[:,0], dataframe[:,1], 'r.', alpha=0.1)
	plt.show()


def rnn_regression(X_train, X_test, Seq_train, Seq_test):

	seq_length = 7
	data_dim = 5
	hidden_dim = 10
	output_dim = 1
	learning_rate = 0.01
	iterations = 500

	# Open, High, Low, Volume, Close
	xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
	xy = xy[::-1]  # reverse order (chronically ordered)
	xy = MinMaxScaler(xy)
	x = xy
	y = xy[:, [-1]]  # Close as label

	# build a dataset
	dataX = []
	dataY = []
	for i in range(0, len(y) - seq_length):
	    _x = x[i:i + seq_length]
	    _y = y[i + seq_length]  # Next close price
	    print(_x, "->", _y)
	    dataX.append(_x)
	    dataY.append(_y)

	# train/test split
	train_size = int(len(dataY) * 0.7)
	test_size = len(dataY) - train_size
	trainX, testX = np.array(dataX[0:train_size]), np.array(
	    dataX[train_size:len(dataX)])
	trainY, testY = np.array(dataY[0:train_size]), np.array(
	    dataY[train_size:len(dataY)])

	# input place holders
	X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
	Y = tf.placeholder(tf.float32, [None, 1])

	# build a LSTM network
	cell = tf.contrib.rnn.BasicLSTMCell(
	    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
	outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
	Y_pred = tf.contrib.layers.fully_connected(
	    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

def simple_regression(X_train, X_test, hidden):
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	regressor = MLPRegressor(hidden_layer_sizes = (hidden,), alpha = 0.1, max_iter = 10000)
	#regressor = linear_model.LinearRegression()
	regressor.fit(X_train[:,1:], X_train[:,0])
	coeffs = regressor.score(X_test[:,1:], X_test[:,0])
	print (hidden, coeffs)
	

def test_regressor():
	X_train = np.array([[1,2],[2,3],[3,4],[4,5],[5,6], [13,14],[16,17]])
	X_test = np.array([[9,10], [11,12]])

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	print(X_train)
	print(X_test)
	#regressor = linear_model.LinearRegression()
	regressor = MLPRegressor(hidden_layer_sizes = 2, solver='lbfgs', max_iter=100000)
	regressor.fit(X_train[:,1:], X_train[:,0])
	preds = regressor.predict(X_test[:,1:])
	coeffs = regressor.score(X_test[:,1:], X_test[:,0])
	print(preds, coeffs)


def month_to_timestamp(monthtext):
	#monthtext: "1111.11.11"
	t = time.mktime(datetime.strptime(monthtext, "%Y.%m.%d").timetuple())
	return t

def calculate_score(monthdict):
	#monthdict: {'month', 'tier', 'rank', 'point'}
	tier = monthdict['tier']
	rank = monthdict['rank']
	point = monthdict['point']
	score = point + (rank-1)*100 + (tier-1)*500
	if tier == 6:
		score -= 400
	if tier == 7:
		score -= 800
	return score


def model():
	history_with_opgg = userdata_opgg();

	X = []
	lengths = []
	for u in history_with_opgg:
		opgg_record = u['recent']
		matches = u['matchlist']['matches']

		#len = number of months
		monthly_data = []

		for i in range(len(opgg_record)-1):
			if opgg_record[i]['month'] >= '2018.01.30':
				continue
			#starting the month with this score
			start_score = calculate_score(opgg_record[i])
			end_score = calculate_score(opgg_record[i+1])

			ts_start = month_to_timestamp(opgg_record[i]['month'])
			ts_end = month_to_timestamp(opgg_record[i+1]['month'])
			games_in_month = [m for m in matches if ts_start < m['timestamp']//1000 and m['timestamp']//1000 < ts_end]
			if len(games_in_month) == 0:
				rfv = np.array([0,0,0,0,0])
			else:
				rh = np.array([get_role_vector(m) for m in games_in_month])
				rfv = np.sum(rh, axis=0)
			rfv = np.sum(rfv)
			#print("gim, rfv", len(games_in_month), np.shape(rfv))

			#a vector of size 5 + 1
			month_data = np.append([end_score, start_score], rfv)
			#print("month_data", np.shape(month_data))
			monthly_data.append(month_data)

		seq_length = len(monthly_data)
		while len(monthly_data) < 9:
			monthly_data.append(np.zeros(np.shape(monthly_data)[1]))
		#print("monthly_data", np.shape(monthly_data))
		X.append(monthly_data)
		lengths.append(seq_length)

	X = np.array(X) #shape [users,9,7]
	print(X)
	#print(np.shape(X))
	#lengths = np.array(lengths)

	input_dim = np.shape(X)[2] - 1
	output_dim = 1
	hidden_size = 10
	max_seq_len = 9
	iterations = 10000
	learning_rate = 0.1

	train, test, train_len, test_len = train_test_split(X, lengths, test_size=0.2, random_state=0)
	print("train, test shape:", np.shape(train), np.shape(test))
	trainX = train[:,:,1:]
	trainY = train[:,:,0]

	testX = test[:,:,1:]
	testY = test[:,:,0]
	print("testX, testY")
	print(testX[:,:,0], testY)

	BASELINE = False
	if BASELINE:
		predicted = testX[:,:,0]
		true_val = testY
		print(predicted)
		print(true_val)
		rmse = np.sqrt(np.mean(np.square(true_val - predicted)))
		print (rmse)
		return
	SVM = False
	if SVM:
		from sklearn.svm import SVR
		n_samples, n_features = np.shape(testX)[0], np.shape(testX)[1]
		y = testY
		X = 

	#dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
	#dataset = dataset.batch(batch_size)
	#iterator = dataset.make_one_shot_iterator()
	#next_element = iterator.get_next()

	input_data = tf.placeholder(tf.float32, [None, max_seq_len, input_dim])
	result_data = tf.placeholder(tf.float32, [None, max_seq_len])
	batch_size = tf.placeholder(dtype=tf.int32, shape=[])
	seq_len = tf.placeholder(dtype=tf.int32, shape=(None))

	cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
	initial_state = cell.zero_state(batch_size, tf.float32)
	#inputs: [batch_size, max_time, ...]
	#'state' is a tensor of shape [batch_size, cell_state_size]
	#outputs = [batch_size, max_time, cell.output_size]
	outputs, state = tf.nn.dynamic_rnn(cell, input_data,
										sequence_length=seq_len,
										initial_state=initial_state, dtype=tf.float32)

	X_for_fc = tf.reshape(outputs, [-1, hidden_size])
	
	Y_pred = tf.contrib.layers.fully_connected(X_for_fc, output_dim, activation_fn=None)
	# cost/loss
	Y_pred_ = tf.reshape(Y_pred, [batch_size, seq_length])
	loss = tf.reduce_sum(tf.square(Y_pred_ - result_data))  # sum of the squares
	# optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	targets = tf.placeholder(tf.float32, [None, seq_length])
	predictions = tf.placeholder(tf.float32, [None, seq_length])
	rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))



	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		# Training step
		for i in range(iterations):
		    
		    _, step_loss = sess.run([train, loss], feed_dict={
		                            input_data: trainX, result_data: trainY, 
		                            batch_size: len(trainX), seq_len: train_len})
		    if i%100==0:
			    print("[step: {}] loss: {}".format(i, step_loss))

		# Test step
		test_predict = sess.run(Y_pred_, feed_dict={input_data: testX, batch_size: len(testX),
													seq_len: test_len})
		rmse_val = sess.run(rmse, feed_dict={
		                targets: testY, predictions: test_predict, 
		                batch_size: len(testX), seq_len: test_len})
		print("RMSE: {}".format(rmse_val))

		# Plot predictions
		for i in range(len(testY)):

			print(testX[i,:,:])
			plt.plot(testY[i,:])
			plt.plot(test_predict[i,:])
			plt.plot(testX[i,:,1])
			plt.xlabel("Months")
			plt.ylabel("Rank")
			plt.show()


def x():
	fetch_all_user_history(seasons=[9, 10, 11])

def match_stats():
	from collections import defaultdict
	histories = fetch_all_user_history(seasons=[1,2,3,4,5,6,7,8,9,10,11], queues=range(500))
	for u in histories:
		dq = defaultdict(int)
		ds = defaultdict(int)
		for m in u['matchlist']['matches']:
			dq[m['queue']] += 1
			ds[m['season']] += 1
		print (dq)
		print (ds)



if __name__ == "__main__":
    # execute only if run as a script
    print("1: monthly regression \n 2: plot EI \n 3: lasso")
    x = input()
    if x == "1":
    	monthly_regression()
    elif x == "2":
    	plot_EI()
    elif x == '3':
    	lasso_analysis()
    elif x == '4':
    	model()
    elif x == '5':
    	match_stats()
    #test_regressor()
    #plot_EI()
