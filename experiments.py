import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy


#tier

'''
statistics: # of games
'''
def games_played():

	regionnames = ['K', 'N', 'E']
	tiernames = ['B', 'S', 'G', 'P', 'D']
	ranknames = [1,2,3,4,5]
	names = []
	scores = []
	for reg in regionnames:
		for t in tiernames:
			for r in ranknames:
				names.append(reg + t + str(r))

	regions = ['KR', 'NA1', 'EUW']
	tiers = [1,2,3,4,5]
	ranks = [1,2,3,4,5]

	values = []
	for region in regions:
		histories = fetch_all_user_history(region=[region])

		for tier in tiers:
			for rank in ranks:
				g = [num_games_ranked_season7(e['matchlist']) for e in histories if e['tier']==tier and e['rank']==rank]
				values.append(np.mean(g))

	print(len(names), names)
	print(len(values), values)

	#plt.figure(1, figsize=(10, 5))

	plt.bar(names, values)
	plt.title('Categorical Plotting')
	plt.show()

def get_role_vector(match_ref_dto):
	lane = match_ref_dto['lane']
	role = match_ref_dto['role']
	if role == "DUO_SUPPORT":
		return [0,0,0,0,1]
	if role == "DUO_CARRY":
		return [0,0,0,1,0]
	if role == "DUO":
		return [0,0,0,0.5,0.5]
	if role == "SOLO":
		if lane == "TOP" or lane == "BOTTOM":
			return [1,0,0,0,0]
		if lane == "MID":
			return [0,0,1,0,0]
	if role == "NONE":
		if lane == "TOP":
			return [1,0,0,0,0]
		if lane == "JUNGLE":
			return [0,1,0,0,0]
		if lane == "MID":
			return [0,0,1,0,0]
		if lane == "BOTTOM":
			return [0,0,0,0.5,0.5]
	print("role vector exception:" + lane + role)
	return [0.2,0.2,0.2,0.2,0.2]

def is_ranked(match_ref_dto):
	if match_ref_dto['season'] == 9 and match_ref_dto['queue'] == 420:
		return True
	return False




def entropy_rnn():

	#parameters
	WINDOW_SIZE = 10
	MAX_ENTROPY_SEQ = 500

	x_data = []
	x_seq_length = []

	x_tiers=[]
	x_mostroles=[]
	x_entropies=[]
	x_deltas=[]


	regions = ['KR', 'NA1', 'EUW']
	print("starting history fetch")
	histories = fetch_all_user_history()
	print("history fetch end")
	seq_lengths = [u['matchlist']['totalGames'] for u in histories]
	#print("seq_lengths", np.mean(seq_lengths), np.std(seq_lengths))
	print("entropy history preprocessing...")
	print(len(histories))
	for u in histories:
		#make sure this has only seasonid 9 queue 420
		role_history = np.array([get_role_vector(m) for m in u['matchlist']['matches'] if is_ranked(m)])
		if len(role_history) < WINDOW_SIZE*5:
			continue #discard
		entropy_history = [entropy(np.sum(role_history[i:i+WINDOW_SIZE,:], axis=0)) for i in range(min(len(role_history)-WINDOW_SIZE, MAX_ENTROPY_SEQ))]
		#TODO: should i add padding?
		padded_history = np.pad(entropy_history, (0, MAX_ENTROPY_SEQ-len(entropy_history)), 'constant', constant_values=0)
		x_data.append(np.reshape(padded_history,(-1,1)))
		x_seq_length.append(len(entropy_history))
		#print(len(entropy_history), len(padded_history), type(entropy_history), type(padded_history))

		role_sum_vector = np.sum(role_history, axis=0)
		most_role = np.argmax(role_sum_vector)
		entrop = entropy(role_sum_vector)
		recent_history = get_recent_history(u['aid'], region=u['region'])
		rank_delta = get_rank_delta(recent_history)

		x_tiers.append(u['tier'])
		x_mostroles.append(most_role)
		x_entropies.append(entrop)
		x_deltas.append(rank_delta)

	x_data = np.array(x_data)
	print(np.shape(x_data))
	#learning_rate = 0.01
	training_epoch = 10
	#batch_size = 50
	# 신경망 레이어 구성 옵션
	#n_hidden = 64  # 히든 레이어의 뉴런 갯수
	#n_input = 5*64

	tf.set_random_seed(0)
	input_dim = 1 #entropy
	hidden_size = 20
	num_classes = 1
	batch_size = 50
	sequence_length = MAX_ENTROPY_SEQ
	learning_rate = 0.1
	
	with tf.variable_scope('encoder'):
		batch_sequence_length = tf.placeholder(tf.float32, [None])
		encoder_inputs = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
		encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
		#initial_state = encoder_cell.zero_state(batch_size, tf.float32)
		#zero_state: [batch_size, state_size] which is (c:[batch, hidden], h:[batch, hidden])
		encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, 
														   encoder_inputs, 
														   #initial_state=initial_state,
														   sequence_length=batch_sequence_length,
														   dtype=tf.float32)
	#inputs: [batch_size, max_time, input_dim]
	#outputs: [batch_size, max_time, cell.output_size]
	#state: [batch_size, cell.state_size] which is (c:[batch_size, hidden_size], h:[batch_size, hidden_size])
	#cell.state_size = LSTMStateTuple(c=hidden_size, h=hidden_size)
	#cell.output_size = hidden_size
	with tf.variable_scope('decoder'):
		decoder_inputs = tf.placeholder(tf.float32, [None, sequence_length, input_dim])

		decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
		decoder_outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell, 
														   decoder_inputs,
														   initial_state=encoder_state, 
														   sequence_length=batch_sequence_length,
														   dtype=tf.float32)
	#outputs: [batch_size, max_time, cell.output_size]
	#state: [batch_size, cell.state_size]

	reshaped_outputs = tf.reshape(decoder_outputs, [-1, hidden_size])

	fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
	fc_b = tf.get_variable("fc_b", [num_classes])
	predicted = tf.matmul(reshaped_outputs, fc_w) + fc_b
	#predicted = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

	reshaped_predicted = tf.reshape(predicted, [-1, sequence_length, num_classes])



	loss = tf.reduce_sum(tf.square(reshaped_predicted - encoder_inputs))
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	data_count = len(x_data)
	print(data_count)
	total_batch = int(data_count/batch_size)
	print(total_batch)

	for epoch in range(training_epoch):
	    total_cost = 0

	    for i in range(total_batch):
	        encoder_batch = x_data[(i)*batch_size:(i+1)*batch_size,:,:]
	        decoder_batch = x_data[(i)*batch_size:(i+1)*batch_size,:,:]
	        decoder_batch = np.insert(decoder_batch, 0, np.zeros((1,batch_size,input_dim)), axis=1)[:,:-1,:]
	        seq_len_batch = x_seq_length[(i)*batch_size:(i+1)*batch_size]
	        '''
	        batch_xs2 = x_data[(i)*batch_size:(i+1)*batch_size,:,:]
	        for j in range(len(batch_xs2)):
	        	batch_xs2[j] = np.pad(batch_xs2[j], (1,0), 'constant', constant_values=[0])[:-1]
	        '''

	        #print(np.shape(batch_xs))
	        _, cost_val = sess.run([train_op, loss],
	                               feed_dict={encoder_inputs: encoder_batch, 
	                               				decoder_inputs: decoder_batch,
	                               				batch_sequence_length: seq_len_batch})
	        total_cost += cost_val

	    print('Epoch:', '%04d' % (epoch + 1),
	          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

	print('최적화 완료!')

	#encoded_results, decoded_results = sess.run([encoder_state,predicted],
	#					feed_dict={encoder_inputs: x_data, 
	#								batch_sequence_length:x_seq_length})
	encoded_results = sess.run(encoder_state,
						feed_dict={encoder_inputs: x_data, 
									batch_sequence_length:x_seq_length})
	print(np.shape(encoded_results))

	plot_embedding(encoded_results[0], tiers=x_tiers, lengths=x_seq_length, mostroles=x_mostroles, entropies=x_entropies, deltas=x_deltas)

'''
do autoencoder
'''
def autoencoder():
	
	# a list of personal data
	x_data = []
	x_tiers = []
	x_mostroles = []
	x_entropies = []
	x_deltas = []
	regions = ['KR', 'NA1', 'EUW']
	for region in regions:
		histories = fetch_all_user_history(regions=[region])
		for u in histories:
			role_history = [get_role_vector(m) for m in u['matchlist']['matches'] if is_ranked(m)]
			if len(role_history) < 64:
				continue #discard
			if len(role_history) > 64:
				role_history = role_history[:64]
			role_sum_vector = np.sum(role_history, axis=0)
			most_role = np.argmax(role_sum_vector)
			entrop = entropy(role_sum_vector)
			recent_history = get_recent_history(u['aid'], region=region)
			rank_delta = get_rank_delta(recent_history)

			x_tiers.append(u['tier'])
			x_mostroles.append(most_role)
			x_entropies.append(entrop)
			x_deltas.append(rank_delta)
			x_data.append(np.reshape(role_history, (320,)))
	print(min(x_deltas), max(x_deltas))
	x_data = np.array(x_data)
	print(np.shape(x_data))
	#########
	# 옵션 설정
	######
	learning_rate = 0.01
	training_epoch = 150
	batch_size = 50
	# 신경망 레이어 구성 옵션
	n_hidden = 64  # 히든 레이어의 뉴런 갯수
	n_input = 5*64   # 입력값 크기 - 5 roles for 64 games

	#########
	# 신경망 모델 구성
	######
	# Y 가 없습니다. 입력값을 Y로 사용하기 때문입니다.
	X = tf.placeholder(tf.float32, [None, n_input])

	# 인코더 레이어와 디코더 레이어의 가중치와 편향 변수를 설정합니다.
	# 다음과 같이 이어지는 레이어를 구성하기 위한 값들 입니다.
	# input -> encode -> decode -> output
	W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
	b_encode = tf.Variable(tf.random_normal([n_hidden]))
	# sigmoid 함수를 이용해 신경망 레이어를 구성합니다.
	# sigmoid(X * W + b)
	# 인코더 레이어 구성
	encoder = tf.nn.sigmoid(
	                tf.add(tf.matmul(X, W_encode), b_encode))


	# encode 의 아웃풋 크기를 입력값보다 작은 크기로 만들어 정보를 압축하여 특성을 뽑아내고,
	# decode 의 출력을 입력값과 동일한 크기를 갖도록하여 입력과 똑같은 아웃풋을 만들어 내도록 합니다.
	# 히든 레이어의 구성과 특성치을 뽑아내는 알고리즘을 변경하여 다양한 오토인코더를 만들 수 있습니다.
	W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
	b_decode = tf.Variable(tf.random_normal([n_input]))
	# 디코더 레이어 구성
	# 이 디코더가 최종 모델이 됩니다.
	decoder = tf.nn.sigmoid(
	                tf.add(tf.matmul(encoder, W_decode), b_decode))

	# 디코더는 인풋과 최대한 같은 결과를 내야 하므로, 디코딩한 결과를 평가하기 위해
	# 입력 값인 X 값을 평가를 위한 실측 결과 값으로하여 decoder 와의 차이를 손실값으로 설정합니다.
	cost = tf.reduce_mean(tf.pow(X - decoder, 2))
	#cost = tf.Print(cost, [cost])
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

	#########
	# 신경망 모델 학습
	######
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	data_count = len(x_data)
	print(data_count)
	total_batch = int(data_count/batch_size)
	print(total_batch)

	for epoch in range(training_epoch):
	    total_cost = 0

	    for i in range(total_batch):
	        batch_xs = x_data[(i)*batch_size:(i+1)*batch_size]

	        #print(np.shape(batch_xs))
	        _, cost_val = sess.run([optimizer, cost],
	                               feed_dict={X: batch_xs})
	        total_cost += cost_val

	    print('Epoch:', '%04d' % (epoch + 1),
	          'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

	print('최적화 완료!')


	encoded_results, decoded_results = sess.run([encoder, decoder],
						feed_dict={X: x_data})

	print(np.shape(encoded_results))
	print(np.shape(decoded_results))

	DECODER_CHECK = False
	if (DECODER_CHECK):
		for i in range(data_count):
			draw_cumulative(x_data[i], decoded_results[i])

	plot_embedding(encoded_results, tiers=x_tiers, mostroles=x_mostroles, entropies=x_entropies, deltas=x_deltas)


def plot_embedding(encoded_results, lengths=None, tiers=None, mostroles=None, entropies=None, deltas=None):
	num_clusters = 5
	labels = doKmeans(encoded_results, num_clusters)
	#print(labels)
	colors = ['r','g','b','c','m','y','k','w']
	markers = ['o','v',"^",'s','P','x']
	#should return n*2 arrays
	pca_res = doPCA(encoded_results)
	#should return 

	for mode in range(6):
		plt.figure(1, figsize=(10,10))
		
		if mode==2 and tiers != None:
			plt.title("Rank")
			for i in range(len(pca_res)):
				plt.plot([pca_res[i][0]], [pca_res[i][1]], colors[tiers[i]]+markers[labels[i]])
		elif mode==0:
			plt.title("K-means result")
			for i in range(len(pca_res)):
				plt.plot([pca_res[i][0]], [pca_res[i][1]], colors[labels[i]]+markers[labels[i]])
		elif mode==1 and mostroles != None:
			plt.title("Most role played")
			for i in range(len(pca_res)):
				plt.plot([pca_res[i][0]], [pca_res[i][1]], colors[mostroles[i]]+markers[labels[i]])
		elif mode==3 and entropies != None:
			plt.title("Entropy")
			for i in range(len(pca_res)):
				plt.plot([pca_res[i][0]], [pca_res[i][1]], colors[int(entropies[i]*3)]+markers[labels[i]])
		elif mode==4 and deltas!= None:
			average = np.mean(deltas)
			plt.title("Delta Rank")
			for i in range(len(pca_res)):
				if deltas[i] < 0:
					red = ("0x%0.2X" % min(255,abs(deltas[i] * 50)))[2:]
					blue = "00"
				else:
					red = "00"
					blue = ("0x%0.2X" % min(255,abs(deltas[i] * 50)))[2:]
				colortext = "#"+red+"00"+blue
				plt.scatter([pca_res[i][0]], [pca_res[i][1]], c=colortext)
		elif mode==5 and lengths != None:
			plt.title("Num games")
			for i in range(len(pca_res)):
				plt.plot([pca_res[i][0]], [pca_res[i][1]], colors[int(lengths[i]/70)]+markers[labels[i]])
		plt.show()


	#########
	# 결과 확인
	# 입력값(위쪽)과 모델이 생성한 값(아래쪽)을 시각적으로 비교해봅니다.
	######
	'''
	sample_size = 10

	samples = sess.run(decoder,
	                   feed_dict={X: mnist.test.images[:sample_size]})

	fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

	for i in range(sample_size):
	    ax[0][i].set_axis_off()
	    ax[1][i].set_axis_off()
	    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

	plt.show()
	'''

def doKmeans(X, n):
	kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
	print("kmeans labels shape: ", np.shape(kmeans.labels_))
	return kmeans.labels_
def doPCA(X, n=2):
	pca = PCA(n_components=n)
	reduced = pca.fit_transform(X)
	print("pca result shape:", np.shape(reduced))
	return reduced


'''
data format: {'totalGames':int, 'matches'[...]}
'''
def num_games_ranked_season7(data):
	count = 0
	for match_ref_dto in data['matches']:
		if match_ref_dto['season'] == 9 and match_ref_dto['queue'] == 420:
			count += 1
	return count


def fetch_all_user_history(regions=['KR','NA1', 'EUW'], seasons=[9], queues=[420]):
	all_region_histories = []
	for region in regions:
		connection = sqlite3.connect(region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT matchlists.aid, summoners.tier, summoners.rank, matchlists.matchlist FROM matchlists INNER JOIN summoners ON matchlists.aid = summoners.aid")
		rows = cur.fetchall()
		histories = [{'aid':row[0], 'tier':row[1], 'rank':row[2], 'matchlist':json.loads(row[3]), 'region':region} for row in rows]
		for history in histories:
			matchlist = history['matchlist']
			matchlist['matches'] = [m for m in matchlist['matches'] if m['season'] in seasons and m['queue'] in queues]
			matchlist['totalGames'] = len(matchlist['matches'])
		all_region_histories += histories
	return all_region_histories
#histories = [{'aid', 'tier', 'rank', 'matchlist':{'matches', totalGames}, 'region'}]


def fetch_all_recent_history(regions=['KR','NA1', 'EUW']):
	all_recent_histories = []
	for region in regions:
		connection = sqlite3.connect(region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT aid,recent FROM tier_history")
		rows = cur.fetchall()
		histories = [{'sid':row[0], 'region': region, 'recent':[row[1]]} for row in rows]
		all_recent_histories += histories
	return all_recent_histories

def get_recent_history(aid, region='KR'):
	connection = sqlite3.connect(region + '.db')
	cur = connection.cursor()
	cur.execute("SELECT sid FROM summoners WHERE aid=?",(aid,))
	sid = cur.fetchone()[0]
	cur.execute("SELECT recent FROM tier_history WHERE aid=?", (sid,))
	result = cur.fetchone()
	if result == None:
		#print("tier history does not exist", sid, region)
		return None
	return json.loads(result[0]) #[{'month', 'point', 'rank, 'tier'}]

def get_match_history(sid, region='KR'):
	print("don't use this function")
	connection = sqlite3.connect(region + '.db')
	cur = connection.cursor()
	cur.execute("SELECT aid FROM summoners WHERE sid=?",(sid,))
	result = cur.fetchone()
	if result == None: 
		print("aid does not exist", sid, region)
		return None
	aid = result[0]
	cur.execute("SELECT matchlist FROM matchlists WHERE aid=?", (aid,))
	result = cur.fetchone()
	if result == None:
		print("matchlist does not exist", sid, region)
		return None
	return {'aid': aid, 'matchlist': json.loads(result[0])}


def get_rank_delta(recent_history):
	if recent_history==None:
		return 0
	year17 = [0] * 12
	for e in recent_history:
		if e['month'] == "2017.04":
			year17[3] = e['tier']*5+e['rank']
		if e['month'] == "2017.05":
			year17[4] = e['tier']*5+e['rank']
		if e['month'] == "2017.06":
			year17[5] = e['tier']*5+e['rank']
		if e['month'] == "2017.07":
			year17[6] = e['tier']*5+e['rank']
		if e['month'] == "2017.08":
			year17[7] = e['tier']*5+e['rank']
		if e['month'] == "2017.09":
			year17[8] = e['tier']*5+e['rank']
		if e['month'] == "2017.10":
			year17[9] = e['tier']*5+e['rank']
		if e['month'] == "2017.11":
			year17[10] = e['tier']*5+e['rank']
		if e['month'] == "2017.12":
			year17[11] = e['tier']*5+e['rank']
	for m in year17:
		if m != 0:
			start = m
			break
	for i in range(11,-1,-1):
		if year17[i] != 0:
			end = year17[i]
			break
	return end - start

'''
entropy vs tier
'''

def draw_cumulative(original, generated):
	input_cum_roles = get_cumulative_rolemajor(original)
	generated_cum_roles = get_cumulative_rolemajor(generated)
	indices = range(np.shape(input_cum_roles)[1])
	plt.subplot(2,1,1)
	for i in range(5):
		plt.plot(indices, input_cum_roles[i])
	plt.ylabel('original')
	plt.subplot(2,1,2)
	for i in range(5):
		plt.plot(indices, generated_cum_roles[i])
	plt.ylabel('reconstructed')
	plt.xlabel('games played')
	plt.suptitle('cumulative role frequency generation')
	plt.show()

#data shape is (320,)
def get_cumulative_rolemajor(data):
	timemajor = np.reshape(data, (-1,5))
	
	seq_len = len(timemajor)
	for i in range(1,seq_len):
		timemajor[i] += timemajor[i-1]
	print(timemajor)
	rolemajor = timemajor.transpose()
	return rolemajor

def say_hi():
	print("hi")


if __name__ == "__main__":
	#autoencoder()
	entropy_rnn()