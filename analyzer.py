import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
import math
import random


def main():
	print("1 - visualize pick sequence")
	print("2 - entropy over time")
	print("3 - entropy over games")
	print("4 - entropy per tier")
	print("5 - consecutive_victory_plot")
	print("6 - after win analysis")
	print("7 - after loss analysis")
	print("11 - champion frequency sequence")
	print("12 - pdf of champ pick")
	print("9 - exit")
	num = input("enter command: ")
	if num == '1':
		pick_visualize()
	elif num == '2':
		entropy_overtime()
	elif num == '3':
		entropy_overgame()
	elif num == '4':
		entropy_tier()
	elif num == '5':
		consecutive_victory_plot()
	elif num == '6':
		after_win_analysis(False)
	elif num == '7':
		after_win_analysis(True)
	elif num == '8':
		change_point_tester()
	elif num == '9':
		return
	elif num == '11':
		return
	elif num == '12':
		champ_pickratio_pdf()
	return


def pick_visualize():
	"""Show, for every single player, their roles over all timestamps
	"""
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()
	for row in histories:
		matches = row['matchlist']['matches']
		timestamp_sequence = [match_ref_dto['timestamp']/1000 for match_ref_dto in matches]
		role_sequence = [cluster_map[match_ref_dto['champion']] for match_ref_dto in matches]
		plt.title(row['tier'])
		plt.plot(timestamp_sequence, role_sequence, 'r.')
		plt.show()

def entropy_overtime():
	"""Show, for every single player, their champion entropy over timestamp.
	"""
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()
	for row in histories:
		matches = row['matchlist']['matches']
		timestamp_sequence = [match_reference_dto['timestamp']/1000 for match_reference_dto in reversed(matches)]
		histogram = [0] * len(cluster_labels)
		entropy_sequence = []
		for match_reference_dto in reversed(matches):
			histogram[cluster_map[match_reference_dto['champion']]] += 1
			entropy_sequence.append(entropy(histogram))
		plt.title(row['tier'])
		plt.plot(timestamp_sequence, entropy_sequence, 'r.')
		plt.show()

def entropy_overgame():
	"""Show, for every single player, their entropy(cluster or lane) over games
	with recent winrate.
	"""
	PLOT_RANGE = 5000
	mode = input("color: 1 by cluster, 2 by lane")
	winscale = int(input("wingraph: (1) for 0to1 scale, (2) for absolute scale"))

	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	markers = ['o', '+', '.', ',', '-']
	
	for row in histories:
		matches = row['matchlist']['matches']
		matchlist_data = []
		cluster_histogram = [0] * len(cluster_labels)
		lane_histogram = {'TOP':0,'MID':0,'BOTTOM':0,'JUNGLE':0,}
		index = 0
		wins = []
		kdas = []
		net_winloss_final = 0
		for match_reference_dto in matches:
			queue = match_reference_dto['queue']
			if queue != 4 and queue != 420 :
				continue
			if 'win' in match_reference_dto:
				if match_reference_dto['win'] == True:
					net_winloss_final += 1
				else:
					net_winloss_final -= 1
		print('net winloss ', net_winloss_final)
		if winscale == 1:
			if net_winloss_final != 0:
				winloss_weight = 1.0/abs(net_winloss_final)
			else:
				winloss_weight = 0.02
		else:
			winloss_weight = 0.02
		net_winloss = 0
		if net_winloss_final < 0:
			net_winloss = -1 * net_winloss_final
		for match_reference_dto in reversed(matches):
			queue = match_reference_dto['queue']
			if queue != 4 and queue != 420 :
				continue
			champion_id = match_reference_dto['champion']
			cluster = cluster_map[champion_id]
			cluster_histogram[cluster] += 1
			lane = match_reference_dto['lane']
			lane_histogram[lane] += 1
			#win = is_winner(match_reference_dto['gameId'], row['aid'])
			if 'win' in match_reference_dto:
				if match_reference_dto['win'] == True:
					win = 1
				else:
					win = -1
				kills = match_reference_dto['kills']
				deaths = match_reference_dto['deaths']
				assists = match_reference_dto['assists']
				kda = math.log2(max((kills + assists) / max(deaths, 1), 0.1))
			else:
				print("no winloss info")
				win = 0
				kda = 0
			net_winloss += win
			kdas.append(kda)
			matchlist_data.append([index, champion_id, 
				cluster, entropy(cluster_histogram), 
				lane, entropy([e[1] for e in lane_histogram.items()]),
				net_winloss * winloss_weight, kda])

			index += 1
		print('final net winloss ', net_winloss)
		entropy_sequence = [m[5] for m in matchlist_data]
		entropy_change_sequence = []
		for i in range(len(entropy_sequence) - 1):
			entropy_change_sequence.append(entropy_sequence[i+1] - entropy_sequence[i])
		#segments = change_point_analysis(entropy_change_sequence)

		plt.title(row['tier'])
		if mode == '1':
			for label in cluster_labels:
				index_sequence = [match_data[0] for match_data in matchlist_data if match_data[2] == label and match_data[0] in range(PLOT_RANGE)]
				entropy_sequence = [match_data[3] for match_data in matchlist_data if match_data[2] == label and match_data[0] in range(PLOT_RANGE)]
				kda_sequence = [match_data[7] for match_data in matchlist_data if match_data[4] == lane and match_data[0] in range(PLOT_RANGE)]
				#plt.plot(index_sequence, entropy_sequence, colors[label]+markers[2])
				plt.plot(index_sequence, kda_sequence, colors[label]+markers[2])
			#for match_data in matchlist_data:
			#	if match_data[0] in range(PLOT_RANGE):
			#		plt.text(match_data[0], match_data[3], champion_map[match_data[1]], rotation = 75, fontsize = 8)
		elif mode == '2':
			colorid = 0
			for lane in lane_histogram:
				print(colors[colorid], lane)
				index_sequence = [match_data[0] for match_data in matchlist_data if match_data[4] == lane and match_data[0] in range(PLOT_RANGE)]
				entropy_sequence = [match_data[5] for match_data in matchlist_data if match_data[4] == lane and match_data[0] in range(PLOT_RANGE)]
				kda_sequence = [match_data[7] for match_data in matchlist_data if match_data[4] == lane and match_data[0] in range(PLOT_RANGE)]
				#averaged_kda_sequence = [np.mean(kda_sequence[max(i-recent,0):i]) for i in range(len(kda_sequence))]
				plt.plot(index_sequence, entropy_sequence, colors[colorid]+markers[2])
				#plt.plot(index_sequence, averaged_kda_sequence, colors[colorid]+markers[2])
				colorid += 1
			#for match_data in matchlist_data:
			#	if match_data[0] in range(PLOT_RANGE):
			#		plt.text(match_data[0], match_data[5], champion_map[match_data[1]], rotation = 75, fontsize = 8)

		index_sequence = [match_data[0] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		winrate_sequence = [match_data[6] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		kda_sequence = [match_data[7] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		plt.plot(index_sequence, winrate_sequence, 'k-')
		#plt.plot(index_sequence, kda_sequence)

		#segment_sequence = [segment[0] for segment in segments if segment[0] in range(PLOT_RANGE - 1)]
		#bar_height = [1 for i in segment_sequence]
		#plt.plot(segment_sequence, bar_height, 'mo')

		plt.show()



def champ_pickratio_pdf():
	histories = fetch_all_user_history()
	champ_mean_data = []
	champ_std_data = []
	lane_mean_data = []
	lane_std_data = []
	for row in histories:
		matches = row['matchlist']['matches']
		matches_filtered = [m for m in reversed(matches) if (m['queue']==4 or m['queue']==420)]
		champ_ratio_seq, lane_ratio_seq = pickratio_pdf(matches_filtered)
		champ_mean_data.append(np.mean(champ_ratio_seq))
		champ_std_data.append(np.std(champ_ratio_seq))
		lane_mean_data.append(np.mean(lane_ratio_seq))
		lane_std_data.append(np.std(lane_ratio_seq))	
	show_prob_distribution([champ_mean_data, lane_mean_data])
	X = np.array([champ_mean_data, champ_std_data]).T
	clusters = doKmeans(X, 2)
	show_clusters(X, clusters)
	X = np.array([lane_mean_data, lane_std_data]).T
	clusters = doKmeans(X, 2)
	show_clusters(X, clusters)

def doKmeans(X, n):
	kmeans = KMeans(n_clusters=n, random_state=0)
	kmeans.fit(X)
	distances = kmeans.transform(X)
	clusters = np.argmin(distances, axis=1)
	return clusters

def show_clusters(X, labels):
	colors = ['r', 'g']
	markers = ['o', '+']
	for i in range(np.shape(X)[0]):
		label = labels[i]
		color = colors[label%7]
		marker = markers[label//7]
		style = color + marker
		plt.plot(X[i][0], X[i][1], style)
	plt.xlabel('avg')
	plt.ylabel('std')
	plt.title('clusters')
	plt.show()

def pickratio_pdf(matches):
	champ_freq = {}
	lane_freq = {}
	champ_ratio_sequence = []
	lane_ratio_sequence = []
	for i in range(len(matches)):
		match_ref_dto = matches[i]

		champ = match_ref_dto['champion']
		if champ in champ_freq and i > 0:
			ratio = champ_freq[champ]/i
			champ_freq[champ] += 1
		else:
			ratio = 0
			champ_freq[champ] = 1
		champ_ratio_sequence.append(ratio)

		lane = match_ref_dto['lane']
		if lane in lane_freq and i > 0:
			ratio = lane_freq[lane]/i
			lane_freq[lane] += 1
		else:
			ratio = 0
			lane_freq[lane] = 1
		lane_ratio_sequence.append(ratio)
	show_prob_distribution([champ_ratio_sequence, lane_ratio_sequence])
	return champ_ratio_sequence, lane_ratio_sequence

def show_prob_distribution(data_list):
	color = ['red', 'green']
	for i in range(len(data_list)):
		data = data_list[i]
		mu = np.mean(data)
		sigma = np.std(data)
		num_bins = 50
		n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor=color[i], alpha=0.5)
		y = mlab.normpdf(bins, mu, sigma)
		plt.plot(bins, y, 'r--')
	plt.xlabel('champ pickratio')
	plt.ylabel('probability')
	plt.title(r'$\mu={}$, $\sigma={}$'.format(mu, sigma))
	plt.show()

def entropy_tier():
	"""TODO: not yet implemented...what is this?
	"""
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()
	mmr_list = []
	champ_entropy_list = []
	lane_entropy_list = []
	for row in histories:
		mmr = tier_to_MMR(row['tier'])
		champ_freq = {}
		lane_freq = {}
		matches = row['matchlist']['matches']
		matches_filtered = [m for m in reversed(matches) if (m['queue']==4 or m['queue']==420)]
		for match_ref_dto in matches_filtered:
			champ = match_ref_dto['champion']
			if champ in champ_freq:
				champ_freq[champ] += 1
			else:
				champ_freq[champ] = 0
			lane = match_ref_dto['lane']
			if lane in lane_freq:
				lane_freq[lane] += 1
			else:
				lane_freq[lane] = 0
		mmr_list.append(mmr)
		champ_entropy_list.append(entropy([e[1] for e in champ_freq.items()]))
		lane_entropy_list.append(entropy([e[1] for e in lane_freq.items()]))
	#now visualize the 3d plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(champ_entropy_list, lane_entropy_list, mmr_list, c='r', marker='o')
	ax.set_xlabel('champ entropy')
	ax.set_ylabel('lane entropy')
	ax.set_zlabel('estimated mmr')
	plt.show()




def tier_to_MMR(tier):
	if tier == 'BRONZE':
		mmr = 1
	elif tier == 'SILVER':
		mmr = 2
	elif tier == 'GOLD':
		mmr = 3
	elif tier == 'PLATINUM':
		mmr = 4
	elif tier == 'DIAMOND':
		mmr = 5
	elif tier == 'MASTER':
		mmr = 6
	elif tier == 'CHALLENGER':
		mmr = 7
	else:
		print("error")
	return mmr


def consecutive_victory_plot():
	"""show plot of players, 2 points for each player, each denoting probability
	of picking the same champ, either after a victory or a loss with the same champ,
	two values will add up to the prob. of repicking.
	"""
	histories = fetch_all_user_history()
	data = []
	for row in histories:
		tier = row['tier']
		previous_affinity = 0
		same_picks = 0
		win_picks = 0
		loss_picks = 0 
		matches = row['matchlist']['matches']
		divisor = len(matches) - 1
		for i in range(len(matches)-1):
			result_pick = matches[i]['champion']
			prev_pick = matches[i+1]['champion']
			if not 'win' in matches[i+1]:
				continue
			prev_win = matches[i+1]['win']
			if prev_pick == result_pick:
				same_picks += 1
				if prev_win:
					win_picks += 1
				else:
					loss_picks += 1
		userinfo = {}
		userinfo['same_pick'] = same_picks / divisor
		userinfo['win_pick'] = win_picks / divisor
		userinfo['loss_pick'] = loss_picks / divisor
		userinfo['tier'] = tier
		data.append(userinfo)

	#now draw plot
	plt.title = "same_picks"
	plt.xlabel('probablity of re-picking previous pick')
	plt.ylabel('previous pick won/lost')
	x = [user['same_pick'] for user in data]
	y1 = [user['win_pick'] for user in data]
	y2 = [user['loss_pick'] for user in data]
	plt.plot(x, y1, 'r.')
	plt.plot(x, y2, 'b.')
	draw_regression(x, y1)
	draw_regression(x, y2)

	plt.show()

def after_win_analysis(has_lost):
	"""Some people stick with one champ more than others.
	Generally, everyone is more inclined to pick the same champ if they had
	previously won with it.
	But what makes it different for people? maybe their entropy.

	argument: after win analysis or after loss analysis
	"""
	histories = fetch_all_user_history()
	cluster_map, cluster_labels, champion_map = load_cluster_map()	
	data = []
	for row in histories:
		tier = row['tier']
		matches = row['matchlist']['matches']
		wins = 0
		repicks = 0
		relanes = 0
		pick_histogram = {}
		lane_histogram = {'TOP':0,'MID':0,'BOTTOM':0,'JUNGLE':0,}
		role_histogram = [0] * 5
		for champ_id in cluster_map:
			pick_histogram[champ_id] = 0
		for i in range(len(matches)-1):
			queue = matches[i]['queue']
			if queue != 4 and queue != 420 :
				continue
			pick_histogram[matches[i]['champion']] += 1
			lane_histogram[matches[i]['lane']] += 1
			if not 'win' in matches[i+1]:
				continue
			if matches[i+1]['win'] == has_lost:
				continue
			wins += 1
			prev_pick = matches[i+1]['champion']
			result_pick = matches[i]['champion']
			if prev_pick == result_pick:
				repicks += 1
			prev_lane = matches[i+1]['lane']
			result_lane = matches[i]['lane']
			if prev_lane == result_lane:
				relanes += 1
		if wins == 0:
			continue
		userinfo = {}
		userinfo['win_repick'] = repicks / wins
		userinfo['win_relane'] = relanes / wins
		userinfo['tier'] = tier
		userinfo['champ_entropy'] = entropy([e[1] for e in pick_histogram.items()])
		userinfo['lane_entropy'] = entropy([e[1] for e in lane_histogram.items()])
		userinfo['role_entropy'] = 0
		data.append(userinfo)
	
	tiers = ['BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTER', 'CHALLENGER']
	color = ['r.', 'g.', 'y.', 'b.', 'c.', 'm.', 'k.']
	#now draw plot
	plt.title = "same_picks"
	plt.xlabel('player champion entropy')
	plt.ylabel('probablity of re-picking winning pick')
	if has_lost:
		plt.ylabel('probablity of re-picking losing pick')
	for i in range(len(tiers)):
		x1 = [user['champ_entropy'] for user in data if user['tier'] == tiers[i]]
		y1 = [user['win_repick'] for user in data if user['tier'] == tiers[i]]
		plt.plot(x1, y1, color[i])
		draw_regression(x1, y1, color[i][0]+'-')
	plt.show()


	plt.xlabel('player LANE entropy')
	plt.ylabel('probability of re_picking winning LANE')
	if has_lost:
		plt.ylabel('probablity of re-picking losing LANE')
	for i in range(len(tiers)):
		x2 = [user['lane_entropy'] for user in data if user['tier'] == tiers[i]]
		y2 = [user['win_relane'] for user in data if user['tier'] == tiers[i]]
		plt.plot(x2, y2, color[i])
		draw_regression(x2, y2, color[i][0]+'-')
	plt.show()

def fetch_all_user_history():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT matchlist.aid, users.tier, matchlist.matchlist FROM matchlist INNER JOIN users ON matchlist.aid = users.aid")
	rows = cur.fetchall()
	histories = [{'aid':row[0], 'tier':row[1], 'matchlist':json.loads(row[2])} for row in rows]
	return histories

def load_cluster_map():
	cluster_map = {}
	champion_map = {}
	cluster_labels = {}
	with open('clusters.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			cluster_map[int(row[0])] = int(row[2])
			champion_map[int(row[0])] = row[1]
			if row[1] == "Ashe":
				cluster_labels[int(row[2])] = "Marksman"
			elif row[1] == "Pantheon":
				cluster_labels[int(row[2])] = "Assassin"
			elif row[1] == "Rammus":
				cluster_labels[int(row[2])] = "Tank"
			elif row[1] == "Ryze":
				cluster_labels[int(row[2])] = "Mage"
			elif row[1] == "Soraka":
				cluster_labels[int(row[2])] = "Support"
	return cluster_map, cluster_labels, champion_map

def is_winner(match_id, user_id):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('''SELECT accountId1,accountId2,accountId3,accountId4,accountId5,
	accountId6,accountId7,accountId8,accountId9,accountId10, winner from matches where gameId = ?''', (match_id,))
	row = cur.fetchone()
	if row == None:
		return 0.5
	if row[10] == 100 and user_id in row[:5] or row[10] == 200 and user_id in row[5:]:
		return 1
	else:
		return 0

def get_ranked_sequence(history):
	RANKED_QUEUES = [4,420]
	matches = history['matchlist']['matches'].reverse()
	ranked_sequence = [m for m in matches if m['queue'] in RANKED_QUEUES]
	return ranked_sequence

def generate_entropy_sequence(sequence, categories):
	entropy_seq = []
	histogram = {}
	for category in categories:
		histogram[category] = 0
	for i in range(sequence):
		histogram[sequence[i]] += 1
		list

def user_entropy(history):
	#histories = [{'aid':row[0], 'tier':row[1], 'matchlist':json.loads(row[2])} for row in rows]
	match_seq = get_ranked_sequence(history)
	champ_seq = [match_reference_dto['champion'] for match_reference_dto in match_seq]
	role_seq = None
	champ_entropy_seq = generate_entropy_sequence(champ_seq)
	role_entropy_seq = generate_entropy_sequence(role_seq)

	#TODO: let's work here.

def change_point_tester():
	a = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7]
	print(get_cusum(a))
	segments = change_point_analysis(a)
	print(segments)
	for s in segments:
		for i in range(s[0], s[1]):
			print(a[i], end="")
		print()

def change_point_analysis(data):
	if len(data) < 2:
		return [[0,len(data)]]
	exists_change = bootstrap_analysis(data)
	if exists_change == True:
		index = change_estimator(data)
		subsegments1 = change_point_analysis(data[0:index])
		subsegments2 = change_point_analysis(data[index:len(data)])
		for segment in subsegments2:
			for i in range(2):
				segment[i] += index
		return subsegments1 + subsegments2
	else:
		return [[0,len(data)]]

def bootstrap_analysis(data, num_samples = 100, confidence_required = 0.90):
	cusum_original = get_cusum(data)
	diff_original = max(cusum_original) - min(cusum_original)
	confidence_count = 0
	for i in range(num_samples):
		bootstrap_sample = random.sample(data, len(data))
		cusum_sample = get_cusum(bootstrap_sample)
		diff_sample = max(cusum_sample) - min(cusum_sample)
		if diff_sample < diff_original:
			confidence_count += 1
	confidence_level = confidence_count/num_samples
	print (confidence_level)
	return confidence_level > confidence_required

def get_cusum(data):
	average = np.mean(data)
	cusum = [0]
	for i in range(0, len(data)):
		cusum.append(cusum[-1] + (data[i] - average))
	return cusum

def change_estimator(data):
	cusum = get_cusum(data)
	abs_cusum = [abs(e) for e in cusum]
	return abs_cusum.index(max(abs_cusum)) - 1

def draw_regression(iv_, dv_, color = 'b-'):
	iv = np.array(iv_)[np.newaxis].T
	dv = np.array(dv_)[np.newaxis].T
	regr = linear_model.LinearRegression()
	regr.fit(iv, dv)
	plt.plot(iv, regr.predict(iv), color)

def mean_consecutive_picks():
	histories = fetch_all_user_history()
	cluster_map, cluster_labels, champion_map = load_cluster_map()	
	data = []
	for row in histories:
		tier = row['tier']
		matches = row['matchlist']['matches']
		pick_sequence = [match_reference_dto['champion'] for match_reference_dto in matches]

main()
