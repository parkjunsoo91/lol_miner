import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math
import random

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

#timestamp vs pickrole
def pick_visualize():
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()
	for row in histories:
		matches = row['matchlist']['matches']
		timestamp_sequence = [match_reference_dto['timestamp']/1000 for match_reference_dto in matches]
		role_sequence = [cluster_map[match_reference_dto['champion']] for match_reference_dto in matches]
		plt.title(row['tier'])
		plt.plot(timestamp_sequence, role_sequence, 'r.')
		plt.show()

def entropy_overtime():
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

def entropy_overgame():
	PLOT_RANGE = 200
	mode = input("color: 1 by cluster, 2 by lane")
	recent = int(input("winrate recency: "))


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
					win = min(1, 0.5 + 0.05*recent)
				else:
					win = max(0, 0.5 - 0.05*recent)
				kills = match_reference_dto['kills']
				deaths = match_reference_dto['deaths']
				assists = match_reference_dto['assists']
				kda = math.log2(max((kills + assists) / max(deaths, 1), 0.1))
			else:
				print("no winloss info")
				win = 0.5
				kda = 0
			wins.append(win)
			winrate = sum(wins) / len(wins)
			winrate20 = 0
			if len(wins) > recent:
				winrate20 = sum(wins[-recent:]) / recent
			kdas.append(kda)
			matchlist_data.append([index, champion_id, 
				cluster, entropy(cluster_histogram), 
				lane, entropy([e[1] for e in lane_histogram.items()]),
				winrate20, kda])

			index += 1

		entropy_sequence = [m[5] for m in matchlist_data]
		segments = change_point_analysis(entropy_sequence)

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
				averaged_kda_sequence = [np.mean(kda_sequence[max(i-recent,0):i]) for i in range(len(kda_sequence))]
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

		segment_sequence = [segment[0] for segment in segments if segment[0] in range(PLOT_RANGE)]
		bar_height = [entropy_sequence[i] for i in segment_sequence]
		plt.plot(segment_sequence, bar_height, 'mo')

		plt.show()

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



def entropy_tier():
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()

	for row in histories:
		matches = row['matchlist']['matches']
		matchlist_data = []
		cluster_histogram = [0] * len(cluster_labels)
		lane_histogram = {'TOP':0,'MID':0,'BOTTOM':0,'JUNGLE':0,}
		index = 0
		wins = []
		for match_reference_dto in reversed(matches):
			champion_id = match_reference_dto['champion']
			cluster = cluster_map[champion_id]
			cluster_histogram[cluster] += 1
			lane = match_reference_dto['lane']
			lane_histogram[lane] += 1
			#win = is_winner(match_reference_dto['gameId'], row['aid'])

def consecutive_victory_plot():
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

def draw_regression(iv_, dv_, color = 'b-'):
	iv = np.array(iv_)[np.newaxis].T
	dv = np.array(dv_)[np.newaxis].T
	regr = linear_model.LinearRegression()
	regr.fit(iv, dv)
	plt.plot(iv, regr.predict(iv), color)


def after_win_analysis(has_lost):
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

def mean_consecutive_picks():
	histories = fetch_all_user_history()
	cluster_map, cluster_labels, champion_map = load_cluster_map()	
	data = []
	for row in histories:
		tier = row['tier']
		matches = row['matchlist']['matches']
		pick_sequence = [match_reference_dto['champion'] for match_reference_dto in matches]



def main():
	print("1 - visualize pick sequence")
	print("2 - entropy over time")
	print("3 - entropy over games")
	print("4 - entropy per tier")
	print("5 - consecutive_victory_plot")
	print("6 - after win analysis")
	print("7 - after loss analysis")
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
	return

main()

# x: game, y: entropy, color: lane
# x: game, y: preformance, color: 