import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math

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
	PLOT_RANGE = 1000
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
			if index < PLOT_RANGE:
				print(match_reference_dto)
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
				#plt.plot(index_sequence, entropy_sequence, colors[colorid]+markers[2])
				plt.plot(index_sequence, averaged_kda_sequence, colors[colorid]+markers[2])
				colorid += 1
			#for match_data in matchlist_data:
			#	if match_data[0] in range(PLOT_RANGE):
			#		plt.text(match_data[0], match_data[5], champion_map[match_data[1]], rotation = 75, fontsize = 8)

		index_sequence = [match_data[0] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		winrate_sequence = [match_data[6] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		kda_sequence = [match_data[7] for match_data in matchlist_data if match_data[0] in range(PLOT_RANGE)]
		plt.plot(index_sequence, winrate_sequence, 'k-')
		#plt.plot(index_sequence, kda_sequence)

		plt.show()

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

def main():
	print("1 - visualize pick sequence")
	print("2 - entropy over time")
	print("3 - entropy over games")
	print("4 - entropy per tier")
	print("5 - ")
	print("6 - ")
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
		pass
	elif num == '6':
		pass
	elif num == '9':
		return
	return

main()

# x: game, y: entropy, color: lane
# x: game, y: preformance, color: 