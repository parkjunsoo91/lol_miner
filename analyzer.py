import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

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


def entropy_overgame():
	cluster_map, cluster_labels, champion_map = load_cluster_map()
	histories = fetch_all_user_history()

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	markers = ['o', '+', '.', ',']

	for row in histories:
		matches = row['matchlist']['matches']
		matchlist_data = []
		histogram = [0] * len(cluster_labels)
		index = 0
		for match_reference_dto in reversed(matches):
			cluster = cluster_map[match_reference_dto['champion']]
			champion_id = match_reference_dto['champion']
			histogram[cluster] += 1
			matchlist_data.append([index, cluster, entropy(histogram), champion_id])
			index += 1
		plt.title(row['tier'])
		for label in cluster_labels:
			index_sequence = [match_data[0] for match_data in matchlist_data if match_data[1] == label and match_data[0] in range(100)]
			entropy_sequence = [match_data[2] for match_data in matchlist_data if match_data[1] == label and match_data[0] in range(100)]
			plt.plot(index_sequence, entropy_sequence, colors[label]+markers[2])
		for match_data in matchlist_data:
			if match_data[0] in range(100):
				plt.text(match_data[0], match_data[2], champion_map[match_data[3]], rotation = 75, fontsize = 8)
		plt.show()



def main():
	print("1 - visualize pick sequence")
	print("2 - entropy over time")
	print("3 - entropy over games")
	print("4 - ")
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
		pass
	elif num == '5':
		pass
	elif num == '6':
		pass
	elif num == '9':
		return
	return

main()