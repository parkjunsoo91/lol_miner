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



#timestamp vs pickrole
def pick_visualize():
	cluster_map = {}
	with open('clusters.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			cluster_map[int(row[0])] = int(row[2])

	histories = fetch_all_user_history()
	for row in histories:
		matches = row['matchlist']['matches']
		timestamp_sequence = [match_reference_dto['timestamp'] for match_reference_dto in matches]
		role_sequence = [cluster_map[match_reference_dto['champion']] for match_reference_dto in matches]
		#iv = np.array(iv_)[np.newaxis].T
		#dv = np.array(dv_)[np.newaxis].T
		#regr = linear_model.LinearRegression()
		#regr.fit(iv, dv)
		plt.title(row['tier'])
		plt.plot(timestamp_sequence, role_sequence, 'ro')
		#plt.plot(iv, regr.predict(iv), color=color_dict[tier], linewidth=2)
		plt.show()


def main():
	print("1 - visualize pick sequence")
	print("2 - ")
	print("3 - ")
	print("4 - ")
	print("5 - ")
	print("6 - ")
	print("9 - exit")
	num = input("enter command: ")
	if num == '1':
		pick_visualize()
	elif num == '2':
		pass
	elif num == '3':
		pass
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