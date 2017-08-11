import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def pick_history():
	connection = sqlite3.connect('loldata.db')
	cur = connection.cursor()

	cur.execute("SELECT * FROM accounts")
	print(len(cur.fetchall()), "accounts")

	cur.execute("SELECT * FROM accounts")
	accounts = {}
	while True:
		row = cur.fetchone()
		if row == None:
			break
		#add player list
		accounts[row[0]] = {}

	cur.execute("SELECT * FROM matches")
	while True:
		row = cur.fetchone()
		if row == None:
			break
		#add match info into structure
		timestamp = row[6]
		winner = row[7] #100 or 200
		players = [0] * 10
		bans = [0] * 10
		picks = [0] * 10
		for i in range(10):
			players[i] = row[8+i]
			bans[i] = row[18+i]
			picks[i] = row[28+i]
		for i in range(10):
			accountId = players[i]
			if accountId in accounts:
				if (i < 5 and winner == 100) or (i >=5 and winner == 200):
					accounts[accountId][timestamp] = {'pick':picks[i], 'win':1}
				else:
					accounts[accountId][timestamp] = {'pick':picks[i], 'win':0}
	# accounts = {int(id): {int(time): (int(champ), bool)}}
	
	with open('champions.json') as f:
		champ_data = json.load(f)['data']

	#write this to a csv file
	with open('picks.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		for accountId in accounts:
			if len(accounts[accountId]) == 0:
				continue
			orderedTimes = sorted(accounts[accountId]) #sort by timestamp
			#writer.writerow(orderedTimes)
			orderedWins = [accounts[accountId][t]['win'] for t in orderedTimes]
			writer.writerow(orderedWins)
			orderedPicks = [champ_data[str(accounts[accountId][t]['pick'])]['name'] for t in orderedTimes]
			#orderedPicks = [champ_data[str(accounts[accountId][t][0])]['tags'][0] for t in orderedTimes]
			writer.writerow(orderedPicks)

	#draw statistics
	entropy_winrate(accounts, champ_data)
	return

	for k in accounts:
		account = accounts[k]
		champ_frequency = {}
		for t in account:
			match = account[t]
			if match[0] in champ_frequency:
				champ_frequency[match[0]] = champ_frequency[match[0]] + 1
			else:
				champ_frequency[match[0]] = 1
		#print(sorted(champ))
		#print(sorted(champ_frequency))
		ordered_champions = sorted(champ_frequency, key=lambda x: champ_frequency[x], reverse=True)
		#print(k)
		total = 0
		for champ in ordered_champions:
			total += champ_frequency[champ]
		for champ in ordered_champions:
			p = champ_frequency[champ] / total
			#print('\t', champ_data[str(champ)]['name'], '\t', champ_frequency[champ], "%.3f"%p)
			print(champ_frequency[champ],'\t', "%.2f"%p, "%.2f"%entropy([champ_frequency[k] for k in champ_frequency]))
			break
		#print("entropy: ", entropy([champ_frequency[k] for k in champ_frequency]))

def entropy_winrate(accounts, champ_data):
	MIN_SIZE = 20
	NUM_CLUSTERS = 5
	result = []

	clusters = {}
	with open('clusters.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			clusters[int(row[0])] = int(row[2])
	for k in accounts:
		account = accounts[k]
		champ_frequency = {}
		match_count = 0
		wins = 0
		for t in account:
			match = account[t]
			champ_id = match['pick']
			if champ_id in champ_frequency:
				champ_frequency[champ_id]['played'] += 1
			else:
				champ_frequency[champ_id] = {'played':1, 'wins':0, 'cluster':clusters[champ_id]}
			champ_frequency[champ_id]['wins'] += match['win']

			match_count += 1
			wins += match['win']
		if match_count < MIN_SIZE:
			continue

		ordered_frequency = sorted(champ_frequency, key = lambda x: champ_frequency[x]['played'], reverse=True)
		most = ordered_frequency[0]

		most_cluster = clusters[most]

		
		most_role_occurence = sum(1 for champ_id in champ_frequency if clusters[champ_id] == clusters[most])
		most_role_ratio = most_role_occurence/match_count




		entry = {}
		entry['total_played'] = match_count
		entry['total_wins'] = wins
		entry['entropy'] = entropy([champ_frequency[k]['played'] for k in champ_frequency])
		entry['most_pick'] = most
		entry['most_played'] = champ_frequency[most]['played']
		entry['most_wins'] = champ_frequency[most]['wins']
		entry['most_role_ratio'] = most_role_ratio
		entry['most_champ_ratio'] = entry['most_played'] / entry['total_played']
		result.append(entry)

	show_plot("entropy - winrate", [e['entropy'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("entropy - most_winrate", [e['entropy'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("entropy - most_winrate_relative", [e['entropy'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])

	show_plot("most_role_ratio - winrate ", [e['most_role_ratio'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("most_role_ratio - most_winrate", [e['most_role_ratio'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("most_role_ratio - most_winrate_relative", [e['most_role_ratio'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])

	show_plot("most_champ_ratio - winrate", [e['most_champ_ratio'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("most_champ_ratio - most_winrate", [e['most_champ_ratio'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("most_champ_ratio - most_winrate_relative", [e['most_champ_ratio'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])



def show_plot(title, iv_, dv_):
	plt.title(title)
	iv = np.array(iv_)[np.newaxis].T
	dv = np.array(dv_)[np.newaxis].T
	regr = linear_model.LinearRegression()
	regr.fit(iv, dv)
	plt.plot(iv, dv, 'ro')
	plt.plot(iv, regr.predict(iv), color='blue', linewidth=2)
	plt.show()


pick_history()