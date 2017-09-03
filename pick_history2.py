import sqlite3
import csv
import json
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


#champ coverage: sum of all
def champ_coverage(pick_history):
	
	return

#returns user dictionary {userid, }
def fetch_all_user_history():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT matchlist.aid, users.tier, matchlist.matchlist FROM matchlist INNER JOIN users ON matchlist.aid = users.aid")
	rows = cur.fetchall()
	histories = [{'aid':row[0], 'tier':row[1], 'matchlist':json.loads(row[2])} for row in rows]
	return histories


#input: {aid, tier, matchlist}
def visualize_history(user_history):
	ordered_matchlist = sorted(user_history['matchlist'])
	ordered_champions = sorted(champ_frequency, key=lambda x: champ_frequency[x], reverse=True)

user_history_statistics()

def pick_history():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM users")
	rows = cur.fetchall()
	if rows == None:
		print("no account found")
		return
	accounts = {}
	for row in rows:
		#add player list
		accounts[row[0]] = {"tier": row[2]}


	cur.execute("SELECT * FROM matches")
	print("decoding matches...")
	count = 0
	while True:
		count += 1
		if count % 10000 == 0:
			print("fetching " + str(count) + " matches")
		#if count == 100000:
		#	break
		row = cur.fetchone()
		if row == None:
			break
		#add match info into structure
		timestamp = row[6]
		winner = row[7] #100 or 200
		players = [0] * 10
		picks = [0] * 10
		bans = [0] * 10
		for i in range(10):
			players[i] = row[8+i]
			picks[i] = row[18+i]
			bans[i] = row[28+i]
		for i in range(10):
			accountId = players[i]
			if accountId in accounts:
				if (i < 5 and winner == 100) or (i >=5 and winner == 200):
					accounts[accountId][timestamp] = {'pick':picks[i], 'win':1}
				else:
					accounts[accountId][timestamp] = {'pick':picks[i], 'win':0}
	# accounts = {int(id): {time: {pick:int, win:int}, time: {pick:int, win:int}, ... , tier: silver}}

	with open('champions.json') as f:
		champ_data = json.load(f)['data']
	'''
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
	'''
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
	tiers = {'BRONZE':[],'SILVER':[],'GOLD':[],'PLATINUM':[],'DIAMOND':[],'MASTER':[],'CHALLENGER':[]}

	clusters = {}
	with open('clusters.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			clusters[int(row[0])] = int(row[2])
	#clusters = {champid: clusterid, ...}
	counter = 0
	for k in accounts:
		counter += 1
		account = accounts[k]
		champ_frequency = {}
		match_count = 0
		wins = 0
		for t in account:
			if t == 'tier':
				continue
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

		cluster_frequency = [0] * NUM_CLUSTERS
		for champ_id in champ_frequency:
			cluster = clusters[champ_id]
			cluster_frequency[cluster] += champ_frequency[champ_id]['played']

		most_role_ratio = max(cluster_frequency) / sum(cluster_frequency)

		if counter == 0:
			print(ordered_frequency)
			print("most champ:", most)
			print("")

		entry = {}
		entry['total_played'] = match_count
		entry['total_wins'] = wins
		entry['entropy'] = entropy([champ_frequency[k]['played'] for k in champ_frequency])
		entry['most_pick'] = most
		entry['most_played'] = champ_frequency[most]['played']
		entry['most_wins'] = champ_frequency[most]['wins']
		entry['most_role_ratio'] = most_role_ratio
		entry['most_champ_ratio'] = entry['most_played'] / entry['total_played']

		tiers[account['tier']].append(entry)

	tiers

	shape_dict = {}
	color_dict = {}
	for tier in tiers:
		shape_dict['BRONZE'] = 'ro'
		shape_dict['SILVER'] = 'go'
		shape_dict['GOLD'] = 'yo'
		shape_dict['PLATINUM'] = 'bo'
		shape_dict['DIAMOND'] = 'co'
		shape_dict['MASTER'] = 'mo'
		shape_dict['CHALLENGER'] = 'ko'
		color_dict['BRONZE'] = 'red'
		color_dict['SILVER'] = 'green'
		color_dict['GOLD'] = 'yellow'
		color_dict['PLATINUM'] = 'blue'
		color_dict['DIAMOND'] = 'cyan'
		color_dict['MASTER'] = 'magenta'
		color_dict['CHALLENGER'] = 'black'


	plt.title("entropy - winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['entropy'] for e in result]
		dv_ = [e['total_wins']/e['total_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()

	plt.title("entropy - most_winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['entropy'] for e in result]
		dv_ = [e['most_wins']/e['most_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("entropy - most_winrate_relative")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['entropy'] for e in result]
		dv_ = [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("most_role_ratio - winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_role_ratio'] for e in result]
		dv_ = [e['total_wins']/e['total_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("most_role_ratio - most_winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_role_ratio'] for e in result]
		dv_ = [e['most_wins']/e['most_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("most_role_ratio - most_winrate_relative")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_role_ratio'] for e in result]
		dv_ = [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("most_champ_ratio - winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_champ_ratio'] for e in result]
		dv_ = [e['total_wins']/e['total_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()
	
	plt.title("most_champ_ratio - most_winrate")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_champ_ratio'] for e in result]
		dv_ = [e['most_wins']/e['most_played'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()

	plt.title("most_champ_ratio - most_winrate_relative")
	for tier in tiers:
		result = tiers[tier]
		iv_ = [e['most_champ_ratio'] for e in result]
		dv_ = [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result]
		draw_plot(iv_, dv_, tier, shape_dict, color_dict)
	plt.show()


	return

	result = tiers["SILVER"]
	print(len(result), "silvers")

	show_plot("entropy - winrate", [e['entropy'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("entropy - most_winrate", [e['entropy'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("entropy - most_winrate_relative", [e['entropy'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])

	show_plot("most_role_ratio - winrate ", [e['most_role_ratio'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("most_role_ratio - most_winrate", [e['most_role_ratio'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("most_role_ratio - most_winrate_relative", [e['most_role_ratio'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])

	show_plot("most_champ_ratio - winrate", [e['most_champ_ratio'] for e in result], [e['total_wins']/e['total_played'] for e in result])
	show_plot("most_champ_ratio - most_winrate", [e['most_champ_ratio'] for e in result], [e['most_wins']/e['most_played'] for e in result])
	show_plot("most_champ_ratio - most_winrate_relative", [e['most_champ_ratio'] for e in result], [e['most_wins']/e['most_played']*e['total_played']/e['total_wins'] for e in result])

def draw_plot(iv_, dv_, tier, shape_dict, color_dict):
	if len(iv_) == 0:
		return
	iv = np.array(iv_)[np.newaxis].T
	dv = np.array(dv_)[np.newaxis].T
	regr = linear_model.LinearRegression()
	regr.fit(iv, dv)
	plt.plot(iv, dv, shape_dict[tier])
	plt.plot(iv, regr.predict(iv), color=color_dict[tier], linewidth=2)

def show_plot(title, iv_, dv_):
	if len(iv_) == 0:
		return
	plt.title(title)
	iv = np.array(iv_)[np.newaxis].T
	dv = np.array(dv_)[np.newaxis].T
	regr = linear_model.LinearRegression()
	regr.fit(iv, dv)
	plt.plot(iv, dv, 'ro')
	plt.plot(iv, regr.predict(iv), color='blue', linewidth=2)
	plt.show()


#pick_history()