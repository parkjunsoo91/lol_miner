import csv
import json
import sqlite3
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

def LoadItemData():
	with open('items.json') as f:
		with open('items.csv', 'w', newline='') as g:
			fullData = json.load(f)
			data = fullData['data']

			writer = csv.writer(g)
			for k, v in sorted(data.items()):
				name = ""
				if 'name' in v:
					name = v['name']
				writer.writerow([k, name])


def LoadChampFeatures():

	with open('champions.json') as f:
		champion_dto_map = json.load(f)['data']
	with open('items.json') as f:
		item_dto_map = json.load(f)['data']

	champ_usage_map = {}
	for champion_id in champion_dto_map:
		#champion id is in string
		champ_usage_map[int(champion_id)] = {'TOP':0, 'MID':0, 'BOTTOM':0, 'JUNGLE':0,
									'SOLO':0, 'DUO':0, 'DUO_CARRY':0, 'DUO_SUPPORT':0, 'NONE':0}

	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT matchlist FROM matchlist")
	rows = cur.fetchall()
	for row in rows:
		matchlist_dto = json.loads(row[0]) 
		for match_reference_dto in matchlist_dto['matches']:
			champion = match_reference_dto['champion']
			lane = match_reference_dto['lane']
			role = match_reference_dto['role']
			if lane in champ_usage_map[champion]:
				champ_usage_map[champion][lane] += 1
			else:
				print("{} ({}), lane label {} doesn't exist".format(
						champion_dto_map[str(champion)]['name'], champion, lane))
			if role in champ_usage_map[champion]:
				champ_usage_map[champion][role] += 1
			else:
				print("{} ({}), role label {} doesn't exist".format(
						champion_dto_map[str(champion)]['name'], champion, role))
	#normalize frequency
	for champion_id in champ_usage_map:
		total = (champ_usage_map[champion_id]['TOP'] + 
				 champ_usage_map[champion_id]['MID'] +
				 champ_usage_map[champion_id]['BOTTOM'] +
				 champ_usage_map[champion_id]['JUNGLE'])
		for key in champ_usage_map[champion_id]:
			champ_usage_map[champion_id][key] /= total

	with open('champions.csv', 'w', newline='') as g:
		fieldnames = []

		fieldnames.append('Id')
		fieldnames.append('Name')

		#6 types
		fieldnames.append('Assassin')
		fieldnames.append('Fighter')
		fieldnames.append('Mage')
		fieldnames.append('Marksman')
		fieldnames.append('Support')
		fieldnames.append('Tank')

		#lane info...
		fieldnames.append('TOP')
		fieldnames.append('MID')
		fieldnames.append('BOTTOM')
		fieldnames.append('JUNGLE')

		#role info...
		fieldnames.append('SOLO')
		fieldnames.append('DUO')
		fieldnames.append('DUO_CARRY')
		fieldnames.append('DUO_SUPPORT')
		fieldnames.append('NONE')

		#4 info, 20 stats
		champ = champion_dto_map['1']
		for k in champ['info']:
			fieldnames.append(k)
		for k in champ['stats']:
			fieldnames.append(k)

		#items
		items = {}
		for key in champion_dto_map:
			champ = champion_dto_map[key]
			for item in getRecommended(champ):
				items[item] = 0
		
		item_count = len(items)
		for k in sorted(items):
			fieldnames.append(k)
		print(len(fieldnames))


		writer = csv.DictWriter(g, fieldnames=fieldnames)
		writer.writeheader()

		for key in sorted(champion_dto_map):
			champion_dto = champion_dto_map[key]
			features = {}

			#initialize fields to default values
			for field in fieldnames:
				features[field] = 0

			features['Id'] = key
			features['Name'] = champion_dto["name"]

			for t in champion_dto['tags']:
				tag_string = t
				if "melee" in tag_string:
					tag_string = "Tank"
				features[tag_string] = 1
			features.update(champ_usage_map[int(key)])
			features.update(champion_dto['info'])
			features.update(champion_dto['stats'])

			items = getRecommended(champion_dto)
			for item in items:
				features[item] = 1
			
			itemStrings = [item_dto_map[str(e)]['name'] for e in items]

			print(champion_dto["name"], len(items), itemStrings)

			writer.writerow(features)

def getRecommended(champ):
	items = []
	for recommended in champ['recommended']:
		if recommended['mode'] == 'CLASSIC' and recommended['map'] == 'SR':
			for block in recommended['blocks']:
				if block['type'] == "essential":
					for item in block['items']:
						items.append(item['id'])
			for block in recommended['blocks']:
				if block['type'] == "standard" \
				or block['type'] == 'situational' \
				or block['type'] == 'offensive'\
				or block['type'] == 'defensive':
					for item in block['items']:
						items.append(item['id'])
	return items

def doPCA(components, num_clusters):
	names = []
	ids = []
	with open('champions.csv') as f:
		reader = csv.reader(f)
		next(reader)
		for r in reader:
			names.append(r[1])
			ids.append(r[0])
	print("pca")
	table = np.genfromtxt('champions.csv', delimiter=',')
	X = table[1:,2:]
	X = preprocessing.scale(X)
	print(X)
	print(np.shape(X))
	pca = PCA(n_components=15)
	pca.fit(X)
	print(pca.explained_variance_ratio_)
	X2 = pca.transform(X)
	print(X2)
	print(np.shape(X2))
	labels = doKmeans(X2, 5)


	with open('clusters.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		for i in range(len(names)):
			writer.writerow([ids[i], names[i], labels[i]])
	pca = PCA(n_components=2)
	pca.fit(X)
	X3 = pca.transform(X)
	#tsne = TSNE(n_components=2)
	#X3 = tsne.fit_transform(X2)


	plt.title('Champion Clusters')
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	markers = ['o', '+']
	for i in range(np.shape(X)[0]):
		label = labels[i]
		color = colors[label%7]
		marker = markers[label//7]
		style = color + marker
		plt.plot(X3[i][0], X3[i][1], style)
		plt.text(X3[i][0], X3[i][1], names[i])

	plt.show()


def doKmeans(X, n):
	kmeans = KMeans(n_clusters=n, random_state=0)
	kmeans.fit(X)
	distances = kmeans.transform(X)
	clusters = np.argmin(distances, axis=1)
	return clusters


def main():
	print("1 - run PCA")
	print("2 - load champ features")
	print("9 - exit")
	num = input("enter command: ")
	if num == '1':
		components = input("dimension reduce to: ")
		clusters = input("number of clusters: ")
		doPCA(components, clusters)
	elif num == '2':
		LoadChampFeatures()
	elif num == '3':
		pass
	#LoadItemData()


main()

