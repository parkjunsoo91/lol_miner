
import csv
import json
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
		champ_data = json.load(f)['data']
	with open('items.json') as f:
		item_data = json.load(f)['data']
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

		#lane info... TBA

		#4 info, 20 stats
		champ = champ_data['1']
		for k in champ['info']:
			fieldnames.append(k)
		for k in champ['stats']:
			fieldnames.append(k)

		#items
		items = {}
		for key in champ_data:
			champ = champ_data[key]
			for item in getRecommended(champ):
				items[item] = 0
		
		item_count = len(items)
		for k in sorted(items):
			fieldnames.append(k)
		print(len(fieldnames))


		writer = csv.DictWriter(g, fieldnames=fieldnames)
		writer.writeheader()

		for key in sorted(champ_data):
			champ = champ_data[key]
			features = {}

			#initialize fields to default values
			for field in fieldnames:
				features[field] = 0

			features['Id'] = key
			features['Name'] = champ["name"]

			for t in champ['tags']:
				tag_string = t
				if "melee" in tag_string:
					tag_string = "Tank"
				features[tag_string] = 1
			features.update(champ['info'])
			features.update(champ['stats'])

			items = getRecommended(champ)
			for item in items:
				features[item] = 1
			
			itemStrings = [item_data[str(e)]['name'] for e in items]

			print(champ["name"], len(items), itemStrings)

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

def doPCA():
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
	for i in range(np.shape(X)[0]):
		if labels[i] == 0:
			color = 'ro'
		elif labels[i] == 1:
			color = 'r^'
		elif labels[i] == 2:
			color = 'yo'
		elif labels[i] == 3:
			color = 'go'
		elif labels[i] == 4:
			color = 'bo'
		else:
			color = 'b^'
		plt.plot(X3[i][0], X3[i][1], color)
		plt.text(X3[i][0], X3[i][1], names[i])

	plt.show()




def doKmeans(X, n):
	kmeans = KMeans(n_clusters=n, random_state=0)
	kmeans.fit(X)
	distances = kmeans.transform(X)
	clusters = np.argmin(distances, axis=1)
	return clusters

		
#LoadItemData()
LoadChampFeatures()
doPCA()


