import sys
import time
import http.client
from urllib.parse import quote
import json
import sqlite3
import os.path
from pathlib import Path
from collections import deque

class RequesterClient:
	def __init__(self):
		self.region = "kr"
		self.url = self.region + ".api.riotgames.com"
		self.connection = http.client.HTTPSConnection(self.url, timeout=10)
		self.key1 = "RGAPI-0d12560d-f6b9-4d11-941c-8012137cb267" #assigned api key
		self.key = "RGAPI-a7138bc2-7e3e-4f12-95e5-370b3264bcab" #24h api key NA
		self.key2 = "RGAPI-1f9ff964-e27a-4321-8e33-4eced6284871" #24h api key KR

	def SendRequest(self, requestString):
		for i in range (5):
			self.connection = http.client.HTTPSConnection(self.url, timeout=10)
			r = None
			try:
				self.connection.request("GET", requestString, headers={'X-Riot-Token': self.key, })
				r = self.connection.getresponse()
				print(r.status, r.reason)
				if r.status == 200:
					b = r.read()
					dataObject = json.loads(b)
					self.connection.close()
					return dataObject
			except:
				print("error")
				if r != None:
					print(r.status, r.reason)
				self.connection.close()

			time.sleep(2)
			print("retrying...")
		return None

	def GetSummoner(self, name):
		time.sleep(1.2)
		print("getting summoner " + name + "...")
		return self.SendRequest("/lol/summoner/v3/summoners/by-name/" + quote(name))

	def GetMatchList(self, accountId, season=0):
		time.sleep(1.2)
		print("getting matchlist for account id {} ...".format(accountId))
		requestString = "/lol/match/v3/matchlists/by-account/{}".format(accountId)
		if season != 0:
			requestString = requestString + "?season={}".format(season)
		return self.SendRequest(requestString)

	def GetMatch(self, matchId):
		time.sleep(1.2)
		print("getting match for match id {}...".format(matchId))
		responseObject = self.SendRequest("/lol/match/v3/matches/{}".format(matchId))
		return responseObject

	def GetTimeline(self, matchId):
		time.sleep(1.2)
		print("getting timeline for match id {}...".format(matchId))
		return self.SendRequest("/lol/match/v3/timelines/by-match/{}".format(matchId))

	def GetChallengers(self, league, queue):
		if queue != "RANKED_SOLO_5x5" and queue != "RANKED_FLEX_SR":
			return None
		if league != "challengerleagues" and league != "masterleagues":
			return None
		time.sleep(1.2)
		print("getting leagues in {} {}...".format(league, queue))
		return self.SendRequest("/lol/league/v3/{}/by-queue/{}".format(league, queue))

	def GetLeague(self, summonerId):
		time.sleep(1.2)
		return self.SendRequest("")

	def GetChampions(self):
		time.sleep(1.2)
		return self.SendRequest("/lol/static-data/v3/champions?locale=en_US&tags=all&dataById=true")

	def GetItems(self):
		time.sleep(1.2)
		return self.SendRequest("/lol/static-data/v3/items?locale=en_US&tags=all")

class DBClient:
	def __init__(self):
		if not os.path.isfile('loldata.db'):
			self.connection = sqlite3.connect('loldata.db')
			self.CreateTables()
		else:
			self.connection = sqlite3.connect('loldata.db')

	def CreateTables(self):
		cur = self.connection.cursor()
		cur.execute('''CREATE TABLE matches(
			id integer UNIQUE,
			seasonid integer,
			queueid integer,
			gameversion varchar(32),
			platformid varchar(8),
			gameduration integer,
			gamecreation integer,
			winner integer,
			player1 integer,
			player2 integer,
			player3 integer,
			player4 integer,
			player5 integer,
			player6 integer,
			player7 integer,
			player8 integer,
			player9 integer,
			player10 integer,
			ban1 integer,
			ban2 integer,
			ban3 integer,
			ban4 integer,
			ban5 integer,
			ban6 integer,
			ban7 integer,
			ban8 integer,
			ban9 integer,
			ban10 integer,
			pick1 integer,
			pick2 integer,
			pick3 integer,
			pick4 integer,
			pick5 integer,
			pick6 integer,
			pick7 integer,
			pick8 integer,
			pick9 integer,
			pick10 integer)''')
		self.connection.commit()

	def CreatePlayerTable(self):
		cur = self.connection.cursor()
		cur.execute('''CREATE TABLE accounts (
			id integer UNIQUE,
			season3 integer,
			season4 integer,
			season5 integer,
			season6 integer,
			season7 integer)''')
		self.connection.commit()

	def MigrateTable(self):
		cur = self.connection.cursor()
		cur.execute('''CREATE TABLE accounts (
			id integer UNIQUE,
			season0 integer,
			season1 integer,
			season2 integer,
			season3 integer,
			season4 integer,
			season5 integer,
			season6 integer,
			season7 integer,
			season8 integer,
			season9 integer
			)''')
		self.connection.commit()
		cur.execute("SELECT * from ")

	def RunStatement(self, statement):
		cur = self.connection.cursor()
		cur.execute(statement)
		self.connection.commit()

	def AddMatch(self, matchObj):

		if self.MatchExists(matchObj['gameId']):
			return False

		metadata = [
		matchObj['gameId'],
		matchObj['seasonId'],
		matchObj['queueId'],
		matchObj['gameVersion'],
		matchObj['platformId'],
		matchObj['gameDuration'],
		matchObj['gameCreation'],
		]

		team1 = None
		team2 = None
		if matchObj['teams'][0]['teamId'] == 100:
			team1 = matchObj['teams'][0]
			team2 = matchObj['teams'][1]
		else:
			team1 = matchObj['teams'][1]
			team2 = matchObj['teams'][0]
		if team1['win'] == "Win":
			metadata.append(team1['teamId'])
		else:
			metadata.append(team2['teamId'])

		players = [p['player']['accountId'] for p in matchObj['participantIdentities']]

		b1 = [b['championId'] for b in team1['bans']]
		b2 = [b['championId'] for b in team2['bans']]
		while len(b1) != 5:
			b1.append(None)
		while len(b2) != 5:
			b2.append(None)
		bans = b1 + b2
		picks = [p['championId'] for p in matchObj['participants']]
		if len(picks) != 10:
			return False
		matchdatalist = metadata + players + bans + picks
		matchdata = tuple(matchdatalist)

		cur = self.connection.cursor()
		
		cur.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?)", matchdata)
		self.connection.commit()
		return True

	#called by MarkAccount
	def InsertAccount(self, accountId):
		accountStatus = (accountId, 0,0,0,0,0)
		cur = self.connection.cursor()
		cur.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?)", accountStatus)
		self.connection.commit()
		return

	#called by MarkAccount
	def UpdateAccount(self, accountId, seasonId, seasonVal):
		seasonId = "season" + str(seasonId)
		cur = self.connection.cursor()
		cur.execute("UPDATE accounts SET {} = ? where id = ?".format(seasonId), (seasonVal, accountId,))
		self.connection.commit()
		return

	#adds account if it doesnt exist. updates account if already exists.
	def MarkAccount(self, accountId, seasonId, seasonVal):
		cur = self.connection.cursor()
		cur.execute("SELECT * FROM accounts WHERE id = ?", (accountId,))
		row = cur.fetchone()
		if row == None:
			self.InsertAccount(accountId)
		self.UpdateAccount(accountId, seasonId, seasonVal)


	def MatchExists(self, matchId):
		cur = self.connection.cursor()
		cur.execute("SELECT * FROM matches WHERE id=?", (matchId,))
		if cur.fetchone() == None:
			return False
		return True

	def AccountExists(self, accountId):
		cur = self.connection.cursor()
		cur.execute("SELECT * FROM accounts WHERE id=?", (accountId,))
		if cur.fetchone() == None:
			return False
		return True

def unfold(data, order):
	LIST_UNFOLD = 10
	if type(data) == type({}):
		print("  " * order + "{")
		for k in list(data.keys()):
			print("  " * (order+1) + k + ":", end='')
			unfold(data[k], order+2)
		print("  " * order + "}")
			
	elif type(data) == type([]):
		print("")
		print("  " * order + "[")
		for i in range(min(len(data), LIST_UNFOLD)):
			unfold(data[i], order+1)
		if len(data) > LIST_UNFOLD:
			print("  " * order + "...")
		print("  " * order + "]")
	else:
		print("  " + str(data))

def getChampions():
	client = RequesterClient()
	with open('champions.json', 'w') as f:
		data = client.GetChampions()
		json.dump(data, f)

def getItems():
	client = RequesterClient()
	with open('items.json', 'w') as f:
		data = client.GetItems()
		json.dump(data, f)

def collect():		

	client = RequesterClient()
	db = DBClient()

	nameQueue = deque([])
	accountQueue = deque([])

	result = client.GetChallengers("challengerleagues", "RANKED_SOLO_5x5")
	for e in result['entries']:
		nameQueue.append(e['playerOrTeamName'])
	result = client.GetChallengers("challengerleagues", "RANKED_FLEX_SR")
	for e in result['entries']:
		nameQueue.append(e['playerOrTeamName'])
	result = client.GetChallengers("masterleagues", "RANKED_SOLO_5x5")
	for e in result['entries']:
		nameQueue.append(e['playerOrTeamName'])
	result = client.GetChallengers("masterleagues", "RANKED_FLEX_SR")
	for e in result['entries']:
		nameQueue.append(e['playerOrTeamName'])

	while len(nameQueue) > 0:
		playerName = nameQueue.popleft()
		result = client.GetSummoner(playerName)
		if result == None:
			continue
		accountId = result['accountId']
		matchQueue = deque([])
		result = client.GetMatchList(accountId, season=6)
		if result == None:
			continue
		for m in result['matches']:
			if not db.MatchExists(m['gameId']):
				matchQueue.append(m['gameId'])
		while len(matchQueue) > 0:
			matchId = matchQueue.popleft()
			result = client.GetMatch(matchId)
			if result == None:
				continue
			db.AddMatch(result)

		db.MarkAccount(accountId, 6, 1)
		
	print('done')

def collectPropagate():
	#getallmatches in match db
	return


def collect_all():
	challengers = []
	for P in challengers:
		check(P, season)
		matches = []
		matches = getMatches(P)
		for M in matches:
			wait(1)
			check(M)
			record(request(M))
		mark(P)

def collect_league(seed_id):
	QUEUE_ID = 999
	SEASON_ID = 999
	seed_summoner_ids = []
	summoner_dto = get_summoner_by_account_id(seed_id)
	seed_summoner_id = summoner_dto['id']
	#api call for league
	league_list_set_dto = get_league(seed_summoner_id)
	for league_list_dto in league_list_set_dto:
		if league_list_dto['queue'] != "RANKED_SOLO_5x5":
			continue
		tier = league_list_dto['tier']
		for league_item_dto in league_list_dto['entries']:
			summoner_id = league_item_dto['playerOrTeamId']
			if (exists_summoner_id(summoner_id, season=SEASON_ID)):
				continue
			#api call for summoner info
			account_id = get_summoner_by_summoner_id(summoner_id)
			#api call for matchlist
			matchlist_dto = get_matchlist(account_id, queue=[QUEUE_ID], season=SEASON_ID)
			if matchlist_dto != None:
				for match_reference_dto in matchlist_dto['matches']:
					game_id = match_reference_dto['game_id']
					#api call for match
					match_dto = get_match(game_id)
					record(match_dto)
					seeds.append(get_seed(match_dto))
			record(account_id, summoner_id, tier, season)
	return seed_summoner_ids

def create_user_table():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('''CREATE TABLE users (
		aid integer UNIQUE,
		sid integer,
		tier text,
		season0 integer,
		season1 integer,
		season2 integer,
		season3 integer,
		season4 integer,
		season5 integer,
		season6 integer,
		season7 integer,
		season8 integer,
		season9 integer)''')
	self.connection.commit()

def create_match_table():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('''CREATE TABLE matches (
		gameId integer UNIQUE,
		seasonId integer,
		queueId integer,
		gameVersion text,
		platformId text,
		gameDuration integer,
		gameCreation integer,
		accountId1 integer,
		accountId2 integer,
		accountId3 integer,
		accountId4 integer,
		accountId5 integer,
		accountId6 integer,
		accountId7 integer,
		accountId8 integer,
		accountId9 integer,
		accountId10 integer,
		pick1 integer,
		pick2 integer,
		pick3 integer,
		pick4 integer,
		pick5 integer,
		pick6 integer,
		pick7 integer,
		pick8 integer,
		pick9 integer,
		pick10 integer,
		ban1 integer,
		ban2 integer,
		ban3 integer,
		ban4 integer,
		ban5 integer,
		ban6 integer,
		ban7 integer,
		ban8 integer,
		ban9 integer,
		ban10 integer,
		winner integer,
		wholejson text

		''')
	self.connection.commit()


def exists_account_id(account_id):
	query = ""
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute(query)



def get_summoner_by_account_id(account_id):
	request_body = ""
	return = send_request(request_body)
	
def get_summoner_by_summoner_id(summoner_id):
	request_body = ""
	return send_request(request_body)

def get_league(summoner_id):
	request_body = ""
	return send_request(request_body)
	
def send_request(request_body):
	url = ""
	key = ""
	for i in range (5):
		connection = http.client.HTTPSConnection(url, timeout=10)
		response = None
		try:
			connection.request("GET", request_body, headers={'X-Riot-Token': key, })
			response = connection.getresponse()
			print(response.status, response.reason)
			if response.status == 200:
				b = response.read()
				dataObject = json.loads(b)
				connection.close()
				return dataObject
		except:
			print("error")
			if response != None:
				print(response.status, response.reason)
			connection.close()
		time.sleep(1.2)
		print("retrying...")
	return None



def main():
	collect()

if __name__ == '__main__':
	main()
#getItems()
#getChampions()


