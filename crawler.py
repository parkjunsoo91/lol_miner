import sys
import time
import http.client
from urllib.parse import quote
import json
import sqlite3
import datetime






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
	connection.commit()

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
		winner integer,
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
		data text)
		''')
	connection.commit()


def exists_account_id(account_id, season_id = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id == 0:
		cur.execute("SELECT * FROM users WHERE aid = ?", (account_id,))
	else:
		cur.execute("SELECT * FROM users WHERE aid=:aid and season{}=:1".format(season_id), {"aid":account_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True

def exists_summoner_id(summoner_id, season_id = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id == 0:
		cur.execute("SELECT * FROM users WHERE sid = ?", (summoner_id,))
	else:
		cur.execute("SELECT * FROM users WHERE sid=:sid and season{}=1".format(season_id), {"sid":summoner_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True

def exists_match(game_id):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM matches WHERE gameId = ?", (game_id,))
	row = cur.fetchone()
	if row == None:
		return False
	return True

def record_match(match_dto):
	if exists_match(match_dto["gameId"]):
		return
	values = []
	values.append(match_dto['gameId'])
	values.append(match_dto['seasonId'])
	values.append(match_dto['queueId'])
	values.append(match_dto['gameVersion'])
	values.append(match_dto['platformId'])
	values.append(match_dto['gameDuration'])
	values.append(match_dto['gameCreation'])
	winner = 100 if match_dto['teams'][0]['win']=='Win' else 200
	values.append(winner)
	for i in range (10):
		participant_identity_dto = match_dto['participantIdentities'][i]
		values.append(participant_identity_dto['player']['accountId'])
	for i in range(10):
		participant_dto = match_dto['participants'][i]
		values.append(participant_dto['championId'])
	bans = []
	for i in range(2):
		team_stats_dto = match_dto['teams'][i]
		for j in range(5):
			if j < len(team_stats_dto['bans']):
				bans.append(team_stats_dto['bans'][j]['championId'])
			else:
				bans.append(0)
	values += bans
	values.append(json.dumps(match_dto))

	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('''INSERT INTO matches VALUES (
		?,?,?,?,?,?,?,?,
		?,?,?,?,?,?,?,?,?,?,
		?,?,?,?,?,?,?,?,?,?,
		?,?,?,?,?,?,?,?,?,?,
		?)''', tuple(values))
	connection.commit()

def record_user(account_id, summoner_id, tier, season_id):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if exists_account_id(account_id) == False:
		values = []
		values.append(account_id)
		values.append(summoner_id)
		values.append(tier)
		values += [0] * 10
		cur.execute('''INSERT INTO users VALUES (
			?,?,?,
			?,?,?,?,?,?,?,?,?,?
			)''', tuple(values))
		connection.commit()

	cur = connection.cursor()
	cur.execute("UPDATE users SET season{}=1 where aid = ?".format(season_id), (account_id,))
	connection.commit()






def get_summoner_by_account_id(account_id):
	print("getting summoner of aid", account_id)
	request_body = "/lol/summoner/v3/summoners/by-account/{}".format(account_id)
	return send_request(request_body)
	
def get_summoner_by_summoner_id(summoner_id):
	print("getting summoner of sid", summoner_id)
	request_body = "/lol/summoner/v3/summoners/{}".format(summoner_id)
	return send_request(request_body)

def get_league(summoner_id):
	print("getting league of sid", summoner_id)
	request_body = "/lol/league/v3/leagues/by-summoner/{}".format(summoner_id)
	return send_request(request_body)

def get_matchlist(account_id, queue_id=0, season_id = 0):
	print("getting matchlist for aid", account_id)
	request_body = "/lol/match/v3/matchlists/by-account/{}".format(account_id)
	if season_id != 0:
		request_body = request_body + "?season=" + str(season_id)
	return send_request(request_body)

def get_match(game_id):
	global counter
	counter += 1
	print("getting match of gameId", game_id, "(", counter, ")")
	request_body = "/lol/match/v3/matches/{}".format(game_id)
	return send_request(request_body)

	
def send_request(request_body):
	time.sleep(1.2)
	url = "kr.api.riotgames.com"
	key = API_KEY
	for i in range (5):
		connection = http.client.HTTPSConnection(url, timeout=10)
		response = None
		try:
			connection.request("GET", request_body, headers={'X-Riot-Token': key, })
			response = connection.getresponse()
			print(datetime.datetime.now(), response.status, response.reason)
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






def get_seed(match_dto):
	seeds = []
	for participant_identity_dto in match_dto['participantIdentities']:
		seeds.append(participant_identity_dto['player']['summonerId'])
	return seeds


def collect_league(seed_id):
	seed_summoner_id = seed_id
	seed_summoner_ids = []
	#api call for league
	league_list_set_dto = get_league(seed_summoner_id)
	if league_list_set_dto == None:
		return []
	for league_list_dto in league_list_set_dto:
		if league_list_dto['queue'] != "RANKED_SOLO_5x5":
			continue
		tier = league_list_dto['tier']
		for league_item_dto in league_list_dto['entries']:
			summoner_id = league_item_dto['playerOrTeamId']
			if exists_summoner_id(summoner_id, season_id=SEASON_ID):
				continue
			print(SEED)
			#api call for summoner info
			summoner_dto = get_summoner_by_summoner_id(summoner_id)
			account_id = summoner_dto['accountId']
			#api call for matchlist
			matchlist_dto = get_matchlist(account_id, queue_id=[QUEUE_ID], season_id=SEASON_ID)
			if matchlist_dto != None:
				for match_reference_dto in matchlist_dto['matches']:
					game_id = match_reference_dto['gameId']
					if exists_match(game_id):
						continue
					#api call for match
					match_dto = get_match(game_id)
					record_match(match_dto)
					seed_summoner_ids += get_seed(match_dto)
			record_user(account_id, summoner_id, tier, SEASON_ID)
	return seed_summoner_ids

def main():
	#create_user_table()
	#create_match_table()
	seeds = [SEED]
	while True:
		seed = seeds.pop(0)
		new_seeds = collect_league(seed)
		seeds = seeds + new_seeds

counter = 0
BRONZE = 0
SILVER = 2833703
GOLD = 0
PLATINUM = 0
DIAMOND = 2109280
MASTER = 1229790
CHALLENGER = 1222794
SEASON_ID = 9
QUEUE_ID = 420

KEY_NA = "RGAPI-8996de32-03b0-4029-bdf1-6ef1468a9966"
KEY_KR1 = "RGAPI-ed3577b4-fff3-4972-b66a-9d95c0d301f0"
KEY_KR2 = "RGAPI-c0e0da6b-8272-4f0a-a47d-ec5fc5d7cd32"
KEY_KR3 = "RGAPI-6fa1e75e-6810-4fe6-a7ff-f654e5e23613"

API_KEY = KEY_KR3
SEED = DIAMOND

main()