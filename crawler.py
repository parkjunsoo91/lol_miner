import sys
import time
import http.client
from urllib.parse import quote
import json
import sqlite3







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
		data text
		''')
	connection.commit()


def exists_account_id(account_id, season_id = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id == 0:
		cur.execute("SELECT * FROM users WHERE aid = ?", account_id)
	else:
		cur.execute("SELECT * FROM users SELECT * WHERE aid=:aid and season{}=:1".format(season_id), {"aid":account_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True

def exists_summoner_id(summoner_id, season_id = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id == 0:
		cur.execute("SELECT * FROM users WHERE sid = ?", summoner_id)
	else:
		cur.execute("SELECT * FROM users WHERE sid=:sid and season{}=1".format(season_id), {"sid":summoner_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True

def exists_match(game_id):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM matches WHERE gameID = ?", game_id)
	row = cur.fetchone()
	if row == None:
		return False
	return True

def record_match(match_dto):
	if exists_match():
		return
	values = {}
	values['seasonId'] = match_dto['seasonId']
	values['queueId'] = match_dto['queueId']
	values['gameId'] = match_dto['gameId']
	values['gameVersion'] = match_dto['gameVersion']
	values['platformId'] = match_dto['platformId']
	values['gameDuration'] = match_dto['gameDuration']
	values['gameCreation'] = match_dto['gameCreation']
	values['winner'] = 100 if match_dto['teams'][0]['win']=='Win' else 200
	for i in range (10):
		participant_identity_dto = match_dto['participantIdentities'][i]
		participantId = participant_identity_dto['participantId']
		values['accountId'+participantId] = participant_identity_dto['player']['accountId']

		partcipant_dto = match_dto['participants'][i]
		participantid = participant_dto['participantId']
		values['pick'+participantId] = participant_dto['championId']

		values['ban'+(i+1)] = 0

	for team_stats_dto in match_dto['teams']:
		for team_bans_dto in team_stats_dto['bans']:
			values['ban'+team_bans_dto['pickTurn']] = team_bans_dto['championId']

	values['data'] = json.dumps(match_dto)

	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('''INSERT INTO matches VALUES (
		gameId=:gameId
		seasonId=:seasonId,
		queueId=:queueId,
		gameVersion=:gameVersion,
		platformId=:platformId,
		gameDuration=:gameDuration,
		gameCreation=:gameCreation,
		winner=:winner,
		accountId1=:accountId1,	accountId2=:accountId2, accountId3=:accountId3,
		accountId4=:accountId4, accountId5=:accountId5, accountId6=:accountId6,
		accountId7=:accountId7, accountId8=:accountId8, accountId9=:accountId9,
		accountId10=:accountId10,
		pick1=:pick1, pick2=:pick2, pick3=:pick3, pick4=:pick4, pick5=:pick5,
		pick6=:pick6, pick7=:pick7, pick8=:pick8, pick9=:pick9, pick10=:pick10,
		ban1=:ban1, ban2=:ban2, ban3=:ban3, ban4=:ban4, ban5=:ban5,
		ban6=:ban6, ban7=:ban7, ban8=:ban8, ban9=:ban9, ban10=:ban10,
		data=:data)''', values)
	connection.commit()

def record_user(account_id, summoner_id, tier, season):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if exists_account_id(account_id) == False:
		values = {}
		values['aid'] = account_id
		values['sid'] = summoner_id
		values['tier'] = tier
		cur.execute('''INSERT INTO matches VALUES (
			aid=:aid,
			sid=:sid,
			tier=:tier,
			season0=0,
			season1=0,
			season2=0,
			season3=0,
			season4=0,
			season5=0,
			season6=0,
			season7=0,
			season8=0,
			season9=0)''', values)
		connection.commit()

	cur = connection.cursor()
	cur.execute("UPDATE accounts SET season{}=1 where aid = ?".format(seasonId), accountId)
	connection.commit()






def get_summoner_by_account_id(account_id):
	request_body = "/lol/summoner/v3/summoners/by-account/{}".format(account_id)
	return send_request(request_body)
	
def get_summoner_by_summoner_id(summoner_id):
	request_body = "/lol/summoner/v3/summoners/{}".format(summoner_id)
	return send_request(request_body)

def get_league(summoner_id):
	request_body = "/lol/league/v3/leagues/by-summoner/{}".format(summoner_id)
	return send_request(request_body)

def get_matchlist(account_id, queue_id=0, season_id = 0):
	request_body = "/lol/match/v3/matchlists/by-account/{}".format(account_id)
	if season_id != 0:
		request_body = request_body + "?season=" + str(season_id)
	return send_request(request_body)

def get_match(game_id):
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
			#api call for summoner info
			account_id = get_summoner_by_summoner_id(summoner_id)
			#api call for matchlist
			matchlist_dto = get_matchlist(account_id, queue_id=[QUEUE_ID], season_id=SEASON_ID)
			if matchlist_dto != None:
				for match_reference_dto in matchlist_dto['matches']:
					game_id = match_reference_dto['game_id']
					if exists_match(game_id):
						continue
					#api call for match
					match_dto = get_match(game_id)
					record_match(match_dto)
					seed_summoner_ids += get_seed(match_dto)
			record_user(account_id, summoner_id, tier, season)
	return seed_summoner_ids

def main():
	#create_user_table()
	#create_match_table()
	seeds = [SILVER]
	while True:
		seed = seeds.pop(0)
		new_seeds = collect_league(seed)
		seeds = seeds + new_seeds


API_NA = "RGAPI-8996de32-03b0-4029-bdf1-6ef1468a9966"
API_KR1 = ""
API_KR2 = ""
API_KR3 = ""

API_KEY = API_NA
BRONZE = 123
SILVER = 2833703
GOLD = 123
PLATINUM = 123
DIAMOND = 123
SEASON_ID = 9
QUEUE_ID = 420

main()