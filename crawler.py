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
	self.connection.commit()


def exists_account_id(account_id, season_id = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id == 0:
		cur.execute("FROM users SELECT * WHERE aid = ?", account_id)
	else:
		cur.execute("FROM users SELECT * WHERE aid=:aid and season{}=:1".format(season_id), {"aid":account_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True

def exists_summoner_id(summoner_id, season = 0):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if season_id = 0:
		cur.execute("FROM users SELECT * WHERE sid = ?", summoner_id)
	else:
		cur.execute("FROM users SELECT * WHERE sid=:sid and season{}=:1".format(season_id), {"sid":account_id})
	row = cur.fetchone()
	if row == None:
		return False
	return True








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



def collect_league(seed_id):
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
			if (exists_summoner_id(summ
				oner_id, season=SEASON_ID)):
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

def main():
	create_tables()
	seeds = []
	while True:
		seed = seeds.pop(0)
		new_seeds = collect_league(seeds[0])
		seeds = seeds + new_seeds

BRONZE = 123
SILVER = 123
GOLD = 123
PLATINUM = 123
DIAMOND = 123
SEASON_ID = 9
QUEUE_ID = 420

seeds = []
main()