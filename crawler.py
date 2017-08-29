import sys
import time
from urllib.parse import quote
import json
import sqlite3
from riot_api import *


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

def create_matchlist_table():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if True:
		cur.execute("DROP TABLE matchlist")
		cur.execute('''CREATE TABLE matchlist (
			aid integer UNIQUE,
			matchlist text)''')
		connection.commit()

	cur.execute("SELECT aid from users")
	rows = cur.fetchall()
	for row in rows:
		aid = row[0]
		cur.execute("SELECT count(*) from matchlist where aid = ?", (aid,))
		if cur.fetchone()[0] >= 1:
			continue
		matchlist = []
		matchlist_dto = api.get_matchlist(aid)
		cur.execute("INSERT INTO matchlist values (?, ?)", (aid, json.dumps(matchlist_dto),))
		connection.commit()

def add_winloss_info():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM matchlist")
	rows = cur.fetchall()
	accounts = []
	for row in rows:
		aid = row[0]
		matchlist_dto = json.loads(row[1])
		for match_reference_dto in matchlist_dto['matches']:
			if 'win' in match_reference_dto:
				continue
			game_id = match_reference_dto['gameId']
			cur.execute("SELECT accountId1,accountId2,accountId3,accountId4,accountId5,accountId6,accountId7,accountId8,accountId9,accountId10,winner from matches where gameId = ?",(game_id,))
			matchrow = cur.fetchone()
			if matchrow == None:
				continue
			if aid in matchrow[:5] and matchrow[10] == 100 or aid in matchrow[5:] and matchrow[10] == 200:
				win = True
			else:
				win = False
			match_reference_dto['win'] = win
		cur.execute("UPDATE matchlist SET matchlist=? where aid=?",(json.dumps(matchlist_dto), aid,))
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


#seasonid 0~9
def exists_account_id(account_id, season_id = None):
	return exists_user(account_id, None, season_id)

def exists_summoner_id(summoner_id, season_id = None):
	return exists_user(None, summoner_id, season_id)

def exists_user(account_id = None, summoner_id = None, season_id = None):
	assert (season_id == None) or (0 <= season_id and season_id <= 9)
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	if account_id == None:
		assert summoner_id != None
		if season_id == None:
			cur.execute("SELECT count(*) FROM users WHERE aid = ?", (summoner_id,))
		else:
			cur.execute("SELECT count(*) FROM users WHERE aid = ? and season{} = 1".format(season_id), (summoner_id,))
	else:
		if season_id == None:
			cur.execute("SELECT count(*) FROM users WHERE aid = ?", (account_id,))
		else:
			cur.execute("SELECT count(*) FROM users WHERE aid = ? and season{} = 1".format(season_id), (account_id,))
	count = cur.fetchone()[0]
	if count == 0:
		return False
	elif count == 1:
		return True
	else:
		assert False

def exists_match(game_id):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT count(*) FROM matches WHERE gameId = ?", (game_id,))
	count = cur.fetchone()[0]
	if count == 0:
		return False
	elif count == 1:
		return True
	else:
		assert False

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
	if len(match_dto['participantIdentities']) != 10:
		return
	if len(match_dto['participants']) != 10:
		return
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




def get_seed(match_dto):
	seeds = []
	for participant_identity_dto in match_dto['participantIdentities']:
		seeds.append(participant_identity_dto['player']['summonerId'])
	return seeds


def collect_league(seed_id):
	seed_summoner_id = seed_id
	seed_summoner_ids = []
	#api call for league
	league_list_set_dto = api.get_league(seed_summoner_id)
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
			summoner_dto = api.get_summoner_by_summoner_id(summoner_id)
			account_id = summoner_dto['accountId']
			#api call for matchlist
			matchlist_dto = api.get_matchlist(account_id, season_id=SEASON_ID)
			if matchlist_dto != None:
				for match_reference_dto in matchlist_dto['matches']:
					game_id = match_reference_dto['gameId']
					if exists_match(game_id):
						continue
					#api call for match
					match_dto = api.get_match(game_id)
					record_match(match_dto)
					seed_summoner_ids += get_seed(match_dto)
			record_user(account_id, summoner_id, tier, SEASON_ID)
	return seed_summoner_ids

def record_user_matchlist(account_id):
	return

def collect_all_season(account_id):
	seasons = [9,8,7,6,5,4,3,2,1,0]
	for season_id in seasons:
		#check if the season is already covered.
		if exists_account_id(account_id, season_id = season_id):
			continue
		#api call for matchlist
		matchlist_dto = api.get_matchlist(account_id, season_id = season_id)
		if matchlist_dto != None:
			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				if exists_match(game_id):
					continue
				#api call for match
				match_dto = api.get_match(game_id)
				if match_dto == 404:
					continue
				record_match(match_dto)
		record_user(account_id, None, None, season_id)

def collect_all_players_history(tier):
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM users WHERE tier = ?", (tier,))
	rows = cur.fetchall()
	for row in rows:
		account_id = row[0]
		collect_all_season(account_id)

def check_entry():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('select * from matchlist')
	row = cur.fetchone()
	print (row)


def main():
	global api
	api = RiotAPICaller(API_KEY)
	#create_matchlist_table()
	add_winloss_info()
	return

	#collect_all_players_history(TIER)
	#return
	#create_user_table()
	#create_match_table()
	collect_league(SEED)
	return
	seeds = [SEED]
	while True:
		seed = seeds.pop(0)
		new_seeds = collect_league(seed)
		seeds = seeds + new_seeds

api = None
counter = 0
BRONZE = 0
SILVER = 2833703
GOLD = 4023441
PLATINUM = 1647596
DIAMOND = 2109280
MASTER = 1229790
CHALLENGER = 1222794
SEASON_ID = 8
QUEUE_ID = 420

NA1 = "RGAPI-51faa4f3-151a-4209-b1a4-ddaf1c4c05b7" #silver & Bronze
NA2 = "RGAPI-3308a6f6-e618-4d85-b76b-955241b83999" #gold
KR1 = "RGAPI-6e82a0fe-af53-4221-8a20-9058ac557093" #challenger
KR2 = "RGAPI-eab1046b-4a33-4f30-81fa-4743e0eb451f" #master & PLAT
KR3 = "RGAPI-c2f60718-41df-411d-9ca9-28680b1e0a28" #diamond

API_KEY = NA1
TIER = "SILVER"
SEED = CHALLENGER

main()
