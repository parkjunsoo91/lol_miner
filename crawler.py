import sys
import time
from datetime import date
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
		cur.execute("INSERT INTO matchlist values (?, ?)", 
					(aid, json.dumps(matchlist_dto),))
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
			cur.execute("""SELECT accountId1,accountId2,accountId3,accountId4,accountId5,
						accountId6,accountId7,accountId8,accountId9,accountId10,winner 
						from matches where gameId = ?""",(game_id,))
			matchrow = cur.fetchone()
			if matchrow == None:
				continue
			if (aid in matchrow[:5] and matchrow[10] == 100 or
				aid in matchrow[5:] and matchrow[10] == 200):
				win = True
			else:
				win = False
			match_reference_dto['win'] = win
		cur.execute("UPDATE matchlist SET matchlist=? where aid=?",
					(json.dumps(matchlist_dto), aid,))
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
		accountId1 integer, accountId2 integer, accountId3 integer, accountId4 integer, accountId5 integer,
		accountId6 integer, accountId7 integer, accountId8 integer, accountId9 integer,	accountId10 integer,
		pick1 integer, pick2 integer, pick3 integer, pick4 integer, pick5 integer,
		pick6 integer, pick7 integer, pick8 integer, pick9 integer, pick10 integer,
		ban1 integer, ban2 integer, ban3 integer, ban4 integer, ban5 integer,
		ban6 integer, ban7 integer, ban8 integer, ban9 integer, ban10 integer,
		data text)''')
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

'''
initialize tables
given a seed, collect match histories for ppl in same league
collect match info for everyone in user table

collect people matches by league (fill both users and matches)
collect matchlist for each person


apikey = 1 2 (NA) 3 4 5 (KR)

0. collect user list.
1. collect matchlists.
2. collect match info.

3. make matchlists with win-loss and kda statistics for matchlists

5. update matchlists.
6. update match info

'''

def collect_user_list():
	pass
def collect_matchlists():
	pass
def collect_matchinfo():
	tier = input("tier?:")
	key = input("key number?:")
	global API_KEY
	global TIER
	TIER = tier
	if key == '1':
		API_KEY = NA1
	if key == '2':
		API_KEY = NA2
	if key == '3':
		API_KEY = KR1
	if key == '4':
		API_KEY = KR2
	if key == '5':
		API_KEY = KR3
	global api
	api = RiotAPICaller(API_KEY)

	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	#cur.execute('SELECT * FROM matchlist')
	cur.execute("""SELECT matchlist.aid, users.tier, matchlist.matchlist 
		FROM matchlist INNER JOIN users ON matchlist.aid = users.aid""")
	rows = cur.fetchall()
	for row in rows:
		tier = row[1]
		if tier == TIER:
			matchlist_dto = json.loads(row[2])
			for match_reference_dto in matchlist_dto['matches']:
				queue = match_reference_dto['queue']
				season = match_reference_dto['season']
				game_id = match_reference_dto['gameId']
				timestamp = match_reference_dto['timestamp']//1000
				#recording only ranked games for now
				if (queue == 4 or queue == 42 or queue == 410 or
					queue == 420 or queue == 440):
					cur.execute('SELECT count(*) FROM matches WHERE gameId=?',(game_id,))
					if cur.fetchone()[0] == 0:
						match_dto = api.get_match(game_id)
						if match_dto == None or match_dto == 404:
							print("404 for season{}, queue{}, {}, {}".format(season, queue, str(date.fromtimestamp(timestamp)), timestamp))
							continue
						record_match(match_dto)

def add_winloss_kda():
	pass

def check_missing_matches():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute('SELECT * FROM matchlist')
	rows = cur.fetchall()
	queues = {}
	for row in rows:
		matchlist_dto = json.loads(row[1])
		for match_reference_dto in matchlist_dto['matches']:
			queue = match_reference_dto['queue']
			if (queue != 4 and queue != 42 and 
				queue != 420 and queue!= 440):
				continue
			season = match_reference_dto['season']
			game_id = match_reference_dto['gameId']
			cur.execute('SELECT count(*) FROM matches WHERE gameId=?',(game_id,))
			if cur.fetchone()[0] == 0:
				if queue in queues:
					if season in queues[queue]:
						queues[queue][season] += 1
					else:
						queues[queue][season] = 1
				else:
					queues[queue] = {season:1}
	print(queues)

def check_table_matches():
	cur = sqlite3.connect('loldata2.db').cursor()
	cur.execute('SELECT count(*) FROM matches')
	print(cur.fetchone()[0])

def check_matchlist_queues():
	cur = sqlite3.connect('loldata2.db').cursor()
	cur.execute('SELECT * FROM matchlist')
	rows = cur.fetchall()
	queues = {}
	for row in rows:
		matchlist_dto = json.loads(row[1])
		for match_reference_dto in matchlist_dto['matches']:
			queue = match_reference_dto['queue']
			season = match_reference_dto['season']
			game_id = match_reference_dto['gameId']
			if queue in queues:
				if season in queues[queue]:
					queues[queue][season] += 1
				else:
					queues[queue][season] = 1
			else:
				queues[queue] = {season:1}
	for q in sorted(queues):
		print("queue {}: ".format(q), end='')
		for s in queues[q]:
			print("(s{}:{})".format(s, queues[q][s]), end='')
		print()

def check_timestamp():
	cur = sqlite3.connect('loldata2.db').cursor()
	cur.execute('SELECT * FROM matchlist')
	rows = cur.fetchall()
	count = 0
	for row in rows:
		matchlist_dto = json.loads(row[1])
		for match_reference_dto in matchlist_dto['matches']:
			timestamp = match_reference_dto['timestamp'] // 1000
			if timestamp < 1412916917:
				print(date.fromtimestamp(timestamp))
				count += 1
	print(count)

def check_match_queues():
	cur = sqlite3.connect('loldata2.db').cursor()
	cur.execute('SELECT queueId, seasonId FROM matches')
	rows = cur.fetchall()# this takes up the most time...
	queues = {}
	for row in rows:
		queue = row[0]
		season = row[1]
		if queue in queues:
			if season in queues[queue]:
				queues[queue][season] += 1
			else:
				queues[queue][season] = 1
		else:
			queues[queue] = {season:1}
	for q in sorted(queues):
		print("queue {}: ".format(q), end='')
		for s in queues[q]:
			print("(s{}:{})".format(s, queues[q][s]), end='')
		print()



def update_static_data():
	champion_list_dto = api.get_champions()
	if champion_list_dto == None or type(champion_list_dto) is int:
		print("champion list not updated")
	else:
		with open('champions.json', 'w') as f:
			json.dump(champion_list_dto, f)
	item_list_dto = api.get_items()
	if item_list_dto == None or type(item_list_dto) is int:
		print("item list not updated")
	else:
		with open('items.json', 'w') as f:
			json.dump(item_list_dto, f)

def main():
	global api
	api = RiotAPICaller(API_KEY)
	print("1 - ")
	print("2 - collect matches")
	print("3 - check missing matches")
	print("4 - check_db")
	print("5 - check_matchlist queues")
	print("6 - check match queues")
	print("7 - check timestamp")
	print("8 - update static data")
	print("9 - exit")
	num = input("enter command: ")
	if num == '1':
		pass
	elif num == '2':
		collect_matchinfo()
	elif num == '3':
		check_missing_matches()
	elif num == '4':
		check_table_matches()
	elif num == '5':
		check_matchlist_queues()
	elif num == '6':
		check_match_queues()
	elif num == '7':
		check_timestamp()
	elif num == '8':
		update_static_data()
	elif num == '9':
		return
	return
	
	create_matchlist_table()
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

NA1 = "RGAPI-e25c265e-685b-4fbc-8350-6245d0a04237" #silver & Bronze
NA2 = "RGAPI-a313e30a-99f7-4917-8a0f-9e953ca2611b" #gold
KR1 = "RGAPI-3b09835e-2116-41b2-aa7e-2ecdf4f9033f" #challenger
KR2 = "RGAPI-e7174d16-1448-41d0-ab19-659f5a561bfe" #master & PLAT
KR3 = "RGAPI-7bd52325-6d2f-4d77-b420-bd28099188cc" #diamond

API_KEY = NA1
TIER = "SILVER"
SEED = CHALLENGER

main()
