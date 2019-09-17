import json
import sqlite3
from riot_api import *
from scraper import *

seeds = {'2': {'sid':3836801, 'region': 'KR'},#이것도쓰이냐 silver
		'8': {'sid':52845797, 'region': 'NA1'},#toxicyoshi7, gold
		'4': {'sid':11727339, 'region': 'KR'},#L o k, platinum
		'3': {'sid':2758443, 'region': 'KR'},#상꾸꾸꾸꾸, gold
		'1': {'sid':21814567, 'region': 'KR'}, #THe Androidian, bronze
		'5': {'sid':3477769, 'region': 'KR'}, #LPL is my dream, diamond3
		'10': {'sid':73170103, 'region': 'NA1'}, # no regret, diamond3
		'9': {'sid':67579899, 'region': 'NA1'}, #kadian, plat3
		'7': {'sid':34941132, 'region': 'NA1'}, #ancient zodiac, silver3
		'6': {'sid':71331250, 'region': 'NA1'}, #sting685, bronze2
		'11': {'sid':23830827, 'region': 'EUW'}, #Bugggy, bronze3
		'12': {'sid':72359917, 'region': 'EUW'}, #stewee98, silver3
		'13': {'sid':93736809, 'region': 'EUW'}, #SQL WildCard, gold4
		'14': {'sid':32351965, 'region': 'EUW'}, #xMaryl, plat3
		'15': {'sid':43571592, 'region': 'EUW'}, #Faced My Fears, dia3
		}
keys = {'1': "RGAPI-4b89e5af-0c4e-4f98-9350-e5c143d43b7e",
		'2': "RGAPI-7fc2c501-3369-42a4-9211-bfee3f8ebc70",
		'3': "RGAPI-e0ba00c4-815f-4db0-91b0-be8b9cc16a14",
		'4': "RGAPI-b0816323-d917-4743-8805-a2c9b28f6b54",
		'5': "RGAPI-b0816323-d917-4743-8805-a2c9b28f6b54",
		'6': "RGAPI-2BADD8BE-6261-495E-9DB4-A6728B10BCA8"} 

class DataManager:

	def __init__(self):
		self.menu()
		
	'''prompt menu'''
	def menu(self):
		print("-----Welcome to Data Manager-----")
		print("1: collect all")
		print("2: init tables")
		print("3: opgg_crawl")
		print("4: fill incomplete collections")
		print("5: remove records without tier history")
		print("8: db stats")
		print("9: test routine")
		print("0: interactive mode")


		command = input("enter command: ")

		if command == '1':
			seed_id = input("enter seed id: ")
			key_id = input("enter key id: ")
			self.region = seeds[seed_id]['region']
			self.key = keys[key_id]
			self.api = RiotAPICaller(self.key, self.region)
			self.seed_ids = [seeds[seed_id]['sid']]
			self.scraper = Scraper(self.region)
			self.collect_recursively()
		elif command == '2':
			self.region = input("enter region: ")
			self.create_table_summoners()
			self.create_table_matchlists()
			self.create_table_matches()
			self.create_table_tier_history()
		elif command == '3':
			#TO BE IMPLEMENTED, DO NOT USE YET
			self.api = RiotAPICaller(self.key, self.region)
			self.scraper = Scraper(self.region)
			self.opgg_crawl()
		elif command == '4':
			self.key = keys['6']
			regions = ['KR', 'NA1', 'EUW']
			for region in regions:
				self.region = region
				self.api = RiotAPICaller(self.key, self.region)
				self.fill_incomplete()
		elif command == '5':
			regions = ['KR', 'NA1', 'EUW']
			for region in regions:
				self.region = region
				self.remove_ones_without_tierhistory()
		elif command == '8':
			self.db_statistics()
		elif command == '9':
			key_id = input("enter key id: ")
			self.region = "KR"
			self.key = keys[key_id]
			self.api = RiotAPICaller(self.key, self.region)
			self.scraper = Scraper(self.region)
			self.test_routine()
		elif command == '0':
			return
		self.menu()

	def remove_ones_without_tierhistory(self):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT matchlists.aid, summoners.sid FROM matchlists INNER JOIN summoners ON matchlists.aid = summoners.aid")
		rows = cur.fetchall()
		count = 0
		for row in rows:
			aid = row[0]
			sid = row[1]
			cur.execute("SELECT count(*) FROM tier_history WHERE aid=?",(sid,))
			if cur.fetchone()[0] == 0:
				#self.remove_matchlist(aid)
				count+=1
		print (count)
		#connection.commit()

	def db_statistics(self):
		regions = ['KR', 'NA1', 'EUW']
		for region in regions:
			connection = sqlite3.connect(region + '.db')
			cur = connection.cursor()
			cur.execute("SELECT tier FROM summoners")
			rows = cur.fetchall()
			tiers = [0,0,0,0,0,0,0]
			for row in rows:
				tiers[row[0]-1] += 1
			print (tiers)

			cur.execute("SELECT aid FROM tier_history")
			rows = cur.fetchall()
			print(len(rows))

	def test_routine(self):
		success, data = self.scraper.get_user("롤롤시지")
		
		print(data)


	def collect_recursively(self):
		while len(self.seed_ids) > 0:
			next_sid = self.seed_ids.pop()

			#look up his league
			status, leagues = self.api.get_position(next_sid)
			if status != 200:
				continue
			league_position_dto = None
			for l in leagues:
				if l['queueType'] == "RANKED_SOLO_5x5":
					league_position_dto = l
			if league_position_dto == None:
				continue
			league_id = league_position_dto['leagueId']

			self.collect_league(league_id)
			#as a side effect, a queue is filled up with potential next seed candidates



	'''
	parameter: league id (int)
	collect season 9 match info for everyone in the league.
	fill up queue with next seed summoner ids
	'''
	def collect_league(self, league_id):
		status, league_list_dto = self.api.get_league(league_id)
		if status != 200:
			return
		tier = league_list_dto['tier']

		#for every person look up his aid
		for league_item_dto in league_list_dto['entries']:
			sid = league_item_dto['playerOrTeamId']
			rank = league_item_dto['rank']
			name = league_item_dto['playerOrTeamName']

			if self.exists_tier_history(sid):
				pass
			else:			
				success, userdata = self.scraper.get_user(name)
				if success == False:
					continue
				if self.is_adequate(userdata):
					self.record_tier_history(sid, userdata)
				else:
					continue

			success, summoner_dto = self.get_summoner(sid, save=True)
			if success == False:
				continue
			self.update_tier(summoner_dto["accountId"], tier, rank)

			aid = summoner_dto['accountId']
			success, matchlist_dto = self.get_matchlist(aid, season=9, save=True) #season7
			if success == False:
				continue
			#print(matchlist_dto)
			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				queue = match_reference_dto['queue']
				season = match_reference_dto['season']
				if queue not in [420] or season != 9:
					continue
				success, match_dto = self.get_match(game_id, save=True)
				if success == False:
					continue
				if len(self.seed_ids) < 50:
					for p_id_dto in match_dto['participantIdentities']:
						self.seed_ids.append(p_id_dto['player']['summonerId'])
	'''
	data in format given by scraper
	policy for deciding adquately active user for data collection
	{'recent': [{tier, rank, point, month}, ...], 'past':[{season, tier}, ...]}
	'''
	def is_adequate(self, userdata):
		recent = userdata['recent']
		month_count = len(recent)
		if month_count < 10:
			return False
		return True

	def fill_incomplete(self):
		#check opgg entries in db and perform collection
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT aid FROM tier_history")
		rows = cur.fetchall()
		for row in rows:
			if row == None:
				print("nothing bro")
				break
			sid = row[0]

			success, summoner_dto = self.get_summoner(sid, save=True)
			if success == False:
				continue

			aid = summoner_dto['accountId']
			success, matchlist_dto = self.get_matchlist(aid, season=9, save=True) #season7
			if success == False:
				continue
			#print(matchlist_dto)
			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				queue = match_reference_dto['queue']
				season = match_reference_dto['season']
				if queue not in [420] or season != 9:
					continue
				success, match_dto = self.get_match(game_id, save=True)
				if success == False:
					continue

	'''
	get from db, else get from api.
	if save=True, then save in db
	parameters: summoner id(int), save(bool)
	return success(bool), data(dict)
	'''
	def get_summoner(self, sid, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM summoners WHERE sid = ?", (sid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, {'accountId': rows[0][0], 'id': rows[0][1]} 
		else:
			status, summoner_dto = self.api.get_summoner_by_summoner_id(sid)
			if status != 200:
				return False, summoner_dto
			if save == True:
				self.record_summoner(summoner_dto)
			return True, summoner_dto

	def update_tier(self, aid, tier, rank):
		tier_val = 0
		tier_code = ["BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND", "MASTER", "CHALLENGER"]
		tier_val = tier_code.index(tier) + 1
		assert(tier_val != 0)
		rank_val = 0
		rank_code = ["V", "IV", "III", "II", "I"]
		rank_val = rank_code.index(rank) + 1
			
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("UPDATE summoners SET tier=?, rank=? where aid=?",
					(tier_val, rank_val, aid,))
		connection.commit()

	'''	
	get from db, else get from api.
	if save=True, then save in db
	if season=0, collect all seasons.
	parameters: summoner id(int), save(bool)
	return success(bool), data(dict)
	'''
	def get_matchlist(self, aid, season=0, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM matchlists WHERE aid = ?", (aid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, json.loads(rows[0][1])
		else:
			status, matchlist_dto = self.api.get_matchlist(aid, season_id=season)
			if status != 200:
				return False, matchlist_dto
			if save == True:
				self.record_matchlist(aid, matchlist_dto)
			return True, matchlist_dto

	'''
	get from db, else get from api.
	if save=True, then save in db
	parameters: game id(int), save(bool)
	return success(bool), data(dict)
	'''
	def get_match(self, game_id, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM matches WHERE gameId = ?", (game_id,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, json.loads(rows[0][4])
		else:
			status, match_dto = self.api.get_match(game_id)
			if status != 200:
				return False, match_dto
			if save == True:
				self.record_match(match_dto)
			return True, match_dto

	def exists_tier_history(self, sid):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT aid FROM tier_history WHERE aid = ?", (sid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True
		else:
			return False
	'''
	record functions - record data objects into sql tables
	'''
	def record_summoner(self, summoner_dto):
		aid = summoner_dto['accountId']
		sid = summoner_dto['id']
		#tier and rank later to be written as 1~7, 1~5
		tier = 0
		rank = 0
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("INSERT OR IGNORE INTO summoners VALUES (?,?,?,?)",
					(aid, sid, tier, rank))
		connection.commit()

	'''
	record functions - record data objects into sql tables
	data format: {'totalGames':int, 'matches'[...]}
	TODO: if entry exists, add unexisting ones into the original.
	for now: always update matchlist
	'''
	def record_matchlist(self, aid, matchlist_dto):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("UPDATE matchlists SET matchlist=? where aid=?",
					(json.dumps(matchlist_dto), aid,))
		cur.execute("INSERT OR IGNORE INTO matchlists VALUES (?,?)",
					(aid, json.dumps(matchlist_dto),))
		connection.commit()

	def remove_matchlist(self, aid):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("DELETE FROM matchlists where aid=?", (aid,))
		connection.commit()

	
	'''
	record functions - record data objects into sql tables
	'''		
	def record_match(self, match_dto):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?)",
					(match_dto['gameId'],
					match_dto['seasonId'],
					match_dto['queueId'],
					match_dto['gameVersion'],
					json.dumps(match_dto),))
		connection.commit()

	def record_tier_history(self, sid, userdata):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("INSERT OR IGNORE INTO tier_history VALUES (?,?,?)",
					(sid,
					json.dumps(userdata['recent']),
					json.dumps(userdata['past']),))
		connection.commit()


	def create_table_summoners(self):
		query = '''CREATE TABLE summoners (
			aid integer UNIQUE,
			sid integer,
			tier integer,
			rank integer)'''
		self.create_table(query)

	def create_table_matchlists(self):
		query = '''CREATE TABLE matchlists (
			aid integer UNIQUE,
			matchlist text)'''
		self.create_table(query)

	def create_table_matches(self):
		query = '''CREATE TABLE matches (
			gameId integer UNIQUE,
			seasonId integer,
			queueId integer,
			gameVersion text,
			match text)'''
		self.create_table(query)

	def create_table_tier_history(self):
		query = '''CREATE TABLE tier_history (
			aid integer UNIQUE, #this is a misnomer: it contains SID
			recent text,
			past text)'''
		self.create_table(query)

	def create_table(self, query):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute(query)
		connection.commit()

d = DataManager()