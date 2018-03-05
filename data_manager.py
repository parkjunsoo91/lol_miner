import json
import sqlite3
from riot_api import *

seeds = {'1': {'sid':3836801, 'region':'KR'}}
keys = {'1': "RGAPI-2e71e249-7d5d-4593-bf52-8a8cdc656dbb"}
class DataManager:

	def __init__(self):
		while True:
			self.menu()
		
	'''prompt menu'''
	def menu(self):
		print("-----Welcome to Data Manager-----")
		print("1: collect all")
		print("2: init tables")
		print("")

		command = input("enter command: ")

		if command == '1':
			seed_id = input("enter seed id: ")
			key_id = input("enter key id: ")
			self.region = seeds[seed_id]['region']
			self.key = keys[key_id]
			self.api = RiotAPICaller(self.key, self.region)
			self.seed_ids = [seed_id]
			self.collect_recursively()
		elif command == '2':
			self.region = input("enter region: ")
			self.create_table_summoners()
			self.create_table_matchlists()
			self.create_table_matches()


	
	def collect_recursively(self):
		next_sid = self.seed_ids.pop()

		#look up his league
		league_position_dto = api.get_position(next_sid)
		league_id = soloqueue_league(league_position_dto)

		self.collect_league(league_id)
		#as a side effect, a queue is filled up with potential next seed candidates
		self.collect_recursively()
	'''
	parameter: league id (int)
	collect season 9 match info for everyone in the league.
	fill up queue with next seed summoner ids
	'''
	def collect_league(league_id):
		status, league_list_dto = api.get_league(league_id)
		if status != 200:
			return

		#save everyone in the league in a queue
		sid_queue = [e['playerOrTeamId'] for e in league_list_dto['entries']]

		#for every person look up his aid
		while len(sid_queue) > 0:
			sid = sid_queue.pop()

			success, summoner_dto = self.get_summoner(sid, save=True)
			if success == False:
				continue
			aid = summoner_dto['accountId']
			success, matchlist_dto = self.get_matchlist(aid, season=9, save=True) #season7
			if success == False:
				continue
		
			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				match_dto = self.get_match(game_id, save=True)
				if len(sid_queue) < 20:
					for p_id_dto in match_dto['participantIdentities']:
						sid.queue.append(p_id_dto['player']['summonerId'])

	'''
	get from db, else get from api.
	if save=True, then save in db
	parameters: summoner id(int), save(bool)
	return success(bool), data(dict)
	'''
	def get_summoner(sid, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM summoners WHERE sid = ?", (sid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, {'accountId': rows[0][0], 'id': rows[0][1]} 
		else:
			status, summoner_dto = api.get_summoner_by_summoner_id(sid)
			if status != 200:
				return False, summoner_dto
			if save == True:
				self.record_summoner(summoner_dto)
			return True, summoner_dto
	'''	
	get from db, else get from api.
	if save=True, then save in db
	if season=0, collect all seasons.
	parameters: summoner id(int), save(bool)
	return success(bool), data(dict)
	'''
	def get_matchlist(aid, season=0, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM matchlists WHERE aid = ?", (aid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, json.loads(rows[0][1])
		else:
			status, matchlist_dto = api.get_matchlist(aid, season_id=season)
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
	def get_match(game_id, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM matches WHERE gameId = ?", (game_id,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return True, json.loads(rows[0][4])
		else:
			status, match_dto = api.get_match(aid, game_id)
			if status != 200:
				return False, match_dto
			if save == True:
				self.record_matchlist(aid, matchlist_dto)
			return True, matchlist_dto

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
	'''
	def record_matchlist(self, aid, matchlist_dto):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("INSERT OR IGNORE INTO summoners VALUES (?,?)",
					(aid, json.dumps(matchlist_dto),))
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
		pass

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

	def create_table(self, query):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute(query)
		connection.commit()

d = DataManager()