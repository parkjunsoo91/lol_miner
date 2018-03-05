import json
import sqlite3
from riot_api import *

seeds = {'1': {'sid':3836801, 'region':'KR'}}
keys = {'1': "RGAPI-9b504ad4-41c7-4224-aa9e-fa16ba5028b7"}
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
			self.seed_ids = [seeds[seed_id]['sid']]
			self.collect_recursively()
		elif command == '2':
			self.region = input("enter region: ")
			self.create_table_summoners()
			self.create_table_matchlists()
			self.create_table_matches()


	
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
			success, summoner_dto = self.get_summoner(sid, save=True)
			if success == False:
				continue
			self.update_tier(summoner_dto["accountId"], tier, rank)

			aid = summoner_dto['accountId']
			success, matchlist_dto = self.get_matchlist(aid, season=9, save=True) #season7
			if success == False:
				continue
			print(matchlist_dto)
			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				success, match_dto = self.get_match(game_id, save=True)
				if success == False:
					continue
				if len(self.seed_ids) < 20:
					for p_id_dto in match_dto['participantIdentities']:
						self.seed_ids.append(p_id_dto['player']['summonerId'])

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
		if tier == "BRONZE":
			tier_val = 1
		if tier == "SILVER":
			tier_val = 2
		if tier == "GOLD":
			tier_val = 3
		if tier == "PLATINUM":
			tier_val = 4
		if tier == "DIAMOND":
			tier_val = 5
		if tier == "MASTER":
			tier_val = 6
		if tier == "CHALLENGER":
			tier_val = 7
		rank_val = 0
		if rank == "I":
			rank_val = 1
		if rank == "II":
			rank_val = 2
		if rank == "III":
			rank_val = 3
		if rank == "IV":
			rank_val = 4
		if rank == "V":
			rank_val = 5
		if rank == "VI":
			rank_val = 6
		if rank == "VII":
			rank_val = 7
			
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