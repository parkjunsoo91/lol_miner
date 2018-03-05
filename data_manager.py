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

	def collect_league(league_id):
		league_list_dto = api.get_league(league_id)

		#save everyone in the league in a queue
		sid_queue = [e['playerOrTeamId'] for e in league_list_dto['entries']]

		#for every person look up his aid
		while len(sid_queue) > 0:
			sid = sid_queue.pop()

			summoner_dto = self.get_summoner(sid, save=True)

			aid = summoner_dto['accountId']
			matchlist_dto = api.get_matchlist(aid, season=9, save=True) #season7
			self.record_matchlist(aid, matchlist_dto['matches'])

			for match_reference_dto in matchlist_dto['matches']:
				game_id = match_reference_dto['gameId']
				match_dto = self.record_match(game_id)
				if len(sid_queue) < 20:
					sid.queue += participant_sid(match_dto)

	'''get from db, else get from api, then save in db'''
	def get_summoner(sid, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM summoners WHERE sid = ?", (sid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return {'accountId': rows[0][0], 'id': rows[0][1]} 
		else:
			summoner_dto = api.get_summoner_by_summoner_id(sid)
			if save == True:
				self.record_summoner(summoner_dto)
			return summoner_dto

	def get_matchlist(aid, season=0, save=False):
		connection = sqlite3.connect(self.region + '.db')
		cur = connection.cursor()
		cur.execute("SELECT * FROM matchlists WHERE aid = ?", (aid,))
		rows = cur.fetchall()
		if len(rows) != 0:
			return json.loads(rows[0][1])
		else:
			matchlist_dto = api.get_matchlist(aid, season_id=season)
			if save == True:
				self.record_summoner(summoner_dto)
			return summoner_dto


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

	def record_matchlist(self, listof_match_reference_dto):
		pass

	def record_match(self, match_id):
		#check if this exists, if not call api
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