import http.client
import json
import time
import datetime
import sys

class RiotAPICaller:
	def __init__(self, api_key, region = "KR"):
		self.KEY = api_key
		if region == "NA1":
			self.URL = "na1.api.riotgames.com"
		if region == "EUW":
			self.URL = "euw1.api.riotgames.com"
		if region == "KR":
			self.URL = "kr.api.riotgames.com"
		
		self.counter = 0

	def get_summoner_by_account_id(self, account_id):
		request_body = "/lol/summoner/v3/summoners/by-account/{}".format(account_id)
		return self.send_request(request_body)
		
	def get_summoner_by_summoner_id(self, summoner_id):
		request_body = "/lol/summoner/v3/summoners/{}".format(summoner_id)
		return self.send_request(request_body)

	def get_position(self, summoner_id):
		request_body = "/lol/league/v3/positions/by-summoner/{}".format(summoner_id)
		return self.send_request(request_body)

	def get_league(self, league_id):
		request_body = "/lol/league/v3/leagues/{}".format(league_id)
		return self.send_request(request_body)

	def get_matchlist(self, account_id, season_id = None):
		assert (season_id == None) or (0 <= season_id and season_id <= 11)
		request_body = "/lol/match/v3/matchlists/by-account/{}".format(account_id)
		#if season_id != None:
		#	request_body = "{}?season={}".format(request_body, season_id)
		status, matchlist_dto = self.send_request(request_body)
		if status != 200:
			return status, {}
		matches = matchlist_dto['matches']
		total_games = matchlist_dto['totalGames']
		start_index = matchlist_dto['startIndex']
		end_index = matchlist_dto['endIndex']
		print ("len(matches) = {}, total_games = {}".format(len(matches), total_games))
		while end_index < total_games:
			next_request_body = "{}?beginIndex={}".format(request_body, end_index)
			status, matchlist_dto = self.send_request(next_request_body)
			if status != 200:
				return status, {}
			matches += matchlist_dto['matches']
			total_games = matchlist_dto['totalGames']
			end_index = matchlist_dto['endIndex']
			print ("len(matches) = {}, total_games = {}".format(len(matches), total_games))
		#if season_id == None:
		assert len(matches) == total_games
		return status, {'totalGames': total_games, 'matches': matches}

	def get_match(self, game_id):
		request_body = "/lol/match/v3/matches/{}".format(game_id)
		return self.send_request(request_body)

	def get_champions(self):
		request_body = "/lol/static-data/v3/champions?locale=en_US&tags=all&dataById=true"
		return self.send_request(request_body)

	def get_items(self):
		request_body = "/lol/static-data/v3/items?locale=en_US&tags=all"
		return self.send_request(request_body)

	'''
	send request and receive response. Repeat until getting correct response.
	argument: request body
	returns: response status, response bodymd
	'''
	def send_request(self, request_body):
		time.sleep(1.2)
		print(request_body)
		for i in range (3):
			connection = http.client.HTTPSConnection(self.URL, timeout=10)
			response = None
			try:
				connection.request("GET", request_body, headers={'X-Riot-Token': self.KEY, })
				response = connection.getresponse()
				self.counter += 1
				print(datetime.datetime.now(), self.counter, response.status, response.reason)
				if response.status == 200:
					b = response.read()
					dataObject = json.loads(b)
					connection.close()
					return response.status, dataObject
				elif response.status == 404:
					return response.status, {}
			except http.client.HTTPException as e:
				#does this part ever run?
				print(e)
				if response != None:
					print(response.status, response.reason)
				connection.close()
			except:
				print("Unexpected error:", sys.exc_info()[0])
				connection.close()
				time.sleep(3.0)
			time.sleep(1.2)

			print("retrying...")
		return response.status, {}
