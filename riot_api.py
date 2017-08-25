import http.client
import json
import time
import datetime


class RiotAPICaller:
	def __init__(self, api_key):
		self.KEY = api_key
		self.URL = "kr.api.riotgames.com"

	def get_summoner_by_account_id(self, account_id):
		request_body = "/lol/summoner/v3/summoners/by-account/{}".format(account_id)
		return self.send_request(request_body)
		
	def get_summoner_by_summoner_id(self, summoner_id):
		request_body = "/lol/summoner/v3/summoners/{}".format(summoner_id)
		return self.send_request(request_body)

	def get_league(self, summoner_id):
		request_body = "/lol/league/v3/leagues/by-summoner/{}".format(summoner_id)
		return self.send_request(request_body)

	def get_matchlist(self, account_id, season_id = None):
		assert (season_id == None) or (0 <= season_id and season_id <= 9)
		request_body = "/lol/match/v3/matchlists/by-account/{}".format(account_id)
		if season_id != None:
			request_body = "{}?season={}".format(request_body, season_id)
		return self.send_request(request_body)

	def get_match(self, game_id):
		request_body = "/lol/match/v3/matches/{}".format(game_id)
		return self.send_request(request_body)
		
	def send_request(self, request_body):
		time.sleep(1.2)
		print(request_body)
		for i in range (5):
			connection = http.client.HTTPSConnection(self.URL, timeout=10)
			response = None
			try:
				connection.request("GET", request_body, headers={'X-Riot-Token': self.KEY, })
				response = connection.getresponse()
				print(datetime.datetime.now(), response.status, response.reason)
				if response.status == 200:
					b = response.read()
					dataObject = json.loads(b)
					connection.close()
					return dataObject
			except http.client.HTTPException as e:
				print(e)
				if response != None:
					print(response.status, response.reason)
				connection.close()
			time.sleep(1.2)

			print("retrying...")
		return response.status
