from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import sqlite3
import time

class Scraper():
    def __init__(self, region):
        self.region = region
        self.driver = webdriver.Chrome('C:/Users/Megas/Downloads/chromedriver_win32/chromedriver.exe')
        self.driver.implicitly_wait(8)
    def get_user(self, username):
        driver = self.driver
        if self.region == 'KR':
            driver.get('http://www.op.gg/summoner/userName={}'.format(username))
        elif self.region == 'NA1':
            driver.get('http://na.op.gg/summoner/userName={}'.format(username))
        elif self.region == 'EUW':
            driver.get('http://euw.op.gg/summoner/userName={}'.format(username))
        wait = WebDriverWait(driver, 5)


        data = {}

        past_ranks = []
        try:
            past_rank = driver.find_element_by_css_selector("body > div.l-wrap.l-wrap--summoner > div.l-container > div > div > div.Header > div.PastRank > ul")
            past_ranks = past_rank.find_elements_by_css_selector("li")
        except:
            print("past rank doens't exist")
            pass

        past = []
        for i in range(len(past_ranks)):
            entry = {}
            pair = past_ranks[i].text.strip().split()
            entry['season'] = self.parse_season(pair[0])
            entry['tier'] = self.parse_tier_text(pair[1])
            past.append(entry)
        data['past'] = past
        print(past)


        try:
            #tiergraph = driver.find_element_by_xpath("//button[@class='Button SemiRound White']")
            tiergraph = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "body > div.l-wrap.l-wrap--summoner > div.l-container > div > div > div.Header > div.Profile > div.Buttons > button.Button.SemiRound.White")))
            #tiergraph = driver.find_element_by_css_selector("body > div.l-wrap.l-wrap--summoner > div.l-container > div > div > div.Header > div.Profile > div.Buttons > button.Button.SemiRound.White")
            
            tiergraph.click()
        except:
            print("tiergraph exception")
            return False, {}

        try:
            monthly = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-type='month']")))
            #monthly = driver.find_element_by_xpath("//a[@data-type='month']")
            monthly.click()
        except:
            print("monthly graph exception")
            return False, {}
        time.sleep(1.0)
        try:
            chart = driver.find_element_by_css_selector("#highcharts-12 > div")
            points = chart.find_elements_by_xpath("//div[@style='text-align:center;']")
        except:
            print("chart exception")
            return False, {}

        try:
            month_labels = driver.find_element_by_css_selector("#highcharts-12 > svg > g.highcharts-axis-labels.highcharts-xaxis-labels")
            months = month_labels.find_elements_by_css_selector("text")
        except:
            print("month exception")
            return False, {}



        recent = []
        for i in range(len(points)):
            entry = {}
            pair = points[i].text.split()
            entry['tier'] = self.parse_tier(pair[0])
            entry['rank'] = self.parse_rank(pair[0])
            entry['point'] = int(pair[1].strip("LP"))
            month_text = months[i].text
            if month_text == "이번달":
                month_text = 2018.03
            entry['month'] = month_text
            recent.append(entry)
        print (recent)
        data['recent'] = recent



        return True, data

    def parse_tier(self, text):
        value = 0
        code = ['B', 'S', 'G', 'P', 'D', 'M', 'C']
        value = code.index(text[0]) + 1
        return value
    def parse_rank(self, text):
        value = 0
        return 6 - int(text[1])
    def parse_season(self, text):
        value = 0
        return text[1]
    def parse_tier_text(self, text):
        value = 0
        code = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Challenger"]
        value = code.index(text) + 1
        return value


