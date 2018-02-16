from selenium import webdriver
from bs4 import BeautifulSoup
import sqlite3


class Scraper():
    def __init__(self):
        self.driver = webdriver.Chrome('C:/Users/Megas/Downloads/chromedriver_win32/chromedriver.exe')
        self.driver.implicitly_wait(3)

    def get_user(self, username):
        driver = self.driver
        driver.get('http://www.op.gg/summoner/userName={}'.format(username))

        tiergraph = driver.find_element_by_xpath("//button[@class='Button SemiRound White']")
        tiergraph.click()
        monthly = driver.find_element_by_xpath("//a[@data-type='month']")
        monthly.click()

        chart = driver.find_element_by_xpath("//*[@id='highcharts-12']/div")

        ranks = chart.find_elements_by_css_selector("b")
        points = chart.find_elements_by_xpath("//div[@style='text-align:center;']")
        for rank in ranks:
            print(rank.text)
        for point in points:
            print(point.text)

def fetch_all_user_history():
    connection = sqlite3.connect('loldata2.db')
    cur = connection.cursor()
    cur.execute("SELECT matchlist.aid, users.tier, matchlist.matchlist FROM matchlist INNER JOIN users ON matchlist.aid = users.aid")
    rows = cur.fetchall()
    histories = [{'aid':row[0], 'tier':row[1], 'matchlist':json.loads(row[2])} for row in rows]
    return histories

scraper = Scraper()
scraper.get_user('이것도쓰이냐')
