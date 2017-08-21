import sqlite3




def check_silver():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM users where tier=?", ("SILVER",))
	rows = cur.fetchall()
	for row in rows:
		print(row)



def check_twitch_bans():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	#cur.execute("SELECT * FROM users where tier=?", ("SILVER",))
	cur.execute("SELECT * FROM matches where ban1 = ?", (29,))
	rows = cur.fetchall()
	for row in rows:
		print(row[:38])

def check_usercount():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM users")
	rows = cur.fetchall()
	print(len(rows))

def check_matchcount():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()
	cur.execute("SELECT * FROM matches")
	rows = cur.fetchall()
	print(len(rows))

def check():
	connection = sqlite3.connect('loldata2.db')
	cur = connection.cursor()

	cur.execute("SELECT * FROM users")
	rows = cur.fetchall()
	print("total users:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("BRONZE",))
	rows = cur.fetchall()
	print("bronze:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("SILVER",))
	rows = cur.fetchall()
	print("silver:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("GOLD",))
	rows = cur.fetchall()
	print("gold:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("PLATINUM",))
	rows = cur.fetchall()
	print("platinum:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("DIAMOND",))
	rows = cur.fetchall()
	print("diamond:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("MASTER",))
	rows = cur.fetchall()
	print("master:",len(rows))
	cur.execute("SELECT * FROM users where tier=?", ("CHALLENGER",))
	rows = cur.fetchall()
	print("challenger:",len(rows))
	cur.execute("SELECT * FROM matches")
	rows = cur.fetchall()
	print("total matches:",len(rows))

check()