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

	cur.execute("SELECT count(*) FROM users")
	print("total", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("BRONZE",))
	print("bronze", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("SILVER",))
	print("silver", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("GOLD",))
	print("gold", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("PLATINUM",))
	print("platinum", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("DIAMOND",))
	print("diamond", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("MASTER",))
	print("master", cur.fetchone()[0])
	cur.execute("SELECT count(*) FROM users where tier=?", ("CHALLENGER",))
	print("challenger", cur.fetchone()[0])
	#cur.execute("SELECT count(*) FROM matches")
	#print("matches", cur.fetchone()[0])

check()