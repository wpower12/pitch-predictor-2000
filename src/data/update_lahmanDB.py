#!/usr/bin/python
import sys, mlbgame, pymysql

# Updates the Lahman database to include the mlbgameID.  Useful when 
# we need to access the handedness of the players during sequence collection.
# right now we just match on the first and last name of the players. This 
# might make some mistakes, but w.e. Should be good enough for a first pass. 

# This script is assuming there is a local MySQL DB running on localhost with 
# the following user/pw and database 
HOST = 'localhost'
USER = 'mlbuser'
PASS = 'password'
DB_LAHMAN = 'lahman2016'

def migrate_players_from_game(conn, game_id):
	try: # sometimes the game_id is invalid, and this fails.
		roster = mlbgame.player_stats(game_id)
		batters  = roster['home_batting']+roster['away_batting']
		pitchers = roster['home_pitching']+roster['away_pitching']

		for p in batters+pitchers:
			first, last = p.name_display_first_last.split(' ', 1)
			with conn.cursor() as cursor:
			    # Create a new record
			    sql = "UPDATE `Master` SET mlbgameID=%s WHERE nameFirst=%s AND nameLast=%s"
			    cursor.execute(sql, (p.id, first, last))
			conn.commit()
	except:
		raise

# Main - First arg is the year to update ################################
if __name__ == "__main__":
	YEAR_TO_MIGRATE = int(sys.argv[1])
	conn_ldb = pymysql.connect(host=HOST,
								user=USER,
								password=PASS,
								db=DB_LAHMAN,
								charset='utf8mb4',
								cursorclass=pymysql.cursors.DictCursor)
	  
	year = mlbgame.games(YEAR_TO_MIGRATE)
	year = mlbgame.combine_games(year)
	n = len(year)
	i = 0
	print("processing games from {}:\n0/{}".format(YEAR_TO_MIGRATE, n))
	for g in year:
		i += 1
		try:
			migrate_players_from_game(conn_ldb, g.game_id)
			print('{}/{}'.format(i,n))
		except:
			print('{}/{} - error with game {}'.format(i,n, g.game_id))