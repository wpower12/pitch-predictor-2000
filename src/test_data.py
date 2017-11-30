from data.collect_data import *
import pickle, sys, mlbgame, pymysql

conn_ldb = pymysql.connect(host=HOST,
								user=USER,
								password=PASS,
								db=DB_LAHMAN,
								charset='utf8mb4',
								cursorclass=pymysql.cursors.DictCursor)

month = mlbgame.games(2016, 4)
month = mlbgame.combine_games(month)
g = month[0]

s = get_pitch_seqs_from_game(conn_ldb, g.game_id)

conn_ldb.close()

print(s[0])
