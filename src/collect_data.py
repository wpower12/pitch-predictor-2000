#!/usr/bin/python
import pickle, sys, mlbgame, pymysql

# saves a python object filled with pitch sequences from a year of games.
# assumes there is a MySQL database containing the LahmanDB running on localhost.

PITCH_DATA_DIR = "data/"
HOST = 'localhost'
USER = 'mlbuser'
PASS = 'password'
DB_LAHMAN = 'lahman2016'

def get_feature_vec(conn, batter, pitcher):
	ret = None
	b_bats, p_throws = None, None 
	try:
		with conn.cursor() as cursor:
			sql = "SELECT bats, throws from `Master` WHERE mlbgameID=%s"
			cursor.execute(sql, (batter.id))
			results = cursor.fetchone()
			b_bats = results['bats']
			sql = "SELECT bats, throws from `Master` WHERE mlbgameID=%s"
			cursor.execute(sql, (pitcher.id))
			results = cursor.fetchone()
			p_throws = results['throws']
		ret = [b_bats, p_throws, batter.pos]
	except:
		print("error building feature for {} to {}".format(pitcher.name, batter.name))
	return ret

def get_pitch_seqs_from_game(conn, game_id):
	atbats = []
	events  = mlbgame.game_events(game_id)
	# Need these to get pos :(
	roster = mlbgame.player_stats(game_id)
	batters  = roster['home_batting']+roster['away_batting']
	pitchers = roster['home_pitching']+roster['away_pitching']
	batters  = {p.id: p for p in batters}
	pitchers  = {p.id: p for p in pitchers}
	for i in events:
		inning = events[i]
		for ab in inning['top']+inning['bottom']:
			pitch_seq = []
			for pitch in ab.pitches:
				pitch_seq.append(pitch.pitch_type)
			feature_vec = get_feature_vec(conn, batters[ab.batter], pitchers[ab.pitcher])
			if feature_vec is not None:
				atbats.append([feature_vec, pitch_seq])
	return atbats

# Main - First arg is the year to grab sequences from ##############################
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
	sequences = []
	print("generating all sequences from games from {}:\ngame: 0/{}".format(YEAR_TO_MIGRATE, n))
	for g in year:
		i += 1
		try:
			s = get_pitch_seqs_from_game(conn_ldb, g.game_id)
			sequences += s
			print('{}/{} seq_len: {}'.format(i,n, len(sequences)))
		except:
			print('{}/{} - error with game {}'.format(i,n, g.game_id))

	fname = "{}/pitches_{}.p".format(PITCH_DATA_DIR, YEAR_TO_MIGRATE)
	pickle.dump( sequences, open( fname, "wb" ) )
conn_ldb.close()