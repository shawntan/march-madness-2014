import pandas as pd
import numpy  as np
sel_cols = ['wteam','wscore','lteam','lscore']
def generate_team_ids(teamfile):
	df = pd.read_csv(teamfile,dtype={i:np.int16 for i in sel_cols})
	team_ids = df['id']
	mapping  = { index:team_id for team_id,index in enumerate(team_ids) }
	return mapping,team_ids.values

def load_seasons_data(mapping,filename,seasons):
	df = pd.read_csv(
			filename,
			usecols=['season']+sel_cols,
			dtype={i:np.int16 for i in sel_cols})
	df = df[df.season.isin(list(seasons))][sel_cols]
	df.loc[:,'wteam'] = df['wteam'].apply(mapping.get)
	df.loc[:,'lteam'] = df['lteam'].apply(mapping.get)
	return np.asarray(df.values,dtype=np.int16)

def load_tourney_pairs(mapping,filename,season):
	df = pd.read_csv(filename,usecols=['season','team'])
	teams = df[df.season == season].team
	return [ (mapping[i],mapping[j])
				for i in teams
				for j in teams if i < j ]

def load_tourney_results(mapping,filename,seasons):
	df = pd.read_csv(
			filename,
			usecols=['season']+sel_cols,
			dtype={i:np.int16 for i in sel_cols})
	df = df[df.season.isin(list(seasons))][sel_cols]
	df.loc[:,'team1'] = df.apply(lambda row: min(row['wteam'],row['lteam']),axis=1)
	df.loc[:,'team2'] = df.apply(lambda row: max(row['wteam'],row['lteam']),axis=1)
	df.loc[df.wteam == df.team1,'result'] = 1 
	df.loc[df.wteam != df.team1,'result'] = 0
	df.loc[:,'team1'] = df.team1.apply(mapping.get)
	df.loc[:,'team2'] = df.team2.apply(mapping.get)
	
	return np.asarray(df[['team1','team2','result']].values,dtype=np.int16)


if __name__ == "__main__":
	mapping,_ = generate_team_ids('data/teams.csv')
	matchings = load_tourney_results(mapping,'data/tourney_results.csv','R')
	print matchings
