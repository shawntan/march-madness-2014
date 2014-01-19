import theano
import pandas        as pd
import numpy         as np
import theano.tensor as T
from itertools import izip

def cost(matches,weights):
	wteam_tensor = weights[matches[:,0]]
	wteam_score  =         matches[:,1]
	lteam_tensor = weights[matches[:,2]]
	lteam_score  =         matches[:,3]
	cost = T.mean(
			( (wteam_tensor[:,0]*lteam_tensor[:,1]).sum(1) - wteam_score )**2 +\
			( (wteam_tensor[:,1]*lteam_tensor[:,0]).sum(1) - lteam_score )**2
		)
	return cost


def load_data():
	df = pd.read_csv(
		'regular_season_results.csv',
		usecols = ['wteam','wscore','lteam','lscore']
	)
	team_ids = df['wteam'].append(df['lteam']).unique()
	mapping  = dict(izip(team_ids,range(team_ids.shape[0])))
	df['wteam'] = [ mapping[i] for i in df['wteam'] ]
	df['lteam'] = [ mapping[i] for i in df['lteam'] ]
	return df,mapping

if __name__ == '__main__':
	df,mapping = load_data()
	data = theano.shared(np.asarray(df.values,dtype=np.int16))
	W    = theano.shared(np.random.random([len(mapping),2,1]))

	matches = T.wmatrix('matches')
	weights = T.dtensor3('weights')

	cost = cost(matches,weights)
	grad = T.grad(cost,wrt=weights)
	train = theano.function(
			inputs = [],
			outputs = cost,
			givens  = [
				(matches, data),
				(weights, W)
			],
			updates = [
				(W, W - grad)
			]
		)
	for _ in xrange(1000):
		print train()

