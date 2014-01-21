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
	wteam_predict = T.nnet.softplus((wteam_tensor[:,0]*lteam_tensor[:,1]).sum(1))
	lteam_predict = T.nnet.softplus((wteam_tensor[:,1]*lteam_tensor[:,0]).sum(1))
	accuracy      = T.mean(wteam_predict > lteam_predict)
	cost = T.mean(
			( wteam_predict - wteam_score )**2 +\
			( lteam_predict - lteam_score )**2
		)
	return cost, accuracy


def load_data():
	df = pd.read_csv(
		'regular_season_results.csv',
		usecols = ['season','wteam','wscore','lteam','lscore']
	)
	team_ids = df['wteam'].append(df['lteam']).unique()
	mapping  = dict(izip(team_ids,range(team_ids.shape[0])))
	df['wteam'] = [ mapping[i] for i in df['wteam'] ]
	df['lteam'] = [ mapping[i] for i in df['lteam'] ]

	training = df[df.season.isin(list('PQ'))][['wteam','wscore','lteam','lscore']].values
	testing  = df[df.season == 'R' ][['wteam','wscore','lteam','lscore']].values
	return mapping,training,testing

if __name__ == '__main__':
	mapping,train_data,test_data = load_data()
	test_data = np.asarray(test_data,dtype=np.int16)
	data = theano.shared(np.asarray(train_data,dtype=np.int16))
	W    = theano.shared(np.random.random([len(mapping),2,5]))

	matches = T.wmatrix('matches')
	weights = T.dtensor3('weights')

	cost, accuracy = cost(matches,weights)
	grad = T.grad(cost,wrt=weights)
	train = theano.function(
			inputs = [],
			outputs = cost,
			givens  = { matches: data, weights: W },
			updates = { W: W - grad }
		)
	test = theano.function(
			inputs = [],
			outputs = accuracy,
			givens  = { matches: test_data, weights: W }
		)
	for _ in xrange(1000):
		for _ in xrange(10):
			print "Taking step..." 
			train()
		print test()

