import theano
import pandas        as pd
import numpy         as np
import theano.tensor as T
from itertools import izip

def cost(matches,weights):
	wteam_tensor = weights[matches[:,0]]
	lteam_tensor = weights[matches[:,2]]
	wteam_score  = matches[:,1]
	lteam_score  = matches[:,3]
	"""
	wteam_predict = (wteam_tensor[:,0]*lteam_tensor[:,1]).sum(1)
	lteam_predict = (wteam_tensor[:,1]*lteam_tensor[:,0]).sum(1)
	"""
	wteam_predict = T.nnet.softplus((wteam_tensor[:,0]*lteam_tensor[:,1]).sum(1))
	lteam_predict = T.nnet.softplus((wteam_tensor[:,1]*lteam_tensor[:,0]).sum(1))
	accuracy      = T.mean(wteam_predict > lteam_predict)
	cost = T.mean(
			( wteam_predict - wteam_score )**2 +\
			( lteam_predict - lteam_score )**2
		)
	return cost, accuracy

def load_prep_csv(mapping,filename):
	df = pd.read_csv(
		filename,
		usecols = ['season','wteam','wscore','lteam','lscore']
	)
	team_ids = df['wteam'].append(df['lteam']).unique()
	for team_id in team_ids: 
		if team_id not in mapping:
			mapping[team_id] = len(mapping)
	df['wteam'] = [ mapping[i] for i in df['wteam'] ]
	df['lteam'] = [ mapping[i] for i in df['lteam'] ]
	return df


def load_data():
	mapping = {}
	df  = load_prep_csv(mapping,'regular_season_results.csv')
	dft = load_prep_csv(mapping,'tourney_results.csv')
	training = df[df.season.isin(list('PQ'))][['wteam','wscore','lteam','lscore']]
	training = training.append(dft[dft.season.isin(list('PQR'))][['wteam','wscore','lteam','lscore']])
	testing  = df[df.season == 'R' ][['wteam','wscore','lteam','lscore']]
	return mapping,training.values,testing.values

if __name__ == '__main__':
	mapping,train_data,test_data = load_data()

	test_data = np.asarray(test_data,dtype=np.int16)
	data = theano.shared(np.asarray(train_data,dtype=np.int16))
	init_weights = 0.1*np.random.randn(len(mapping),2,20)
	W    = theano.shared(init_weights)

	matches = T.wmatrix('matches')
	weights = T.dtensor3('weights')
	delta   = theano.shared(np.zeros(init_weights.shape))


	cost, accuracy = cost(matches,weights)
	grad = T.grad(cost,wrt=weights)
	train = theano.function(
			inputs = [],
			outputs = cost,
			givens  = { matches: data, weights: W },
			updates = [
				(W, W - 0.1*( grad + 0.5 * delta )),
				(delta, grad)
			]
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

