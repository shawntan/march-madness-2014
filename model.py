import theano
import theano.tensor as T
import numpy as np
def cost_fn(matches,weights):
	wteam_tensor = weights[matches[:,0]]
	lteam_tensor = weights[matches[:,2]]
	wteam_score  = matches[:,1]
	lteam_score  = matches[:,3]

	wteam_predict = T.nnet.softplus((wteam_tensor[:,0]*lteam_tensor[:,1]).sum(1))
	lteam_predict = T.nnet.softplus((wteam_tensor[:,1]*lteam_tensor[:,0]).sum(1))
	accuracy      = T.mean(wteam_predict > lteam_predict)
	cost = T.mean(
			( wteam_predict - wteam_score )**2 +\
			( lteam_predict - lteam_score )**2
		)
	return cost, accuracy

def trainer_tester(mapping,train_data,test_data):
	data = theano.shared(train_data)
	test_data = theano.shared(test_data)
	init_weights = 0.1*np.random.randn(len(mapping),2,100)
	W = theano.shared(init_weights)

	matches = T.wmatrix('matches')
	weights = T.dtensor3('weights')
	t_matches = T.wmatrix('t_matches')
	delta   = theano.shared(np.zeros(init_weights.shape))

	cost, accuracy = cost_fn(matches,weights)
	log_loss_fn = log_loss(t_matches,weights)
	grad = T.grad(cost,wrt=weights)
	train = theano.function(
			inputs = [],
			outputs = cost,
			givens  = { matches: data, weights: W },
			updates = [
				(W, W - 0.1*( grad + 0.5 * delta )),
				(delta, 0.1*( grad + 0.5 * delta ))
			]
		)
	test = theano.function(
			inputs = [],
			outputs = [log_loss_fn],
			givens  = { t_matches: test_data, weights: W }
		)
	return train,test,W

def log_loss(tourney_results,weights):
	team1_tensor = weights[tourney_results[:,0]]
	team2_tensor = weights[tourney_results[:,1]]
	outcome = tourney_results[:2]
	team1_predict = T.nnet.softplus((team1_tensor[:,0]*team2_tensor[:,1]).sum(1))
	team2_predict = T.nnet.softplus((team1_tensor[:,1]*team2_tensor[:,0]).sum(1))
	outcome_predict = team1_predict > team2_predict
	logloss = -T.mean(
			outcome_predict
#			outcome*T.log(outcome_predict) +\
#			(1-outcome)*T.log(1-outcome_predict)
		)
	return logloss


