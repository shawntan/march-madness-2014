import model
import data
import numpy as np
if __name__ == '__main__':
	mapping,_ = data.generate_team_ids('data/teams.csv')
	training_data = data.load_seasons_data(mapping,'data/regular_season_results.csv','QR')
	test_data = data.load_tourney_results(mapping,'data/tourney_results.csv','R')
	print test_data
	train, test,shared_params = model.trainer_tester(mapping,training_data,test_data)
	patience = 100
	best_cost, best_acc, best_weights = np.inf, None, None
	while True:
		for _ in xrange(10):
			print "Taking step..." 
			train()
		c l,= test()
		if best_cost <= c:
			patience -= 1
			print patience
		else:
			best_cost,best_acc,best_weights = c,accuracy,shared_params.get_value()
			print c,accuracy
		if patience == 0: break
	print best_cost,best_acc
