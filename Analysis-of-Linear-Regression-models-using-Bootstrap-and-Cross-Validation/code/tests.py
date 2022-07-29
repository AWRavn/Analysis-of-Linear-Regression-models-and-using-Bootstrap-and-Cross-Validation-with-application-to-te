import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
from plot import *
from data_prep import *
from regression import *
from mlxtend.evaluate import bias_variance_decomp

RAND=13
np.random.seed(seed=RAND)

def main():
	"""Get plots"""
	"""
	# Franke
	# Define parameters
	N_elements = 20
	N_sizes = [10, 20, 30]
	sigma=0.05
	max_degree = 25
	degrees = [5, 10, 15, 25]
	lambda_= 0
	lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
	lambdas_lasso = [0.00001, 0.0001, 0.001, 0.01]
	test_size=0.2
	model = 'lasso'
	folds = [2, 3, 4, 5, 6, 7, 8, 9, 10]


	# Get dataset
	z, x, y = prepare_data(N=N_elements, sigma=sigma)

	# Plots
	plot_confidence_intervals("./figs/confidence_intervals.png", z, x, y, max_degree, N_elements)
	#print_stats(z, x, y, N_elements, 'ols')

	
	#ols	
	# bias-variance
	plot_MSE_train_test("./figs/mse_train_test", z, x, y, max_degree, model, lambda_)
	for n in N_sizes:
		z, x, y = prepare_data(N=n, sigma=sigma)
		for md in degrees:
			plot_boot("./figs/ols_tradeoff_boot_deg_{}_N{}".format(md, n), z, x, y, md, model, n_boots=50, type='tradeoff')
	
	# fold estiamtion
	for f in folds:
		plot_cv("./figs/ols_cv_folds_{}".format(f), z, x, y, max_degree, model, k=f)

	# mse
	for n in N_sizes:
		z, x, y = prepare_data(N=n, sigma=sigma)
		plot_boot("./figs/ols__boot_N{}".format(n), z, x, y, max_degree, model, n_boots=50, type='mse')
		plot_cv("./figs/ols_cv_N_{}".format(n), z, x, y, max_degree, model, k=5)
			
	

	#ridge
	# bias-variance
	for lamb in lambdas:
		plot_boot("./figs/ridge_tradeoff_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='tradeoff')
	

	# mse
	for lamb in lambdas:
		plot_boot("./figs/ridge_mse_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='mse')
		plot_cv("./figs/ridge_mse_cv_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, k=5)
	

	# lasso
	# bias-variance
	for lamb in lambdas_lasso:
		plot_boot("./figs/lasso_tradeoff_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='tradeoff')
	
	# mse
	for lamb in lambdas_lasso:
		plot_boot("./figs/lasso_mse_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='mse')
		plot_cv("./figs/lasso_mse_cv_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, k=5)

	"""
	
	# Treeain data

	# Define parameters
	N_elements = 20
	max_degree = 30
	lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
	lambdas_lasso = [0.001, 0.01, 1, 10]
	test_size=0.2
	model = 'ols'

	# Get dataset
	filename = "SRTM_data_Norway_1.tif"
	z, x, y = prepare_terrain_data(N_elements, filename)
	
	# Plots
	plot_confidence_intervals("./figs/terrain_confidence_intervals.png", z, x, y, max_degree, N_elements)
	#print_stats(z, x, y, N_elements, 'ols')

	#ols
	# bias-variance
	# plot_MSE_train_test("./figs/terrain_mse_train_test.png", z, x, y, max_degree, model, lambda_=0)
	plot_boot("./figs/terrain_ols_tradeoff_boot.png", z, x, y, max_degree, model, n_boots=50, type='tradeoff')
	# mse
	plot_boot("./figs/terrain_ols_boot_mse.png", z, x, y, max_degree, model, n_boots=50, type='mse')
	plot_cv("./figs/terrain_ols_cv_mse.png", z, x, y, max_degree, model, k=5)
	
	#ridge
	model=('ridge')
	# bias-variance
	for lamb in lambdas:
		plot_boot("./figs/terrain_ridge_tradeoff_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='tradeoff')
	# mse
	for lamb in lambdas:
		plot_boot("./figs/terrain_ridge_mse_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='mse')
		plot_cv("./figs/terrain_ridge_mse_cv_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, k=5)

	# lasso
	model=('lasso')
	# bias-variance
	for lamb in lambdas_lasso:
		plot_boot("./figs/terrain_lasso_tradeoff_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='tradeoff')
	# mse
	for lamb in lambdas_lasso:
		plot_boot("./figs/terrain_lasso_mse_boot_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, n_boots=50, type='mse')
		plot_cv("./figs/terrain_lasso_mse_cv_L_{}.png".format(lamb), z, x, y, max_degree, model, lambda_=lamb, k=5)



if __name__ == "__main__":
	main()
