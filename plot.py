import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

from data_prep import *
from regression import *

RAND=13
np.random.seed(seed=RAND)

def print_stats(z, x, y, degree, model, lambda_=0):

    # Get model
    lr = LinearRegression(z, x, y, degree=degree, model=model, train_test_split=False)

    # Get stats
    mse = lr.MSE(lr.z, lr.z_pred)
    r2 = lr.R2(lr.z, lr.z_pred)

    print("MSE: {}".format(mse))
    print("R2: {}".format(r2))

    return mse, r2

def plot_confidence_intervals(title, z, x, y, degree, N=10, model='ols'):

    # Get model
    lr = LinearRegression(z, x, y, degree=degree, model=model, train_test_split=False)

    # Get confidence intervals
    beta_vars = lr.beta_variances(lr.X)
    beta_confs = 1.96*np.sqrt(beta_vars/N**2)

    # Plot the graph
    plt.figure()

    axis = list(range(0, len(beta_vars)))
    plt.errorbar(axis, lr.beta, yerr=beta_confs, markersize=4, linewidth=1, ecolor="red", fmt='o', capsize=5, capthick=1)

    # Customize
    plt.xlabel('Betas')
    x_labels = [r"$\beta_"+"{{{:}}}$".format(i) for i in range(len(lr.beta))]
    plt.xticks(axis, x_labels)
    plt.title('Bata values')

    # Save output
    plt.savefig(str(title))

    plt.close()

    return beta_confs


def plot_boot(title, z, x, y, degree, model, lambda_=0, n_boots=40, type='tradeoff'):

    # Initialize arrays
    mse_train = np.zeros(degree)
    mse_test = np.zeros(degree)
    bias = np.zeros(degree)
    var = np.zeros(degree)
    degrees = range(1, degree+1)

    for d in degrees:
        # Get model
        lr = LinearRegression(z, x, y, degree=d, model=model, lambda_=lambda_, train_test_split=True)
        mse_test[d-1], mse_train[d-1], bias[d-1], var[d-1] = lr.bootstrap(X_train=lr.X_train, z_train=lr.z_train, X_test=lr.X_test, z_test=lr.z_test, model=model, lambda_=lambda_, n_boots=n_boots)

        # Plot the graph
    if type=='tradeoff':
        plt.figure()
        plt.plot(degrees, mse_test, label='MSE')
        plt.plot(degrees, bias, "--", label='bias')
        plt.plot(degrees, var, "--", label='variance')

        # Customize
        plt.xlabel('Polynomal model complexity [degrees]')
        plt.ylabel('Mean Squared Error')
        plt.title("Bias-variance tradeoff (bootstrap)")
        plt.legend()
        plt.grid()

    elif type=='mse':
        plt.figure()
        plt.plot(degrees, mse_train, label='MSE_train')
        plt.plot(degrees, mse_test, label='MSE_test')

        # Customize
        plt.xlabel('Polynomal model complexity [degrees]')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title("Train and test error (Bootstrap)")
        plt.legend()
        plt.grid()

    # Save output
    plt.savefig(str(title))

    plt.close()


def plot_MSE_train_test(title, z, x, y, degree, model, lambda_=0):

    # Initialize arrays
    mse_train = np.zeros(degree)
    mse_test = np.zeros(degree)
    degrees = range(1, degree+1)

    for d in degrees:
        # Get model
        lr = LinearRegression(z, x, y, degree=d, model=model, lambda_=lambda_, train_test_split=True)
        mse_train[d-1] = lr.MSE(lr.z_train, lr.z_pred_train)
        mse_test[d-1] = lr.MSE(lr.z_test, lr.z_pred_test)


    # Plot the graph
    plt.figure()
    plt.plot(degrees, mse_train, label='MSE_train')
    plt.plot(degrees, mse_test, label='MSE_test')

    # Customize
    plt.xlabel('Polynomal model complexity [degrees]')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Train and test error comparison')
    plt.legend()
    plt.grid()

    # Save output
    plt.savefig(str(title))

    plt.close()


def plot_cv(title, z, x, y, degree, model, lambda_=0, k=5):

    # Initialize arrays
    mse_train = np.zeros(degree)
    mse_test = np.zeros(degree)
    degrees = range(1, degree+1)

    for d in degrees:
        # Get model
        lr = LinearRegression(z, x, y, degree=d, model=model, lambda_=lambda_, train_test_split=True)
        X = lr.design_matrix(x, y, degree=d)

        mse_test[d-1], mse_train[d-1] = lr.cv(X, z, model, k=k)

    # Plot the graph
    plt.figure()
    plt.plot(degrees, mse_train, label='MSE_train')
    plt.plot(degrees, mse_test, label='MSE_test')

    # Customize
    plt.xlabel('Polynomal model complexity [degrees]')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title("Train and test error (CV)")
    plt.legend()
    plt.grid()

    # Save output
    plt.savefig(str(title))

    plt.close()