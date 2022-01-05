import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

RAND=13
np.random.seed(seed=RAND)


class Regression:
    """
    A class to represent a regression instance.

    Attributes:
        z:  np.Array['x, y, n', float]
            Data points to apply regression on
        x:  np.Array['x, y, n', float]
            x-axis of the input data
        y:  np.Array['x, y, n', float]
            y-axis of the input data
        degree: float
            Target complexity degree
        model:  string
            Linear Regression method. Choose between {ols, ridge, lasso}
        lambda_:    float
            Regularization parameter lambda. Must be lambda > 0
        test_size: float
            Percentage of the data to use for testing. Default 0.2
        train_test_split: boolean
            True if data is to be split inro test and train subsets. Default True.

    Methods:
        design_matrix()
            Implements the Nd polynomial design matrix of a given degree based on a dataset. Includes intercept.
        MSE()
            Returns the mean squared error (MSE) of the prediction.
        R2()
            Returns the coefficient of determination (R2) of the prediction.
        beta_variances()
            Returns variances of the beta coefficients of the model.
        bootstrap()
            Performs teh bootstrap resampling using the model.
        cv()
            Performs the k-fold cross-validation resampling using the model.
    """

    def __init__(self, z, x, y, degree, model, lambda_=None, test_size=0.2, train_test_split=True):
        """Constructs the necessary attributes for the regression instance and applies it."""

        self.z = z
        self.x = x
        self.y = y
        self.degree = degree
        self.sklearn = False
        self.lambda_ = None or lambda_
        self.train_test_split = train_test_split


        regression_set = {'ols', 'ridge', 'lasso'}

        if model not in regression_set:
            raise NotImplementedError('Set "reg_method" to a valid keyword. Valid input keywords are {}'.format(regression_set))

        if model!='ols' and self.lambda_==0:
            raise ValueError('Lambda must be greater than zero for {} method. Else use "ols"'.format(model))


        self.X = self.design_matrix(self.x, self.y, self.degree)

        if self.train_test_split:
            self.X_train, self.X_test, self.z_train, self.z_test = self.__split_and_scale_data(self.X, self.z, test_size=0.2)

            self.beta = self.__apply_model(model, self.X_train, self.z_train)
            self.z_pred_train = self.__prediction(self.X_train, self.beta)            
            self.z_pred_test = self.__prediction(self.X_test, self.beta)
        else:
            self.beta = self.__apply_model(model, self.X, self.z)
            self.z_pred = self.__prediction(self.X, self.beta)


    def __apply_model(self, model, X, z):

        if model=='ols':
            if self.sklearn==False:
                beta = self.__ols(X, z)
            else:
                beta = self.__sklearn_ols(X, z)
        elif model=='ridge':
            if self.sklearn==False:
                beta = self.__ridge(X, z)
            else:
                self.beta = self.__sklearn_ridge(X, z)
        elif model=='lasso':
            beta = self.__sklearn_lasso(X, z)

        return beta


    def __ols(self, X, z):
        """Applies the Ordinary Least Squares regression model on data."""
        
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta


    def __sklearn_ols(self, X, z):
        """Applies the Ordinary Least Squares regression model on data using the sklearn package."""

        z = np.ravel(z)
        ols_reg = LinearRegression(fit_intercept=True).fit(X, z)
        beta = ols_reg.coef_.T
        return beta


    def __ridge(self, X, z):
        """Applies the Ridge regression model on data."""

        beta = np.linalg.pinv(X.T @ X + self.lambda_*np.identity(X.shape[1])) @ X.T @ z
        return beta


    def __sklearn_ridge(self, X, z):
        """Applies the Ridge regression model on data using the sklearn package."""

        ridge_reg = Ridge(fit_intercept=True, alpha=self.lambda_).fit(X,z)
        beta = ridge_reg.coef_
        return beta

       
    def __sklearn_lasso(self, X, z):
        """Applies the Lasso regression model on data using the sklearn package."""

        lasso_reg = Lasso(fit_intercept=True, max_iter=1000000, alpha=self.lambda_).fit(X,z)
        beta = lasso_reg.coef_
        return beta


    def __prediction(self, X, beta):
        """
        Returns the predicted regression values against design matrix X.

        Args:
            X np.Array['x, y, n', float]:       The design matrix X
            beta:                               The parameters beta 

        Returns:
            z_pred (np.Array['x, y', float]):   Predicted regression values on test set
        """

        return X @ beta


    def __split_and_scale_data(self, X, z, test_size=0.2):
        """Splits data and scales it by subtracting the mean"""

        # Train test split
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, shuffle=True, random_state=RAND)

        for i in range(len(X_train.T)):
            X_train[:,i] = X_train[:,i] - np.mean(X_train[:,i])
            X_test[:,i]  = X_test[:,i] - np.mean(X_test[:,i])

        z_train = z_train - np.mean(z_train)
        z_test = z_test - np.mean(z_test)

        return X_train, X_test, z_train, z_test


    def design_matrix(self, x, y, degree):
        """"
        Implements the Nd polynomial design matrix of a given degree based on a dataset. Includes intercept.

        Args:
            x (np.Array[float]):            x-data
            y (np.Array[float]):            y-data
            degree (int):                   N degree of the polynomial

        Returns:
            X (np.Array['z, n', float]):    Design matrix X
        """

        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        P = int((degree+1)*(degree+2)/2)
        X = np.ones((N,P))

        for i in range(1,degree+1):
            q = int((i)*(i+1)/2)
            for j in range(i+1):
                X[:,q+j] = x**(i-j) * y**j
 
        return X


    def MSE(self, z_actual, z_pred, sklearn=False):
        """Returns the mean squared error (MSE) of the prediction."""
        if sklearn==False:
            mse = (np.sum((z_actual - z_pred) ** 2)) / (len(z_pred))
            #mse = np.mean((z_actual - z_pred) ** 2)
        else:
            mse = mean_squared_error(z_actual, z_pred)
        return mse
   

    def R2(self, z_actual, z_pred, sklearn=False):
        """Returns the coefficient of determination (R2) of the prediction."""
        if sklearn==False:
            r2 = 1 - ((np.sum((z_actual - z_pred) ** 2)) / (np.sum((z_actual - np.mean(z_actual)) ** 2)))
        else:
            r2 = r2_score(z_actual, z_pred)
        return r2


    def beta_variances(self, X, sigma=1):
        """Returns variances of the beta coefficients of the model."""
        beta_vars = np.diagonal(sigma * (np.linalg.pinv(X.T @ X)))
        return beta_vars


    def bootstrap(self, X_train, z_train, X_test, z_test, model, lambda_=0, n_boots=50, sklearn=False):
        """Performs teh bootstrap resampling using the model."""

        # Initialize arrays
        z_pred_test = np.empty((len(z_test), n_boots))
        z_pred_train = np.empty((len(z_train), n_boots))

        z_pred_tests = np.empty((len(z_test), n_boots))
        z_pred_trains = np.empty((len(z_train), n_boots))

        for n in range(n_boots):
            _X_train, _z_train = resample(X_train, z_train, replace=True)

            _beta = self.__apply_model(model, _X_train, _z_train)
            _z_pred_train = self.__prediction(_X_train, _beta) 
            _z_pred_test = self.__prediction(X_test, _beta)

            z_pred_test[:, n] = _z_pred_test
            z_pred_train[:, n] = _z_pred_train

            z_pred_tests[:, n] = z_test
            z_pred_trains[:, n] = _z_train

        MSE_test = np.mean( np.mean((z_test.reshape(-1,1) - z_pred_test)**2, axis=1, keepdims=True) )
        MSE_train = np.mean( np.mean((z_pred_trains - z_pred_train)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_pred_tests - np.mean(z_pred_test, axis=1, keepdims=True))**2 )
        var = np.mean( np.var(z_pred_test, axis=1, keepdims=True) )
        
        return MSE_test, MSE_train, bias, var


    def cv(self, X, z, model, k=5, sklearn=False):
        """Performs the k-fold cross-validation resampling using the model."""

        X, z = shuffle(X, z, random_state=RAND)

        # Split into sub-vectors
        X_folds = np.array_split(X, k)
        z_folds = np.array_split(z, k)

        MSE_test_total = []
        MSE_train_total = []

        for i in range(k):
            _X_test = X_folds[i]
            _z_test = z_folds[i]
            _X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
            _z_train = np.concatenate(z_folds[:i] + z_folds[i+1:])

            _beta = self.__apply_model(model, _X_train, _z_train)
            _z_pred_train = self.__prediction(_X_train, _beta) 
            _z_pred_test = self.__prediction(_X_test, _beta)

            MSE_test_total.append(self.MSE(_z_test, _z_pred_test))
            MSE_train_total.append(self.MSE(_z_train, _z_pred_train))

        MSE_train = np.mean(MSE_train_total)
        MSE_test = np.mean(MSE_test_total)

        return MSE_test, MSE_train


if __name__ == "__main__":

    from data_prep import *

    N = 20

    z, x, y = prepare_data(N)

    ols_basic = Regression(z, x, y, degree=4, model='ols', sklearn=False, lambda_=None, test_size=0.2)
    print(ols_basic.MSE(ols_basic.z_test, ols_basic.z_pred_test, sklearn=True))
    print(ols_basic.MSE(ols_basic.z_test, ols_basic.z_pred_test, sklearn=False))
    print(ols_basic.R2(ols_basic.z_test, ols_basic.z_pred_test, sklearn=True))
    print(ols_basic.R2(ols_basic.z_test, ols_basic.z_pred_test, sklearn=False))