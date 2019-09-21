#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Modeling():

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train


    def _Ridge_Regression(self, alphas):

    	# initialize a model object
        RidgeReg = RidgeCV(alphas = alphas, store_cv_values=True)
        # train model
        RidgeReg.fit(self.X_train, self.y_train)
        # get optimal alpha 
        print ("The best alpha in Ridge Regression: ", RidgeReg.alpha_)

        return RidgeReg


    def _LASSO_Regression(self, alphas):

    	# initialize a model object
        LassoReg = LassoCV(alphas = alphas, random_state=42, verbose=True, n_jobs=12)
        # train model
        LassoReg.fit(self.X_train, self.y_train)
        # get optimal alpha 
        print ("The best alpha in LASSO Regression: ", LassoReg.alpha_)

        return LassoReg


    def _LASSOLars_Regression(self):

    	# initialize a model object
        LassoLarsReg = LassoLarsCV(cv = 10)
        # train model
        LassoLarsReg.fit(self.X_train, self.y_train)
        # optimal alpha
        print ("The best alpha in LAR is: ", LassoLarsReg.alpha_)

        return LassoLarsReg


    def _ElasticNet_Regression(self, alphas, l1_ratio=0.5):

    	# initialize a model object
        ElasticNetReg = ElasticNetCV(alphas = alphas_elasticnet, l1_ratio = 0.5, n_jobs=-1)
        # train model
        ElasticNetReg.fit(self.X_train, self.y_train)
        # get optimal alpha 
        print ("The best alpha in LAR is: ", ElasticNetReg.alpha_)

        return ElasticNetReg


    def _Gradient_Boosting(self):

    	# initialize model
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        # train model
        gb.fit(self.X_train, self.y_train)

        return gb


    def save_model(self, model, path):

	    with open(path, 'wb') as clf:
	        pickle.dump(model, clf) 





















