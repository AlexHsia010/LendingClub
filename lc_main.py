#!/usr/bin/env python
#-*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from lc_imputation import Imputer
from lc_get_data import get_data
from lc_feature_engineer import FeatureEngineer
from lc_modeling import Modeling 
from lc_test import Testing
# skip all warnings
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    ######## Read data ########
    loan = get_data('./data/LoanStats_2018Q1.csv')
    imputer = Imputer(loan)
    imputed_loan = imputer.main()

    # save intermediate data file
	# imputed_loan.to_csv('./data/LoanStats_model.csv', encoding='utf-8', index=False)

	######## Feature engineering ########
    featureEng = FeatureEngineer(imputed_loan)
    X_train, X_test, y_train, y_test = featureEng.main()

    ######## Train model ########
    ml = Modeling(X_train, y_train)
    alphas = sorted([10**r for r in np.random.uniform(-6,-3,size=100)])
    print ("Training Ridge Regression model...")
    ridge = ml._Ridge_Regression(alphas)
    print ("Training Gradient Boosting model...")
    gb = ml._Gradient_Boosting()

	######## Save Model ########
    ml.save_model(ridge, './model/ridge.pkl')
    ml.save_model(gb, './model/gb.pkl')

	######## Test Model ########
    test = Testing(X_test, y_test)
    rmse_ridge = test.test(ridge)
    rmse_gb = test.test(gb)
    print ("Root mean square error for Ridge: %.2f"%rmse_ridge)
    print ("Root mean square error for Gradient Boosting: %.2f"%rmse_gb)





