#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Testing():

	def __init__(self, X_test, y_test):

		self.X_test = X_test
		self.y_test = y_test


	def test(self, model):

		y_pred = model.predict(self.X_test)
		rmse = np.sqrt(np.mean((self.y_test - y_pred)**2))

		return rmse

