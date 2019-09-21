#!/usr/bin/env python
#-*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer():

	def __init__(self, data):
		"""
		Input data after missing value imputation
		"""
		self.data = data


	def process_y(self):

		self.data['int_rate'] = self.data['int_rate'].apply(lambda x: float(x[:-1]))


	def zip_code(self):

		self.data['zip_code'] = self.data['zip_code'].apply(lambda x: x[0]+x[-1])


	def _remove(self):
		"""
		Remove irrelevant variables
		"""
		features = ['sub_grade', 'addr_state', 'loan_status','grade']
		self.data.drop(features, axis=1, inplace=True)


	def _get_dummies(self):

		# extract categorical variables
		dummy_columns = [x for x in self.data if self.data[x].dtype == 'object']
		self.data = pd.get_dummies(self.data, columns = dummy_columns)


	def split_train_test(self):

		x = self.data[self.data.columns.difference(['int_rate'])].astype(float)
		y = self.data['int_rate'].values

		# Split data into train and test (80% & 20%)
		x_train, x_test, y_train, y_test = train_test_split(
		    x, y, test_size = 0.2, random_state = 42)

		# initialize a scaler object
		scaler = StandardScaler()
		# transform training set
		x_train_std = scaler.fit_transform(x_train)
		# the same transform for test set
		x_test_std = scaler.transform(x_test)

		return x_train_std, x_test_std, y_train, y_test


	def main(self):

		print ("Start feature engineering...")
		self.process_y()
		self.zip_code()
		self._remove()
		self._get_dummies()

		return self.split_train_test()















		
