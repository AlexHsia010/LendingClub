#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import sys
# skip all warnings
import warnings
warnings.filterwarnings('ignore')


class Imputer():

    def __init__(self, data):

        self.data = data

    def remove_samples(self):

        row_missing = self.data.isnull().sum(axis=1) / self.data.shape[1]
        sample_index = row_missing[row_missing>0.8].index
        self.data.drop(labels=sample_index, axis = 'index', inplace = True)

    def hardship(self):

        # hardship related features
        hardship_features = self.data.columns[self.data.columns.str.contains("hardship")]
        # create a indicator feature
        self.data['is_hardship'] = self.data['hardship_status'].isna()
        # drop hardship features
        self.data.drop(hardship_features, axis=1, inplace=True)

    def joint(self):

        # get joint features
        joint_features = ['annual_inc', 'verification_status', 'dti', 'revol_bal']
        # get non-joint features
        non_joint_features = [x + '_joint' for x in joint_features]
        # new joint features
        for f in joint_features:
            # fill na by another column
            self.data[f + '_new'] = self.data[f + '_joint'].fillna(self.data[f], inplace=False)
        # drop original joint and non-joint features
        self.data.drop(joint_features + non_joint_features, axis=1, inplace=True)

    def sec_app(self):

        self.data.drop(['mths_since_last_major_derog', 'sec_app_mths_since_last_major_derog'],
                       axis=1, inplace=True)
        # apply function to convert non-nan value into numerical
        self.data['revol_util'] = self.data['revol_util'].apply(
            lambda x: float(x[:-1]) / 100 if str(x) != 'nan' else x)
        # impute NA with median value
        self.data['revol_util'].fillna(np.nanmedian(self.data['revol_util']), inplace=True)
        first_app_numerical_features = [
            'chargeoff_within_12_mths',
            'collections_12_mths_ex_med',
            'inq_last_6mths',
            'mort_acc',
            'num_rev_accts',
            'open_acc',
            'open_act_il']
        # handle numerical
        for f in first_app_numerical_features:
            self.data['sec_app_' + f].fillna(self.data[f], inplace=True)
            self.data[f + '_new'] = (self.data['sec_app_' + f] + self.data[f]) / 2
        # handle categorical variable
        self.data['sec_app_earliest_cr_line'].fillna(self.data['earliest_cr_line'], inplace=True)
        self.data['earliest_cr_line'] = self.data['earliest_cr_line'].apply(lambda x: x.split('-')[1])
        self.data['sec_app_earliest_cr_line'] = self.data['sec_app_earliest_cr_line'].apply(lambda x: x.split('-')[1])
        self.data['earliest_cr_line_new'] = self.data[['earliest_cr_line', 'sec_app_earliest_cr_line']].min(axis=1)
        self.data['earliest_cr_line_new'] = self.data['earliest_cr_line_new'].apply(lambda x: str(int(x / 10) * 10))
        # remove all relevant features
        for f in first_app_numerical_features + ['earliest_cr_line']:
            self.data.drop([f, 'sec_app_' + f], axis=1, inplace=True)

        return self.data

    def by_median(self):

        features = ['mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_inq',
                    'pct_tl_nvr_dlq', 'all_util', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'percent_bc_gt_75']
        # inplace by median
        for f in features:
            self.data[f].fillna(np.nanmedian(self.data[f]), inplace=True)

    def by_majority(self):

        features = ['last_credit_pull_d']
        # inplace by majority
        for f in features:
            self.data[f].fillna(self.data[f].value_counts().index[0], inplace=True)

    def by_zero(self):

        features = ['settlement_amount', 'settlement_percentage', 'settlement_term',
                    'mths_since_last_delinq', 'mo_sin_old_il_acct', 'num_tl_120dpd_2m', 'il_util_2']
        # create a new feature il_util_2
        self.data['il_util_2'] = self.data['total_bal_il'] / self.data['total_il_high_credit_limit'] * 100
        # fill NAs with 0
        for f in features:
            self.data[f].fillna(0, inplace=True)

    def by_string(self):

        other_features = ['last_pymnt_d', 'next_pymnt_d', 'emp_title', 'emp_length']
        # inplace by other
        for f in other_features:
            self.data[f].fillna('Other', inplace=True)

        # remove space in the beginning and end
        self.data['emp_title'] = self.data['emp_title'].apply(lambda x: x.strip())
        # choose the last word as the general title
        self.data['emp_title'] = self.data['emp_title'].apply(
            lambda x: x.split(' ')[-1].lower() if ' ' in x else x.lower())
        # choose top 20 in order to contain half of the variance
        top_20_title = self.data['emp_title'].value_counts()[:20].index.tolist()
        self.data['emp_title'] = self.data['emp_title'].apply(lambda x: x if x in top_20_title else 'minority')

    def _remove(self):

        # remove other relevant features
        features = ['settlement_date', 'settlement_status', 'mths_since_recent_revol_delinq',
                    'il_util', 'debt_settlement_flag_date', 'id', 'url', 'desc', 'member_id',
                    'orig_projected_additional_accrued_interest', 'deferral_term', 'payment_plan_start_date',
                    'mths_since_last_record', 'mths_since_recent_bc_dlq','sec_app_revol_util']

        one_value_features = []
        for f in self.data:
            if len(self.data[f].value_counts()) == 1:
                one_value_features.append(f)

        self.data.drop(one_value_features + features, axis=1, inplace=True)

    def main(self):

        print ("Start missing value imputation...")
        # print ("Remove some samples...")
        self.remove_samples()
        # print ("Deal with Hardship features...")
        self.hardship()
        # print ("Deal with Joint and Second APP features...")
        self.joint()
        self.sec_app()
        # print ("Impute by median...")
        self.by_median()
        # print ("Impute by majority...")
        self.by_majority()
        # print ("Impute by zero...")
        self.by_zero()
        # print ("Impute by string...")
        self.by_string()
        # print ("Remove irrelevent features...")
        self._remove()

        # check missing value
        if self.data.isnull().sum().sum() != 0:
            print ("There are still %d missing values in the dataset."%self.data.isnull().sum().sum())
            sys.exit()

        return self.data
