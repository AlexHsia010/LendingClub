#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Authors: Eric@FLAGDream <eric.d@flagdream.com>

import pandas as pd
# skip all warnings
import warnings
warnings.filterwarnings('ignore')

def get_data(input_path):

	# get loan data
	df = pd.read_csv(input_path, header=1)

	return df