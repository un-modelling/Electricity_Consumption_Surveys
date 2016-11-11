'''
Author: Rohan Koodli
Implementing Electricity Consumption Surveys with sklearn
Version 2.0
This file is under active development.
'''

import pandas as pd
import numpy as np

# reading in aztlan data on income, kilowatt-hour per capita, and wgt
aztlan = pd.read_csv('\Users\Rohan\Documents\GitHub\Electricity_Consumption_Surveys\ipc_microsim_tool\data\example_aztlan_seed.tab.txt',
                     sep='\t', index_col=None, na_values='')

X = np.array(aztlan['income'])
#print (X)
#X = X[:, None]
# X is all the income data values
X1 = []
for i in X:
    X1.append([i])

#print X1
y = np.array(aztlan['kwhpc'])
# y is all the kilowatt-hour values
print len(y)

from sklearn import neighbors,tree,ensemble,svm
rfr = ensemble.RandomForestRegressor()
dtr = tree.DecisionTreeRegressor()
knr = neighbors.KNeighborsRegressor()
svr = svm.SVR()
#nnn = neural_network.BernoulliRBM()
''''''
from sklearn.cross_validation import cross_val_score
print cross_val_score(rfr,X1,y)
print cross_val_score(dtr,X1,y)
print cross_val_score(knr,X1,y)
print cross_val_score(svr,X1,y)
''''''
from matplotlib import pyplot as plt
import seaborn; seaborn.set()

plt.scatter(X1,y)
