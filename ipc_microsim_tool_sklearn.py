
'''
Rohan Koodli
scikit-learn implementation of Electricity Consumption Surveys project
UN-DESA Modeling Tools
'''

import pandas as pd
import numpy as np
import os

#getting current working directory
path = os.getcwd() 

estimates = pd.read_csv(path + '/ipc_microsim_tool/data/WPP-ESTIMATES-1950-2015.tab.txt',
                        sep='\t', index_col=None, na_values='')


projections = pd.read_csv(path + '/ipc_microsim_tool/data/WPP-PROJECTIONS-2015-2100.tab.txt',
                          sep='\t',index_col=None,na_values='')

#print projections[0:2]

#estimates.drop(estimates.columns[[0,1,2,3]],axis=1,inplace=True)

estimates.drop(estimates.columns[[0,1,2,3]], axis=1, inplace=True)
#print estimates[0:2]
projections.drop(projections.columns[[0,1,2,3,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89]], axis=1, inplace=True)
#print projections[0:2]

medVariance_projections = np.array(projections[0:273])
highVariance_projections = np.array(projections[274:547])
lowVariance_projections = np.array(projections[548:821])
constantFertility_projections = np.array(projections[822:1095])

estimates = np.array(estimates)
from sklearn import ensemble,tree,neighbors
rfr = ensemble.RandomForestRegressor()
dtr = tree.DecisionTreeRegressor()
knr = neighbors.KNeighborsRegressor()
#nnn = neural_network.BernoulliRBM()

from sklearn.cross_validation import cross_val_score
print cross_val_score(rfr,estimates,medVariance_projections)
print cross_val_score(dtr,estimates,medVariance_projections)
print cross_val_score(knr,estimates,medVariance_projections)




