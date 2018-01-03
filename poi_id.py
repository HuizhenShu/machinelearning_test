#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectPercentile, f_classif
import pandas as pd
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
'shared_receipt_with_poi'] # You will need to use more features,'ft_with_poi'


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL', 0)#Remove outliers 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

data_frame = pd.DataFrame(data_dict)

# print data_frame
# print features_list[:-1]
for key in features_list[:-1]:
	checkdata = data_frame.ix[key]
	#print checkdata
	#print checkdata.value_counts()#[0]#.ix['NaN']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in data_dict:
	#print data_dict[key].keys()
	data_dict[key]['ft_with_poi'] = data_dict[key]['from_this_person_to_poi']+data_dict[key]['from_poi_to_this_person']
	if data_dict[key]['ft_with_poi'] =='NaNNaN':
		#print data_dict[key]['ft_with_poi']
		data_dict[key]['ft_with_poi'] = data_dict[key]['ft_with_poi'].replace('NaNNaN','NaN')
		#print data_dict[key]['ft_with_poi']
	#print data_dict[key]['ft_with_poi']
my_dataset = data_dict

#print my_dataset.values()[0].keys()
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# salarys = []
# maxsalary = 0
# for i in range(len(features)):
# 	if features[i][0]>maxsalary:
# 		maxsalary =features[i][0]
# 	salarys.append(features[i][0])



##use image to search outlier
# matplotlib.pyplot.scatter( labels, salarys)
# matplotlib.pyplot.xlabel("poi")
# matplotlib.pyplot.ylabel("salary")
# matplotlib.pyplot.show()
##find out who has the extremely high salary
# for i in range(len(data_dict)):
# 	if data_dict[data_dict.keys()[i]]['salary']==maxsalary:
# 		pass
		#print data_dict.keys()[i]
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree,grid_search
from sklearn import neural_network
from sklearn.neighbors.nearest_centroid import NearestCentroid 


from sklearn.metrics import accuracy_score,recall_score
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
# kf = KFold(len(features),5)
#print features

cv = StratifiedShuffleSplit(labels, 3, 0.3,0.7,random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
	#clf = GaussianNB()
	clf = NearestCentroid(shrink_threshold =1.5)
	#clf = neural_network.MLPClassifier(activation='relu')
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = SVC()
# clf = grid_search.GridSearchCV(svr, parameters)
	#clf =SVC(kernel='rbf',C=10000)
	#clf = tree.DecisionTreeClassifier(min_samples_split=10, random_state=50,min_samples_leaf=1 )#
	### Task 5: Tune your classifier to achieve better than .3 precision and recall 
	### using our testing script. Check the tester.py script in the final project
	### folder for details on the evaluation method, especially the test_classifier
	### function. Because of the small size of the dataset, the script uses
	### stratified shuffle split cross validation. For more info: 
	### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

	# Example starting point. Try investigating other evaluation techniques!
	from sklearn.cross_validation import train_test_split
	features_train, features_test, labels_train, labels_test = \
	    train_test_split(features, labels, test_size=0.3, random_state=42)
	# selector = SelectPercentile(f_classif, percentile=0.8)

	# selector.fit(features_train, labels_train)
	# #print selector.scores_
	# # for i in range(1,len(features_list)):
	# # 	print features_list[i] ,":", (selector.scores_)[i-1]
	# features_train = selector.transform(features_train)#.toarray()
	# features_test  = selector.transform(features_test)#.toarray()

	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)

	print 'accuracy_score:',accuracy_score(pred,labels_test)
	print 'recall_score:',recall_score(pred,labels_test)
	### Task 6: Dump your classifier, dataset, and features_list so anyone can
	### check your results. You do not need to change anything below, but make sure
	### that the version of poi_id.py that you submit can be run on its own and
	### generates the necessary .pkl files for validating your results.

	dump_classifier_and_data(clf, my_dataset, features_list)

