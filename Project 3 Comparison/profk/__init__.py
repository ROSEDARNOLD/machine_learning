
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


# In[ ]:

# Since there are some semantic differences between the way we approach working with 
# datasets based on the problem pace (inferential statistics vs structural modeling) 
# we instantiate an object to deal with features of both.  Since we aren't amending the sklearn API
# We can take the methods of 'fit' and 'predict' as a given; Whatever this class outputs
# should be directly passed to fit metods -> must yield X, y

class Dataset(object):
    
    def __init__(self, data):
        
        # since we are taking subsets later, no reason not to include a constant now
        
        data.insert(0, column='c', value=1)  # Insert throws an exception upon re-run -> data['c'] = 1... bring to front of df
        self.data = data
        
    # When do I drop n/a?
    def getFeatures(self, names): return self.data[feature_names]
    def getTargets(self, name): return self.data[target_name]
    
    # Pass directly to model fit methods
    def getXy(self, feature_names, target_name):       
        
        relevant = self.data[feature_names + target_name] # Not dropping data unneccessarily
        defined = relevant.dropna()
        
        X, y = np.array(defined[feature_names]), np.array(defined[target_name])
        y = y.reshape(len(y), ) # ravel warnings otherwise
        
        return X, y


# In[ ]:

class Selector():
        
    '''Non-operational bag of Methods for selecting between model types and feature inputs given a model.  Parent of the
    operant classes GenericClassifier and GenericRegressor'''    
    
    def ModelSelection(self, folds=10, rstate=420):
        
        cv_scores, cv_summary = {}, {}
        
        X = self.X
        y = self.y
        
        for name, model in self.Models.items():
            
            try:
            
                kfold = model_selection.KFold(n_splits=folds, random_state=rstate) 
                cv_result = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
                cv_summary = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
                cv_scores[name] = cv_result       
                
            
            except Exception as e:
                
                cv_scores[name] = e
                cv_summary[name] = e
        
        self.cv_scores = cv_scores
        
        # Print Summary
        for k, v in self.cv_scores.items():
    
           msg = "%s: %f (%f)" % (k, v.mean(), v.std())
           print(msg)
            
        # We could return a 'best model' for ease of use, but it will require us to be explicit about our selection criteria
        # (MSE, std errors, priors) up front -> seems exceptionally black boxy; we should probably just look at the results
        # and decide manually (What else would anyone pay us for?).

        
    def FeatureSelection(self, folds=10, rstate=420):
        
        '''This section is considerably more sketchy than the model selection component; needs work
        before results are to be trusted'''
        
        feature_cols = self.X.columns
        scores = {}
        kfold = model_selection.KFold(n_splits=folds, random_state=rstate)
        model = self.best_model
        model.fit(self.X, self.y)
        mse_scores = -model_selection.cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        scores[None] = mse_scores
        
        for dropped_x in feature_cols:
    
            feature_subset = [item for item in feature_cols if item != dropped_x]
            X2 = self.X[feature_subset]
            model = self.best_model
            model.fit(X2, y)
            mse_scores = -model_selection.cross_val_score(model, X2, y, cv=kfold, scoring='neg_mean_squared_error')
            scores[dropped_x] = mse_scores
        
        self.feature_scores = scores
        
        summary = {key: {'MEAN MSE': value.mean(), 'MEAN RMSE': np.sqrt(value).mean()} for key, value in scores.items()}
        self.feature_summary = summary 
        


# In[ ]:

class GenericClassifier(Selector):
    
    def __init__(self):
        
        pass
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        
        self.Models = {
                       
            'LR': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            'GBT': GradientBoostingClassifier(),
            'NB': GaussianNB(),
            'SVM': SVC(),
            'DT': DecisionTreeClassifier()
        
        }
        
class GenericRegressor(Selector):
    
    def __init__(self):
        
        pass
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        
        self.Models = {
                       
            # 'OLS': LinearRegression(),
            # etc..
        
        }

