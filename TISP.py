import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class TISPEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, iterations=200, lambda_=0.01):
        self.iterations = iterations
        self.lambda_ = lambda_
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # Penalty using hard thresholding
    def thetaPenalty(self, t, eta=0):
        if np.abs(t)<=self.lambda_:
            return 0
        else:
            return t/(1+eta)
        
    # Updates the weights using log reg with theta penalty
    def update(self, w, X, y, eta):
        
        exp_term = 1 + np.exp(-1*np.matmul(X,w))
        weight_update_array = w + eta*np.matmul(X.T,(y-(1/exp_term)))
        
        w_new = [self.thetaPenalty(t) for t in weight_update_array]
        return w_new
    
    def fit(self, X, y, sample_weight=None):
        
        # For sklearn compatibility, validate shapes/labels and store classes
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        
        # Initialize values for training
        self.w_ = np.zeros(X.shape[1])
        eta = 1/len(X)
        num_features=0
        
        # Train via given number of iterations
        for step in range(1,self.iterations+1):
            self.w_ = self.update(self.w_, X, y, eta)
            num_features = sum([1 for i in self.w_ if np.abs(i)>0])
        
        #print ('num feats:',num_features)
        return self
    
    def predict(self, X):
        
        # Check is fit had been called and validate input (for sklearn compatibility)
        #check_is_fitted(self, 'w_')
        X = check_array(X)
        
        training_yield = np.matmul(X, self.w_)
        preds = np.round(self.sigmoid(training_yield))
        
        returnpreds = np.array([np.array([p, 0]) for p in preds])
        
        return returnpreds
    
    def predict_proba(self, X):
        
        # Check is fit had been called and validate input (for sklearn compatibility)
        #check_is_fitted(self, 'w_')
        X = check_array(X)
        
        training_yield = np.matmul(X, self.w_)
        preds = self.sigmoid(training_yield)
        
        prob_out = np.array([np.array([1-p, p]) for p in preds])
        
        return prob_out
    
    def getWeights(self):
        check_is_fitted(self, 'w_')
        return self.w_
    

def normalize(train_feat, test_feat):
    
    train_feat = np.array(train_feat)
    test_feat = np.array(test_feat)
    
    mean=list(np.mean(test_feat,0))
    stddev=np.std(test_feat,0)
    
    zerocols=(stddev!=0)
    stddev = list(stddev)
    
    for val in range(len(zerocols)):
        if zerocols[val]:
            train_feat[:,val]=(train_feat[:,val]-mean[val])/stddev[val]
            test_feat[:,val]=(test_feat[:,val]-mean[val])/stddev[val]
    
    return train_feat, test_feat

#print (check_estimator(TISPEstimator))