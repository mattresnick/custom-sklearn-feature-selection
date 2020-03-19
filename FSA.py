import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class FSAEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, kvars=[0], N_iter=500, mu=5, s=0.001, eta=1,
                 weight_sets=[], loss_sets=[], SI_sets=[], best_run_ind=0):
        self.kvars = kvars
        self.N_iter = N_iter
        self.mu = mu
        self.s = s
        self.eta = eta
        
        self.weight_sets = weight_sets
        self.loss_sets = loss_sets
        self.SI_sets = SI_sets
        self.best_run_ind = best_run_ind
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def lorenzLoss(self, N, X, y, beta):
        ybx = np.array(np.multiply(y, np.matmul(X, beta))) 
        logic_yield1 = np.array([1 if ybx[i]<=1 else 0 for i in range(len(ybx))])
        logic_yield2 = np.log(1+np.square(ybx-1))
        
        result = (1/N)*sum(np.multiply(logic_yield1,logic_yield2)) + self.s*np.linalg.norm(beta, ord=2)**2
        
        return result

    def invSchedule(self, M, k, i):
        Mi = k + (M - k)*max(0, (self.N_iter - 2*i)/(2*i*self.mu + self.N_iter))
        return Mi
    
    def keyFunc(self, val):
        return val[0]
        
    # Updates the weights -- --
    def update(self, N, X, y, beta):
    
        # [Nx1] = [Nx1] * [NxM]X[Mx1]
        ybx = np.array(np.multiply(y, np.matmul(X, beta)))
        
        # [Nx1]
        logic_yield1 = np.array([1 if ybx[i]<=1 else 0 for i in range(len(ybx))])
        
        # [Nx1]
        logic_yield2 = 2*np.divide((ybx-1),(1 + np.square(ybx-1)))
        
        # [Mx1] = [MxN]X[Nx1]X[Nx1] + [Mx1]
        dLdB = (1/N)*np.matmul(X.T, (np.multiply(np.multiply(logic_yield1, logic_yield2),y))) + 2*self.s*beta
        
        beta_new = np.array(beta) - self.eta*dLdB
        
        return beta_new

    
    def fit(self, X_o, y, stop_crit=1e-4, verbose=True, sample_weight=None):
        
        # For sklearn compatibility, validate shapes/labels and store classes
        X_o, y = check_X_y(X_o, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        
        # Initialize weight vector to size of max features.
        self.w_ = np.zeros(X_o.shape[1])
        
        for k in self.kvars:
            X = X_o
            beta = np.zeros(X.shape[1])
            num_features = 0
            save_indices_perm = [i for i in range(len(beta))]
            M = len(X[0])
            N = len(X)
            
            loss = []
            for step in range(1,self.N_iter+1):
                
                beta = self.update(N, X, y, beta)
                
                Mi = int(self.invSchedule(M, k, step))
                
                num_features = sum([1 for i in beta])
                
                sorter = [[np.abs(beta[i]), save_indices_perm[i], i] for i in range(len(beta))]
                
                sorter.sort(key=self.keyFunc, reverse=True)
                
                save_indices_perm = [v[1] for v in sorter[:Mi]]
                save_indices_temp = [v[2] for v in sorter[:Mi]]
                
                beta = np.array([beta[v] for v in save_indices_temp])
                X = np.array([[xo[v] for v in save_indices_temp] for xo in X])
                
                loss.append(self.lorenzLoss(N, X, y, beta))
                
                
                # Stopping criterion.
                if len(loss)>50 and (loss[-50] - loss[-1])/loss[-50] < stop_crit:
                    break
                
                if step%100==0 and verbose:
                    print('Iteration #%s | # of Features: %s | k: %s ' % (str(step),str(num_features),str(k)))
                    
            self.SI_sets.append(save_indices_perm)
            self.weight_sets.append(beta)
            self.loss_sets.append(loss)
        
        # Pick best set of weights based on loss.
        last_losses = np.array([losses[-1] for losses in self.loss_sets])
        self.best_run_ind = np.argmin(last_losses)
        self.w_ = self.weight_sets[self.best_run_ind]
        
        return self
    
    def predict(self, X):
        X = check_array(X)
        
        # Slice out selected features.
        X = np.array([[xo[v] for v in self.SI_sets[self.best_run_ind]] for xo in X])
        
        training_yield = np.dot(X, self.w_)
        preds = np.round(self.sigmoid(training_yield))        
        #returnpreds = np.array([np.array([p, 0]) for p in preds])                
        
        return preds
    
    def predict_proba(self, X):
        #check_is_fitted(self, 'w_')
        X = check_array(X)
        
        # Slice out selected features.
        X = np.array([[xo[v] for v in self.SI_sets[self.best_run_ind]] for xo in X])
        
        training_yield = np.matmul(X, self.w_)
        preds = self.sigmoid(training_yield)
        
        prob_out = np.array([np.array([1-p, p]) for p in preds])
        
        return prob_out
    
    # Getter function which returns best trained weights and selected indices.
    def getTrainedValues(self):
        check_is_fitted(self, 'w_')
        return self.w_, self.SI_sets[self.best_run_ind]
    
    # Getter function which returns, for all k values, weights, losses, 
    # selected indices, and the overall best run index.
    def getFullValues(self):
        check_is_fitted(self, 'w_')
        return self.weight_sets, self.loss_sets, self.SI_sets, self.best_run_ind
    

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

#print (check_estimator(FSAEstimator))