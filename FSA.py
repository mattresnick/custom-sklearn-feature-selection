import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array


class FSAEstimator(BaseEstimator, ClassifierMixin):
    '''
    Estimator class for Feature Selection with Annealing. This will both select
    features for best prediction, as well as train weights for the prediction to
    be performed.
    
    Parameters:
        - kvars: List containing integer values. Each value for k is the target
        number of features.
        
        - N_iter: Integer valued number of iterations to be performed. Stopping
        criterion can be specified during call to training step.
        
        - Mu: Floating point or integer value. Model constant.
        
        - s: Floating point or integer value. Model constant.
        
        - eta: Floating point or integer value. Model constant/learning rate.
        
        - weight_sets: List of float ndarrays. Existing model weights. Should be 
        left empty unless using values from previously trained model.
        
        - loss_sets: List of float ndarrays. Loss values from previous training 
        run. Should be left empty unless using values from previously trained model.
        
        - SI_sets: List of integer ndarrays. Selected indices from a feature
        set from a previous run. Should be left empty unless using values from 
        previously trained model.
        
        - best_run_ind: Integer. Selected best run index from previously 
        trained model.
    '''
    
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
        '''
        Simple Logistic function.
        '''
        return 1/(1+np.exp(-x))
    
    def lorenzLoss(self, N, X, y, beta):
        '''
        Lorenz loss as described in FSA paper (Barbu et. Al.)
        
        Paramters:
            - N: Integer, number of samples.
            - X: Array-like, normalized floating point values. Input samples.
            - y: Float or integer ndarray. Input target values.
            - beta: Array-like, floating point values. Trainable weights.
        Returns:
            - Floating point value indicating loss.
        '''
        ybx = y*(X@beta)        
        result = (1/N)*sum(np.array([int(i<=1) for i in ybx])*np.log(1+np.square(ybx-1))) + \
        self.s*np.linalg.norm(beta, ord=2)**2
        
        return result

    def invSchedule(self, M, k, i):
        '''
        Feature selection schedule as described in FSA paper (Barbu et. Al.)
        
        Parameters:
            - M: Integer, number of features.
            - k: Integer, number of target features.
            - i: Integer, current interation of training epoch per k value.
        Returns:
            - Integer representing number of features to be kept on the schedule.
        '''
        Mi = k + (M - k)*max(0, (self.N_iter - 2*i)/(2*i*self.mu + self.N_iter))
        return Mi
    
    
    def update(self, N, X, y, beta, l=False):
        '''
        Function for updating model weights based on the derivative of the 
        Lorenz loss function with respect to the weights.
        
        Paramters:
            - N: Integer, number of samples.
            - X: Array-like, normalized floating point values. Input samples.
            - y: Float or integer ndarray. Input target values.
            - beta: Array-like, floating point values. Trainable weights.
            - l: Boolean, loss is calculated and returned if True.
        Returns:
            - beta-like array of updated weight values.
            - Floating point loss value, returned only if l is True.
        '''
        # [Nx1] = [Nx1] * [NxM]X[Mx1]
        ybx = y*(X@beta)
        
        # [Nx1]
        partial1 = np.array([int(i<=1) for i in ybx])
        
        # [Nx1]
        partial2 = 2*np.divide((ybx-1),(1 + np.square(ybx-1)))
        
        # [Mx1] = [MxN]X[Nx1]X[Nx1] + [Mx1]
        dLdB = (1/N)*X.T@(partial1*partial2*y) + 2*self.s*beta
        
        beta_new = np.array(beta) - self.eta*dLdB
        
        if l:
            loss = (1/N)*sum(np.array([int(i<=1) for i in ybx])*np.log(1+np.square(ybx-1))) + \
            self.s*np.linalg.norm(beta, ord=2)**2
            
            return beta_new, loss
        
        return beta_new

    
    def fit(self, X_o, y, stop_crit=1e-4, verbose=True, sample_weight=None):
        '''
        Training/fitting function.
        
        Parameters:
            - X_o: Array-like, normalized floating point values. Input samples.
            - y: Float or integer ndarray. Input target values.
            - stop_crit: Float or integer, stopping criterion. If the difference
            in loss value over 50 iterations does not exceed this value, then 
            training stops.
            - verbose: Boolean. Will allow progress to be printed to the console
            if True.
        '''
        # For sklearn compatibility, validate shapes/labels and store classes
        X_o, y = check_X_y(X_o, y)
        check_classification_targets(y)
        self.classes_ = unique_labels(y)
        
        # Initialize weight vector to size of max features.
        self.w_ = np.zeros(X_o.shape[1])
        
        for k in self.kvars:
            X = X_o
            beta = np.zeros(X.shape[1])
            save_indices_perm = list(range(len(beta)))
            M = X.shape[1]
            N = X.shape[0]
            
            loss = []
            for step in range(1,self.N_iter+1):
                beta, lloss = self.update(N, X, y, beta, l=True)
                
                Mi = int(self.invSchedule(M, k, step))
                
                sorter = [[np.abs(beta[i]), save_indices_perm[i], i] for i in range(len(beta))]
                sorter.sort(key=lambda x: x[0], reverse=True)
                
                save_indices_perm = [v[1] for v in sorter[:Mi]]
                save_indices_temp = [v[2] for v in sorter[:Mi]]
                
                beta = beta[save_indices_temp]
                X = X[:,save_indices_temp]
                
                loss.append(lloss)
                
                # Stopping criterion.
                if len(loss)>50 and (loss[-50] - loss[-1])/loss[-50] < stop_crit:
                    break
                
                if step%10==0 and verbose:
                    print('Iteration #%s | # of Features: %s | k: %s ' % (str(step),str(len(beta)),str(k)))
                    
            self.SI_sets.append(save_indices_perm)
            self.weight_sets.append(beta)
            self.loss_sets.append(loss)
        
        # Pick best set of weights based on loss.
        last_losses = np.array([losses[-1] for losses in self.loss_sets])
        self.best_run_ind = np.argmin(last_losses)
        self.w_ = self.weight_sets[self.best_run_ind]
        
        return self
    
    def predict(self, X, k_ind=-1):
        '''
        Given input samples, will run the model forward to make predictions using
        set of "best" weights as determined during the training step. If a value
        for k_ind is given, the prediction will instead be made from the weights
        and features from the values of that index.
        
        Returns:
            - Array of integers indicating class prediction for every input sample.
        '''
        X = check_array(X)
        
        if k_ind==-1:
            indices = self.SI_sets[self.best_run_ind]
            weights = self.w_
        else:
            indices = self.SI_sets[k_ind]
            weights = self.weight_sets[k_ind]
        
        # Slice out selected features.
        X = X[:,indices]
        
        preds = np.round(self.sigmoid(X@weights))
        return preds
    
    def predict_proba(self, X, k_ind=-1):
        '''
        Given input samples, will run the model forward to return probabilistic
        predictions.
        
        Returns:
            - Array of arrays, each indicating probability for each class for
            every input sample.
        '''
        preds = self.predict(X, k_ind=-1)
        prob_out = np.array([np.array([1-p, p]) for p in preds])
        
        return prob_out
    
    def performanceTable(self, x_train, x_test, y_train, y_test): 
        '''
        Prints a table to console with errors for the model with different
        number of features.
        '''
        check_is_fitted(self, 'w_')
        
        print (' ')     
        print ('   Train Error   |   Test Error   |   k features   ')
        print ('-----------------+----------------+----------------')
        
        for num in range(len(self.kvars)):
            train_error = (1 - np.mean(self.predict(x_train,num)==y_train))*100
            test_error = (1 - np.mean(self.predict(x_test,num)==y_test))*100
            
            f_kvars = "{:03.0f}".format(self.kvars[num])
            f_train_error = "{:05.2f}".format(train_error)
            f_test_error = "{:05.2f}".format(test_error)
        
            print ('      ', f_train_error, '    |     ', f_test_error, '    |       ',f_kvars,'     ')
    
    def lossPlotter(self, k):
        '''
        Plots loss vs training iteration for a given value of k.
        '''
        check_is_fitted(self, 'w_')
        
        loss_set = self.loss_sets[self.kvars.index(k)]
        y = np.array(loss_set)
        x = list(range(len(loss_set)))
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iteration Number', ylabel='Loss',
               title='Loss vs Iteration Number k='+str(k))
        plt.show()
    
    # Getter function which returns best trained weights and selected indices.
    def getTrainedValues(self):
        check_is_fitted(self, 'w_')
        return self.w_, self.SI_sets[self.best_run_ind]
    
    # Getter function which returns, for all k values, weights, losses, 
    # selected indices, and the overall best run index.
    def getFullValues(self):
        check_is_fitted(self, 'w_')
        return self.weight_sets, self.loss_sets, self.SI_sets, self.best_run_ind
