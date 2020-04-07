import numpy as np
from numpy import argmax, argmin, exp, log, ones, zeros, array
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from sklearn.preprocessing import StandardScaler

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
        
        - multi: Boolean value. True if multi-class option, False if binary.
        
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
    
    def __init__(self, kvars=[0], N_iter=500, mu=5, s=0.001, eta=1, multi=False,
                 weight_sets=[], loss_sets=[], SI_sets=[], best_run_ind=0):
        self.kvars = kvars
        self.N_iter = N_iter
        self.mu = mu
        self.s = s
        self.eta = eta
        
        self.multi=multi
        
        self.weight_sets = weight_sets
        self.loss_sets = loss_sets
        self.SI_sets = SI_sets
        self.best_run_ind = best_run_ind
        
        self.loadFunctions()
    
    def loadFunctions(self):
        '''
        Small math functions.
        '''
        
        # Simple Logistic function.
        self.sigmoid = lambda x: 1/(1+exp(-x))
        
        # Lorenz loss as described in FSA paper (Barbu et. Al.)
        self.lorenzLoss = lambda v: np.sum((v<=1)*log(1+(v-1)**2))
        
        # Derivative of Lorentz loss w.r.t. general input.
        self.dLorenz = lambda v: (v<=1)*((2*(v-1))/(1+(v-1)**2))
        
        # Prepend ones to input.
        self.prependOnes = lambda X: np.hstack((np.ones((X.shape[0],1),dtype=int),X))
    
    def softmax(self,x,prob=False):
        '''
        Simple softmax function for X@beta responses.
        
        Parameters:
            - x: Array-like response data.
            - prob: Boolean, determines if output is categorical or probabilistic.
            If False, the output will be the argmax of the softmax, if True, 
            the raw softmax probabilities will be returned.
        '''
        if prob:
            output = []
        else:
            output = zeros((x.shape[0],1))
        
        for i,x_i in enumerate(x):
            soft_row = []
            for x_ij in x_i:
                soft_row.append(exp(x_ij)/sum(exp(x_i)))
            if prob:
                output.append(soft_row)
            else:
                output[i] = argmax(soft_row)
        return output
    
    def labelsToOneHot(self,labels):
        '''
        Given categorical labels of more than two classes, will return
        one-hot arrays corresponding to each label.
        '''
        int_labels = [int(i) for i in labels]
        r = max(int_labels)
        
        one_hot_labels = []
        for label in int_labels:
            next_label = zeros(r+1)
            next_label[label] = 1
            one_hot_labels.append(next_label)
        
        return np.array(one_hot_labels)
    
    def fullLorenzLoss(self, N, ll_val, beta):
        '''
        Calculates full Lorenz loss with beta norm.
        
        Paramters:
            - N: Integer, number of samples.
            - ll_val: Float array or float. Reresents resulting Lorenz loss
            value without beta norm.
            - beta: Array-like, floating point values. Trainable weights.
        Returns:
            - Floating point value indicating loss.
        '''       
        result = (1/N)*ll_val - self.multi*log(2) + self.s*np.linalg.norm(beta, ord=2)**2
        return result

    def invSchedule(self, M, k, i):
        '''
        Feature selection schedule as described in FSA paper (Barbu et. Al.)
        
        Parameters:
            - M: Integer, number of features.
            - k: Integer, number of target features.
            - i: Integer, current iteration of training epoch per k value.
        Returns:
            - Integer representing number of features to be kept on the schedule.
        '''
        Mi = k+(M-k)*max(0,(self.N_iter-2*i)/(2*i*self.mu+self.N_iter))
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
        ybx = y*(X@beta)
        
        dLdB = (1/N)*X.T@((ybx<=1)*2*((ybx-1)/(1 + (ybx-1)**2))*y) + 2*self.s*beta
        beta_new = array(beta) - self.eta*dLdB
        
        if l:
            loss = self.fullLorenzLoss(N, ybx, beta_new)
            return beta_new, loss
        
        return beta_new
    
    
    
    def multiclass_update(self, N, X, y, beta, l=False):
        '''
        This yields Vapnik loss derived update. Vapnik loss is simply Lorentz loss
        of the difference u_y and u_k. These values are, per input sample i, the
        matrix product of the trainable weights beta at locations y_i and k,
        respectively, with input X at location i.
        
        Parameters:
            - N: Integer, number of samples.
            - X: Array-like, normalized floating point values. Input samples.
            - y: Array of one-hot ndarrays. Input target values.
            - beta: Array-like, floating point values. Trainable weights.
            - l: Boolean, loss is calculated and returned if True.
        
        Returns:
            - beta-like array of updated weight values.
            - Floating point loss value, returned only if l is True.
        '''
        r = y.shape[1]
        
        u_yi = np.reshape(np.diag((X@beta)@y.T),(N,1))@ones((1,r))
        u_k = X@beta
        
        dl = self.dLorenz(u_yi - u_k)
        H = np.sum(dl, axis=1)
        
        dLdW_jacobian = (-1)*dl + np.reshape(H,(N,1))@ones((1,r))*y
        
        # Derivative of loss w.r.t. one-hot-encoded beta weights.
        dLdB = (1/N)*(X.T@dLdW_jacobian) + 2*self.s*beta
        beta_new = beta - self.eta*dLdB
        
        if l:
            ll = self.lorenzLoss(u_yi - u_k)
            loss = self.fullLorenzLoss(N, ll, beta_new)
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
        self.w_ = zeros(X_o.shape[1])
        
        # Format data for multi-class, if needed.
        if self.multi:
            y = self.labelsToOneHot(y) # Labels into one-hot.
            X_o = self.prependOnes(X_o)
            
        for k in self.kvars:
            X = X_o
            N, M = X.shape
            r = y.shape[1]
            
            if self.multi:
                beta = zeros((M,r))
            else:
                beta = zeros(M)
            
            save_indices_perm = list(range(len(beta)))
            
            
            loss = []
            for step in range(1,self.N_iter+1):
                
                if self.multi:
                    beta, lloss = self.multiclass_update(N, X, y, beta, l=True)
                else:
                    beta, lloss = self.update(N, X, y, beta, l=True)
                
                Mi = int(self.invSchedule(M, k, step))
                
                if self.multi:
                    sorter = [[np.linalg.norm(beta[i], ord=1), save_indices_perm[i], i] for i in range(len(beta))]
                else:
                    sorter = [[np.abs(beta[i]), save_indices_perm[i], i] for i in range(len(beta))]
                
                sorter.sort(key=lambda x: x[0], reverse=True)
                
                save_indices_perm = [v[1] for v in sorter[:Mi]]
                save_indices_temp = [v[2] for v in sorter[:Mi]]
                
                beta = beta[save_indices_temp]
                X = X[:,save_indices_temp]
                
                loss.append(lloss)
                
                # Stopping criterion.
                if len(loss)>50 and (np.mean(loss[-50:-45])-np.mean(loss[-5:]))/np.mean(loss[-50:-45]) < stop_crit:
                    break
                
                if step%100==0 and verbose:
                    print('Iteration #%s | # of Features: %s | k: %s ' % (str(step),str(len(beta)),str(k)))
                    
            self.SI_sets.append(save_indices_perm)
            self.weight_sets.append(beta)
            self.loss_sets.append(loss)
        
        # Pick best set of weights based on loss.
        last_losses = array([losses[-1] for losses in self.loss_sets])
        self.best_run_ind = argmin(last_losses)
        self.w_ = self.weight_sets[self.best_run_ind]
        
        return self
    
    def predict(self, X, k_ind=-1, prob=False):
        '''
        Given input samples, will run the model forward to make predictions using
        set of "best" weights as determined during the training step. If a value
        for k_ind is given, the prediction will instead be made from the weights
        and features from the values of that index.
        
        Returns:
            - Array of integers indicating class prediction for every input sample.
            
            or
            
            - Array of arrays, each indicating probability for each class for
            every input sample, if prob=True
        '''
        
        if k_ind==-1:
            indices = self.SI_sets[self.best_run_ind]
            weights = self.w_
        else:
            indices = self.SI_sets[k_ind]
            weights = self.weight_sets[k_ind]
        
        # For multi-class prediction.
        if self.multi:
            X = self.prependOnes(X)
            X = X[:,indices]
            return self.softmax(X@weights, prob=prob)
            
        # For binary-class prediction.
        X = X[:,indices]
        if prob:
            return self.sigmoid(X@weights)
        else:
            return np.round(self.sigmoid(X@weights))
    
    def predict_proba(self, X, k_ind=-1):
        '''
        Given input samples, will run the model forward to return probabilistic
        predictions.
        
        Returns:
            - Array of arrays, each indicating probability for each class for
            every input sample.
        '''
        return self.predict(X, k_ind=-1, prob=True)
    
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
            train_error = (1 - np.mean(self.predict(x_train,k_ind=num)==y_train))*100
            test_error = (1 - np.mean(self.predict(x_test,k_ind=num)==y_test))*100
            
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
        y = array(loss_set)
        x = list(range(len(loss_set)))
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iteration Number', ylabel='Loss',
               title='Loss vs Iteration Number k='+str(k))
        plt.show()
    
    def getTrainedValues(self):
        '''
        Getter function which returns best trained weights and selected indices.
        '''
        check_is_fitted(self, 'w_')
        return self.w_, self.SI_sets[self.best_run_ind]
    
    def getFullValues(self):
        '''
        Getter function which returns, for all k values, weights, losses, 
        selected indices, and the overall best run index.
        '''
        check_is_fitted(self, 'w_')
        return self.weight_sets, self.loss_sets, self.SI_sets, self.best_run_ind




def normalize(train_feat, test_feat=None, std_sclr=None):
    '''
    A few ways to normalize the data. The non-StandardScaler() methods I made 
    from scratch, and should be able to handle zero values in the event of sparse data
    
    Parameters:
        - train_feat: Array-like feature data. Does not actually have to only
        be training data.
        - test_feat: Array-like feature data or None. If values are provided,
        it is normalized using the mean and std deviation of the training data.
        So if it is desired to normalize using its own mean/stddev, then pass
        testing data alone to train_feat.
        - std_sclr: Boolean, if True, will use StandardScaler() to normalize.
    '''
    
    if test_feat is not None:
        if std_sclr:
            s = StandardScaler()
            s.fit(train_feat)
            train_feat=s.transform(train_feat)
            test_feat=s.transform(test_feat)
            return train_feat, test_feat
        else:
            train_feat = array(train_feat)
            test_feat = array(test_feat)
            
            mean=list(np.mean(test_feat,0))
            stddev=np.std(test_feat,0)
            
            zerocols=(stddev!=0)
            stddev = list(stddev)
            
            for val in range(len(zerocols)):
                if zerocols[val]:
                    train_feat[:,val]=(train_feat[:,val]-mean[val])/stddev[val]
                    test_feat[:,val]=(test_feat[:,val]-mean[val])/stddev[val]
            
            return train_feat, test_feat
    
    else:
        mean=list(np.mean(train_feat,0))
        stddev=np.std(train_feat,0)
        
        zerocols=(stddev!=0)
        stddev = list(stddev)
        
        for val in range(len(zerocols)):
            if zerocols[val]:
                train_feat[:,val]=(train_feat[:,val]-mean[val])/stddev[val]
        
        return train_feat
        
