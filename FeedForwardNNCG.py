import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt
from sklearn.model_selection import train_test_split

from utils import augment, sigmoid, net_signal

class FeedForwardNNCG:

    def __init__(self, n_inputs, n_classes, type, n_hidden):
        self.I = n_inputs
        self.J = n_inputs if n_hidden == 0 else n_hidden
        self.K = 1 if n_classes == 2 else n_classes

        if type == 'classification':
            self.clf = True
            self.regr = False
        elif type == 'regression':
            self.clf = False
            self.regr = True
        else:
            print('ERROR: Invalid type specified.')
            exit()

        # initialize weights -> represented as one vector
        self.W = {}
        r = 1/sqrt(self.I+1)
        self.W[0] = np.random.rand(self.I+1, self.J)*2*r - r
        r = 1/sqrt(self.J+1)
        self.W[1] = np.random.rand(self.J+1, self.K)*2*r - r


    def fit(self, X, y):
        '''Train the weights of the neural network using conjugate gradient
        with fletcher reeves updates algorithm.

        Parameters:
        X (numpy.ndarray): N x d array of training samples.
        y (numpy.ndarray): N x k array of targets
        ''' 
        # Augment input to account for bias term
        X = augment(X)

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        N,_ = X.shape

        W = {}
        W[0] = self.W[0]
        W[1] = self.W[1]

        w = np.hstack((W[0].reshape(1,-1),W[1].reshape(1,-1)))
        print(w)    

        # determine n_w = #model parameters (weights + biases)
        n_w = (self.I+1)*self.J + (self.J+1)*self.K
        eps=1e-11

        Ev_avg = 0
        Ev_std = 0
        Ev_best = 100
        W_best = w
        Ev_best_overall = 100
        W_best_overall = w

        x = 0
        self.train_errors = []
        self.val_errors = []
        stop = False
        while (not stop) and x < 5: 
            # Calculate gradient E'(w(0))
            E, dE = self._accummulate(X, y, w)

            # Compute first direction vector p(0) = -E'(w(0))            
            p = -dE
            
            for t in range(n_w):
                # Calculate step size
                # n(t) = min E(w(t) + np(t))
                n = self._line_search(X, y, w, p)
                
                # restart if stopping conditions not yet reached
                if n == 0:
                    break

                # Calculate new weight vector
                # w(t+1) = w(t) + n(t)p(t)
                w_new = w + n*p

                # Calculate scale factors
                # B(t) = (E'(w(t+1))^T x E'(w(t+1))) / (E'(w(t))^T x E(w(t)))
                E_new, dE_new = self._accummulate(X, y, w_new)
                num =  dE_new.reshape(1,-1).dot(dE_new.reshape(-1,1))
                if num <= eps:
                    print('numerator too small')
                    break

                B = num/dE.reshape(1,-1).dot(dE.reshape(-1,1))

                # Calculate new direction vector
                # p(t+1) = -E'(w(t+1)) + B(t)p(t)
                p = -dE_new + B*p

                # Prepare for new iteration
                w = np.copy(w_new)
                E, Ed = E_new, dE_new

                self.train_errors.append(E)
                Ev = self._val_error(X_val, y_val, w)
                self.val_errors.append(Ev)

                if t > 20 and Ev > Ev_avg + Ev_std:
                    stop = True
                Ev_avg, Ev_std = self._running_stats(Ev, x*n_w + t, Ev_avg, Ev_std)

                if Ev < Ev_best:
                    Ev_best = Ev
                    W_best = w
                    t_best = x*n_w + t

            if Ev_best < Ev_best_overall:
                Ev_best_overall = Ev_best
                W_best_overall = W_best
            x += 1

        self.W[0] = W_best_overall[0,:(self.I+1)*self.J].reshape((self.I+1),self.J)
        self.W[1] = W_best_overall[0,(self.I+1)*self.J:].reshape((self.J+1),self.K)


    def show_err_vs_epoch(self):
        '''Plot validation and training errors per epoch made during 
        training.
        '''
        plt.plot(self.train_errors, c='teal')
        plt.plot(self.val_errors, c='r')
        plt.xlabel('Epoch')
        plt.ylabel('SSE')
        plt.legend(['Training', 'Validation'])
        plt.show()

    
    def pred(self, X):
        '''Calculate predicted values for given samples.

        Parameters:
        X (numpy.ndarray): N x d array of samples.
        '''
        X = augment(X)
        y_pred = [self._pred(X[i,:]) for i in range(X.shape[0])]
        return np.array(y_pred)

    
    def _pred(self, x):
        h_actv, o_actv = self._forward_pass(self.W, x)
        if self.clf == True:
            pred = np.zeros((len(o_actv)))
            if self.K > 1:
                o_actv = (1/np.sum(o_actv)) * o_actv
                pred[np.argmax(o_actv)] = 1
            else:
                pred = [round(o_actv[0])]
            return pred
        else:
            return o_actv


    def _running_stats(self, err, t, m, std):
        if t == 0: return (m, std)
        m_prev = m
        m = m + (err - m)/t
        std = std + (err - m)*(err - m_prev)
        return (m, std)


    def _val_error(self, X, y, w):
        N, d = X.shape
        W_vij = w[0,:(self.I+1)*self.J].reshape((self.I+1),self.J)
        W_wkj = w[0,(self.I+1)*self.J:].reshape((self.J+1),self.K)
        W = {}
        W[0] = W_vij
        W[1] = W_wkj
        sse = 0
        for i in range(N):
            label = y[i,:]
            h_actv, o_actv = self._forward_pass(W,X[i,:])
            if self.clf == True and self.K > 1:
                o_actv = (1/np.sum(o_actv)) * o_actv
            sse += np.sum([(label[k] - o_actv[k])**2 for k in range(self.K)])
        return sse/(2*N)


    def _accummulate(self, X, y, w): 
        N, d = X.shape
        W_vij = w[0,:(self.I+1)*self.J].reshape((self.I+1),self.J)
        W_wkj = w[0,(self.I+1)*self.J:].reshape((self.J+1),self.K)
        W = {}
        W[0] = W_vij
        W[1] = W_wkj
        
        dE_wkj = np.zeros((self.J+1, self.K))
        dE_vji = np.zeros((self.I+1, self.J))
        E = 0

        for p in range(N):
            pattern = X[p,:]
            target = y[p,:]
            hidden_activations, output_activations = self._forward_pass(W, pattern)

            output_err_signals = [
                self._err_signal_output(output_activations[k], target[k]) for k in range(self.K)
            ]
            hidden_err_signals = [
                self._err_signal_hidden(output_err_signals, W[1][j,:], hidden_activations[j]) for j in range(self.J)
            ]
            
            dE_wkj += np.array([[(output_err_signals[k]*hidden_activations[j]) for k in range(self.K)] for j in range(self.J+1)])

            dE_vji += np.array([[(hidden_err_signals[j]*pattern[i]) for j in range(self.J)] for i in range(self.I+1)])

            if self.clf == True  and self.K > 1:
                output_activations = output_activations/np.sum(output_activations)
            E += np.sum([(target[k] - output_activations[k])**2 for k in range(self.K)])
        
        E = E/(2*N)
        dE = np.hstack((dE_vji.reshape(1,-1), dE_wkj.reshape(1,-1)))/N
        return (E, dE)


    def _forward_pass(self, W, p):
        # Hidden neurons
        net_signals = np.zeros((self.J))
        for j in range(self.J):
            net_signals[j] = net_signal(W[0][:,j], p)
        hidden_activations = [sigmoid(net_signals[j]) for j in range(self.J)]

        # Output neurons
        hidden_activations = np.append(hidden_activations, [-1])
        p = hidden_activations
        net_signals = np.zeros((self.K))
        for k in range(self.K):
            net_signals[k] = net_signal(W[1][:,k], p)
        if self.clf == True:
            output_activations = [sigmoid(net_signals[k]) for k in range(self.K)]
        else:
            output_activations = net_signals

        return (np.array(hidden_activations), np.array(output_activations))


    def _err_signal_output(self, o_k, t_k):
        if self.clf == True:
            return -1*(t_k - o_k)*(1-o_k)*o_k
        else:
            return -1*(t_k - o_k)
    

    def _err_signal_hidden(self, err_o, w_j, y_j):
        return np.sum(err_o*w_j*(1-y_j)*y_j)


    def _line_search(self, X, y, w, p):
        eps = 1e-9
        maximum_iterations = 100
        E, dE = self._accummulate(X, y, w)
        
        # checking that the given direction is indeed a descent direciton
        if np.vdot(p, dE)  >= 0:
            print('Not descent direction')
            return 0
        
        else:
            # setting an upper bound on the optimum.
            MIN_t = 0
            MAX_t = 1
            iterations = 0

            E, dE = self._accummulate(X, y, w + MAX_t*p)

            while np.vdot(p, dE) < 0:
                MAX_t *= 2
                E, dE = self._accummulate(X, y, w + MAX_t*p)
                
                iterations += 1
                
                if iterations >= maximum_iterations:
                    raise ValueError("Too many iterations")
            
            # bisection search in the interval (MIN_t, MAX_t)
            iterations = 0

            while True:        
                t = (MAX_t + MIN_t)/2
                
                E, dE = self._accummulate(X, y, w + t*p)          
                suboptimality = abs(np.vdot(p, dE))*(MAX_t - t)
                
                if suboptimality <= eps:
                    break
                
                if np.vdot(p, dE) < 0:
                    MIN_t = t
                else:
                    MAX_t = t
                
                iterations += 1
                if iterations >= maximum_iterations:
                    raise ValueError("Too many iterations")
                
            return t