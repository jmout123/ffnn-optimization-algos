import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt, floor
from sklearn.model_selection import train_test_split

from utils import augment, sigmoid, net_signal

class FeedForwardNNGD:

    HIDDEN = 0
    OUTPUT = 1
    
    def __init__(self, n_inputs, n_classes, n_hidden, type, lr):
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

        # initialize weights
        self.W = {}
        r = 1/sqrt(self.I+1)
        self.W[self.HIDDEN] = np.random.rand(self.I+1, self.J)*2*r - r
        r = 1/sqrt(self.J+1)
        self.W[self.OUTPUT] = np.random.rand(self.J+1, self.K)*2*r - r

        # initialize learning rate 
        self.eta = lr
        
        # initialize momentum
        self.alpha = {}
        self.alpha[self.HIDDEN] = 0.1*np.ones((self.I+1, self.J))
        self.alpha[self.OUTPUT] = 0.1*np.ones((self.J+1, self.K))

        # initialize prev step sizes
        self.prev_step = {}
        self.prev_step[self.HIDDEN] = np.zeros((self.I+1, self.J))
        self.prev_step[self.OUTPUT] = np.zeros((self.J+1, self.K))
        self.prev_dE_vji = np.zeros((self.I+1, self.J))
        self.prev_dE_wkj = np.zeros((self.J+1, self.K))


    def fit(self, X, y):
        '''Train the weights of the neural network using stochastic gradient descent optimization.

        Parameters:
        X (numpy.ndarray): N x d array of training samples.
        y (numpy.ndarray): N x k array of targets
        ''' 
        # Add bias term to data
        X = augment(X)

        # Split into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        N,_ = X_train.shape
        
        # initialize starting epoch
        t = 1

        #stopping conditions
        max_iter = 1000

        # Early stopping parameters
        Ev_avg = 0
        Ev_std = 0
        Et = 100
        Ev_best = 100
        t_best = t
        W_best = {}

        self.train_errors = []
        self.validation_errors = []
        stop = False

        while (not stop) and t <= max_iter:
            Et = 0
            Et_prev = 1
            idx = list(range(0,N))
            random.shuffle(idx)

            for p in idx:
                pattern = X_train[p, :]
                label = y_train[p, :]

                # feed forward phase
                hidden_activations, output_activations = self._forward_pass(pattern)
                
                # Calculate error signals
                output_err_signals = np.array([
                    self._err_signal_output(output_activations[k], label[k]) for k in range(self.K)
                ])
                hidden_err_signals = np.array([
                    self._err_signal_hidden(output_err_signals, self.W[self.OUTPUT][j,:], hidden_activations[j]) for j in range(self.J)
                ])

                # Backward propagation phase
                self._backprop(output_err_signals, hidden_err_signals, hidden_activations, pattern)

                # Update error
                if self.clf == True and self.K > 1:
                    output_activations = output_activations/np.sum(output_activations)
                Et += (1/(2*N))*np.sum([(label[k] - output_activations[k])**2 for k in range(self.K)])
                    
            Ev = self._val_error(X_val, y_val)
            self.train_errors.append(Et)
            self.validation_errors.append(Ev)
            Et_prev = Et

            # Stop if overfitting is detected
            if t > 20 and Ev > Ev_avg + Ev_std:
                stop = True
            Ev_avg, Ev_std = self._running_stats(Ev, t, Ev_avg, Ev_std)
                
            if Ev < Ev_best:
                Ev_best = Ev
                Et_best = Et
                W_best[self.HIDDEN] = self.W[self.HIDDEN]
                W_best[self.OUTPUT] = self.W[self.OUTPUT]
                t_best = t
                
            t += 1

        self.W[self.HIDDEN] = W_best[self.HIDDEN]
        self.W[self.OUTPUT] = W_best[self.OUTPUT]


    def show_err_vs_epoch(self):
        '''Plot validation and training errors per epoch made during 
        training.
        '''
        plt.plot(self.train_errors, c='teal')
        plt.plot(self.validation_errors, c='r')
        plt.xlabel('Epoch')
        plt.ylabel('SSE')
        plt.legend(['Training', 'Validation'])
        plt.show()


    def pred(self, X):
        '''Calculate predicted values for given samples.

        Parameters:
        X (numpy.ndarray): N x d array of samples.
        '''
        N, d = X.shape
        Xb = -1*np.ones((N, 1))
        X = np.hstack((X, Xb))
        y_pred = [self._pred(X[i,:]) for i in range(X.shape[0])]
        return np.array(y_pred)

    
    def _pred(self, x):
        h_actv, o_actv = self._forward_pass(x)
        if self.clf:
            pred = np.zeros((len(o_actv)))
            if self.K > 1:
                o_actv = (1/np.sum(o_actv)) * o_actv
                pred[np.argmax(o_actv)] = 1
            else:
                pred = [round(o_actv[0])]
                #print(pred)
            return pred
        else:
            return o_actv



    def _forward_pass(self, p):
        # Calculate activation of each hidden unit and output unit for 
        # pattern p in the forward pass phase

        # Hidden neurons
        net_signals = np.zeros((self.J))
        for j in range(self.J):
            net_signals[j] = net_signal(self.W[self.HIDDEN][:,j], p)
        hidden_activations =[sigmoid(net_signals[j]) for j in range(self.J)]

        # Output neurons
        hidden_activations = np.append(hidden_activations, [-1])
        p = hidden_activations
        net_signals = np.zeros((self.K))
        for k in range(self.K):
            net_signals[k] = net_signal(self.W[self.OUTPUT][:,k], p)
        if self.clf:
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


    def _backprop(self, err_o, err_h, actv_h, p):
        # update input-hidden weights: v_ji
        dE_vji = np.array([[(err_h[j]*p[i]) for j in range(self.J)] for i in range(self.I+1)])
        step_sizes = -self.eta*dE_vji

        self.W[self.HIDDEN] += step_sizes + self.alpha[self.HIDDEN]*self.prev_step[self.HIDDEN]
        
        self.prev_step[self.HIDDEN] = step_sizes
        self.prev_dE_vji = dE_vji

        # update hidden-output weights: w_jk
        dE_wkj = np.array([[(err_o[k]*actv_h[j]) for k in range(self.K)] for j in range(self.J+1)])
        step_sizes = -self.eta*dE_wkj

        # update momentum
        new_alpha = self._quickprop(self.prev_dE_vji, dE_vji)
        if new_alpha is not None:
            self.alpha[self.HIDDEN] = new_alpha

        self.W[self.OUTPUT] += step_sizes + self.alpha[self.OUTPUT]*self.prev_step[self.OUTPUT]

        self.prev_step[self.OUTPUT] = step_sizes
        self.prev_dE_wkj = dE_wkj


    def _quickprop(self, prev_step, curr_step):
        #alpha_jk(t) =           dE /dw_kj(t)
        #                -----------------------------
        #                 dE/dw_kj(t-1) - dE/dw_kj(t)
        eps = 1e-10
        num = prev_step - curr_step
        r, c = prev_step.shape
        update = np.zeros((r,c))
        for x in range(r):
            for y in range(c):
                if num[x,y] > eps:
                    update[x,y] = curr_step[x,y]/num[x,y]

        return update


    def _running_stats(self, err, t, m, std):
        m_prev = m
        m = m + (err - m)/t
        std = std + (err - m)*(err - m_prev)
        return (m, std)


    def _val_error(self, X, y):
        N, d = X.shape
        sse = 0
        for i in range(N):
            label = y[i,:]
            h_actv, o_actv = self._forward_pass(X[i,:])
            if self.clf == True and self.K > 1:
                o_actv = (1/np.sum(o_actv)) * o_actv
            sse += np.sum([(label[k] - o_actv[k])**2 for k in range(self.K)])
        return sse/(2*N)       