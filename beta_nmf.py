# -*- coding: utf-8 -*-
"""
beta\_nmf.py
~~~~~~~~~~~

.. topic:: Contents

  The beta_nmf module includes the beta\_nmf class,
  fit function and theano functions to compute updates and cost."""

import time
import numpy as np
import theano
import base
import updates
import costs


class BetaNMF(object):
    """BetaNMF class

    Performs nonnegative matrix factorization with Theano.

    Parameters
    ----------
    data_shape : tuple composed of integers
        the shape of the data to approximate

    n_components : positive integer (default 50)
        the number of latent components for the NMF model

    beta : arbitrary float (default 2)
        the beta-divergence to consider, particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : Positive integer (default 100)
        number of iterations

    fixed_factors : array (default Null)
        list of factors that are not updated
            e.g. fixed_factors = [0] -> H is not updated

            fixed_factors = [1] -> W is not updated

    verbose : Integer
        the frequence at which the score should be computed and displayed
        (number of iterations between each computation)


    Attributes
    ----------
    factors : list of arrays

        The estimated factors (factors[0] = H)"""

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=2, n_iter=100,
                 fixed_factors=None, verbose=0, cold_start=True):
        self.data_shape = data_shape
        self.n_components = n_components
        self.n_components = np.asarray(n_components, dtype='int32')
        self.beta = theano.shared(np.asarray(beta, theano.config.floatX),
                                  name="beta")
        self.verbose = verbose
        self.n_iter = n_iter
        self.scores = []
        self.cold_start = cold_start
        if fixed_factors is None:
            fixed_factors = []
        self.fixed_factors = fixed_factors
        fact_ = [base.nnrandn((dim, self.n_components)) for dim in data_shape]
        self.w = theano.shared(fact_[1].astype(theano.config.floatX),
                               name="W", borrow=True, allow_downcast=True)
        self.h = theano.shared(fact_[0].astype(theano.config.floatX),
                               name="H", borrow=True, allow_downcast=True)
        self.factors = [self.h, self.w]
        self.x = theano.shared(
          np.zeros((data_shape)).astype(theano.config.floatX), name="X")
        self.get_updates_functions()
        self.get_div_function()

    def check_shape(self):
        """Check that all the matrix have consistent shapes"""
        self.data_shape = self.x.get_value().shape
        dim = long(self.n_components)
        if self.w.get_value().shape != (self.data_shape[1], dim):
            print "Inconsistent data for W, expected {1}, found {0}".format(
                self.w.get_value().shape,
                (self.data_shape[1], dim))
            raise SystemExit
        if self.h.get_value().shape != (self.data_shape[0], dim):
            print "Inconsistent shape for H, expected {1}, found {0}".format(
                self.h.get_value().shape,
                (self.data_shape[0], dim))
            raise SystemExit

    def fit(self, data, warm_start=False):
        """Learns NMF model

        Parameters
        ----------
        X : ndarray with nonnegative entries
            The input array
        warm_start : Boolean (default False)
            start from new values
        """

        if not warm_start:
            self.set_factors(data, self.fixed_factors)
        self.x.set_value(data.astype(theano.config.floatX))
        self.check_shape()

        print 'Fitting NMF model with %d iterations....' % self.n_iter

        # main loop
        for it in range(self.n_iter):
            if 'tick' not in locals():
                tick = time.time()
            if self.verbose > 0:
                if it == 0:
                    score = self.score()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it, self.n_iter, (time.time() - tick) * 1000,
                              score))
            if 1 not in self.fixed_factors:
                self.train_w()
            if 0 not in self.fixed_factors:
                self.train_h()
            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    score = self.score()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it+1, self.n_iter, (time.time() - tick) * 1000,
                              score))
                    tick = time.time()
        print 'Done.'

    def get_div_function(self):
        """Compile the theano-based divergence function"""
        self.div = theano.function(inputs=[],
                                   outputs=costs.beta_div(self.x,
                                                          self.w.T,
                                                          self.h,
                                                          self.beta),
                                   name="div",
                                   allow_input_downcast=True)

    def get_updates_functions(self):
        """Compile the theano based update functions"""
        print "Standard rules for beta-divergence"
        h_update = updates.beta_H(self.x, self.w, self.h, self.beta)
        w_update = updates.beta_W(self.x, self.w, self.h, self.beta)
        self.train_h = theano.function(inputs=[],
                                       outputs=[],
                                       updates={self.h: h_update},
                                       name="trainH",
                                       allow_input_downcast=True)
        self.train_w = theano.function(inputs=[],
                                       outputs=[],
                                       updates={self.w: w_update},
                                       name="trainW",
                                       allow_input_downcast=True)

    def score(self):
        """Compute factorisation score

        Returns
        -------
        out : Float
            factorisation score"""
        return self.div()

    def set_factors(self, X, fixed_factors=None):
        """reset factors

        Parameters
        ----------
        X : array
            The input data
        fixed_factors : array  (default Null)
            list of factors that are not updated
                e.g. fixed_factors = [0] -> H is not updated

                fixed_factors = [1] -> W is not updated
        """
        self.data_shape = X.shape
        fact_ = [base.nnrandn((dim, self.n_components))
                 for dim in self.data_shape]
        if fixed_factors is None:
            fixed_factors = []
        if 1 not in fixed_factors:
            self.w.set_value(fact_[1])
        if 0 not in fixed_factors:
            self.h.set_value(fact_[0])
        self.factors = [self.h, self.w]

    def transform(self, X, warm_start=False):
        """Project data X on the basis W

        Parameters
        ----------
        X : array
            The input data
        warm_start : Boolean (default False)
            start from previous values

        Returns
        -------
        H : array
            Activations
        """
        self.fixed_factors = [1]
        if not warm_start:
            print "cold start"
            self.set_factors(X, self.fixed_factors)
        self.fit(X, warm_start=True)
        return self.h.get_value()
