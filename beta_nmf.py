# -*- coding: utf-8 -*-
# Copyright © 2015 Telecom ParisTech, TSI
# Auteur(s) : Romain Serizel
# the beta_nmf module for GPGPU is free software: you can redistribute it
# or modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU LesserGeneral Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
beta_nmf.py
~~~~~~~~~~~

The beta_nmf module includes the beta_nmf class,
fit function and theano functions to compute updates and cost."""

import time
import numpy as np
import theano
import theano.tensor as T
from base import beta_div
from base import nnrandn
from base import load_data

FILE_NAME = ("../short_set_cqt.h5")


class BetaNMF:
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
                 fixed_factors=None, verbose=0):
        self.data_shape = data_shape
        self.n_components = n_components
        self.beta = float(beta)
        self.n_iter = n_iter
        self.verbose = verbose
        if fixed_factors is None:
            fixed_factors = []
        self.fixed_factors = fixed_factors
        self.factors = [nnrandn((dim, self.n_components))
                        for dim in data_shape]

    def fit(self, X):
        """Learns NMF model

        Parameters
        ----------
        X : ndarray with nonnegative entries
            The input array
        W : ndarray
            Optional ndarray that can be broadcasted with X and
            gives weights to apply on the cost function
        """

        tbeta = theano.shared(self.beta, name="beta")
        tX = theano.shared(X.astype(theano.config.floatX), name="X")
        tH = theano.shared(self.factors[0].astype(theano.config.floatX),
                           name="H")
        tW = theano.shared(self.factors[1].astype(theano.config.floatX),
                           name="W")

        upW = tW*((T.dot(T.mul(T.power(T.dot(tH, tW.T), (tbeta - 2)),
                               tX).T, tH)) /
                  (T.dot(T.power(T.dot(tH, tW.T), (tbeta-1)).T, tH)))
        upH = tH*((T.dot(T.mul(T.power(T.dot(tH, tW.T), (tbeta - 2)),
                               tX), tW)) /
                  (T.dot(T.power(T.dot(tH, tW.T), (tbeta-1)), tW)))
        trainW = theano.function(inputs=[],
                                 outputs=[],
                                 updates={tW: upW},
                                 name="trainH")
        trainH = theano.function(inputs=[],
                                 outputs=[],
                                 updates={tH: upH},
                                 name="trainH")

        print 'Fitting NMF model with %d iterations....' % self.n_iter

        # main loop
        for it in range(self.n_iter):
            if self.verbose > 0:
                if it == 0:
                    if 'tick' not in locals():
                        tick = time.time()
                    if 0 not in self.fixed_factors:
                        self.factors[0] = tH.get_value()
                    if 1 not in self.fixed_factors:
                        self.factors[1] = tW .get_value()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it, self.n_iter, (time.time() - tick) * 1000,
                              self.score(X)))
            if 1 not in self.fixed_factors:
                trainW()
            if 0 not in self.fixed_factors:
                trainH()

            if self.verbose > 0:
                if (it+1) % self.verbose == 0:
                    if 'tick' not in locals():
                        tick = time.time()
                    if 0 not in self.fixed_factors:
                        self.factors[0] = tH.get_value()
                    if 1 not in self.fixed_factors:
                        self.factors[1] = tW .get_value()
                    print ('Iteration %d / %d, duration=%.1fms, cost=%f'
                           % ((it+1), self.n_iter, (time.time() - tick) * 1000,
                              self.score(X)))
                    tick = time.time()

        self.factors[0] = tH.get_value()
        self.factors[1] = tW.get_value()
        print 'Done.'
        return self

    def score(self, X):
        """Computes the total beta-divergence between the current model and X

        Parameters
        ----------
        X : array
            The input data

        Returns
        -------
        out : float
            The beta-divergence
        """
        tbeta = theano.shared(self.beta, name="beta")
        tX = theano.shared(X.astype(theano.config.floatX), name="X")
        tH = theano.shared(self.factors[0].astype(theano.config.floatX),
                           name="H")
        tW = theano.shared(self.factors[1].astype(theano.config.floatX),
                           name="W")

        div = theano.function(inputs=[],
                              outputs=beta_div(tX, tW.T, tH, tbeta),
                              name="div")
        return div()

if __name__ == '__main__':
    """Example script"""
    dataset = load_data(f_name=FILE_NAME)
    x = dataset['x_train']

    beta_nmf = BetaNMF(x.shape,
                       n_components=100,
                       beta=2,
                       n_iter=1000,
                       verbose=20)

    # Fit the model
    tic = time.time()
    beta_nmf.fit(x)
    tic = time.time() - tic
    print 'NMF model trained, duration=%.1fs' % tic
    print 'Resulting score', beta_nmf.score(x)

    # Project data on H while keeping the bases W fixed
    tic = time.time()
    beta_nmf1 = BetaNMF(x.shape,
                        n_components=100,
                        beta=2,
                        fixed_factors=[1],
                        n_iter=100,
                        verbose=20)
    beta_nmf1.factors[1] = beta_nmf.factors[1]
    beta_nmf1.fit(x[0:1500])
    tic = time.time() - tic
    print 'NMF model trained, duration=%.1fs' % tic
    print 'Resulting score', beta_nmf.score(x)
