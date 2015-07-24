# -*- coding: utf-8 -*-
"""
Copyright © 2015 Telecom ParisTech, TSI
Auteur(s) : Romain Serizel
the beta_ntf module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU LesserGeneral Public License
along with this program. If not, see <http://www.gnu.org/licenses/>."""

import time
import numpy as np
import theano
import theano.tensor as T

FILE_NAME = ("short_set_cqt.h5")


def beta_div(X, W, H, beta):
    """Compute betat divergence"""
    if beta == 0:
        return T.sum(X / T.dot(H, W) - T.log(X / T.dot(H, W)) - 1)
    if beta == 1:
        return T.sum(T.mul(X, (T.log(X) - T.log(T.dot(H, W)))) + T.dot(H, W) - X)
    return T.sum(1. / (beta * (beta - 1.))
                 * (T.power(X, beta)
                    + (beta - 1.) * T.power(T.dot(H, W), beta)
                    - beta * T.power(T.mul(X, T.dot(H, W)), (beta - 1))))


def load_data(f_name, scale=True, rnd=True):
    """Get data with labels, split into training, validation and test set."""
    from sklearn import preprocessing
    import h5py
    train_df = h5py.File(f_name, 'r')
    x_train = train_df['x_train'][0:15000]
    train_df.close()
    #X = numpy.asarray(mnist.data, dtype='float32')
    if scale:
        print "scaling..."
        x_train = preprocessing.scale(x_train, with_mean=False)
    print "Total dataset size:"
    print "n train samples: %d" % x_train.shape[0]
    print "n features: %d" % x_train.shape[1]

    if rnd:
        print "Radomizing..."
        np.random.shuffle(x_train)

    return dict(
        x_train=x_train,
    )



class BetaNMF:
    """BetaNMF class

    Performs nonnegative matrix factorization with Theano.

    Parameters
    ----------
    data_shape : the shape of the data to approximate
        tuple composed of integers

    n_components : the number of latent components for the NMF model
        positive integer

    beta : the beta-divergence to consider
        Arbitrary float. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : number of iterations
        Positive integer

    Attributes
    ----------
    factors_: list of arrays
        The estimated factors
    """

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=0, n_iter=50,
                 fixed_factors=[], verbose=0):
        self.data_shape = data_shape
        self.n_components = n_components
        self.beta = float(beta)
        self.n_iter = n_iter
        self.verbose = verbose
        self.fixed_factors = fixed_factors
        self.factors_ = [nnrandn((dim, self.n_components)) for dim in data_shape]

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

        beta = self.beta
        gamma = 1
        if beta < 1:
            gamma = 1/(2-beta)
        if beta > 2:
            gamma = 1/(beta-1)
        tX = theano.shared(X.astype(theano.config.floatX), name="X")
        tH = theano.shared(self.factors_[0].astype(theano.config.floatX), name="H")
        tW = theano.shared(self.factors_[1].astype(theano.config.floatX), name="W")

        trainW = theano.function(inputs=[],
                                 outputs=[],
                                 updates={tW:tW*(T.power(((T.dot(T.mul(T.power(T.dot(tH, tW.T),
                                                                               (beta - 2)), tX).T,
                                                                 tH))
                                                          /(T.dot(T.power(T.dot(tH, tW.T),
                                                                          (beta-1)).T, tH))),
                                                         gamma))},
                                 name="trainH")
        trainH = theano.function(inputs=[],
                                 outputs=[],
                                 updates={tH:tH*(T.power(((T.dot(T.mul(T.power(T.dot(tH, tW.T),
                                                                               (beta - 2)), tX),
                                                                 tW))
                                                          /(T.dot(T.power(T.dot(tH, tW.T),
                                                                          (beta-1)), tW))),
                                                         gamma))},
                                 name="trainH")

        print 'Fitting NMF model with %d iterations....' % self.n_iter

        # main loop
        for it in range(self.n_iter):
            if self.verbose > 0:
                if it == 0:
                    if 'tick' not in locals():
                        tick = time.time()
                    if 0 not in self.fixed_factors:
                        self.factors_[0] = tH.get_value()
                    if 1 not in self.fixed_factors:
                        self.factors_[1] = tW .get_value()
                    print ('NMF model, iteration %d / %d, duration=%.1fms, cost=%f'
                           % (it, self.n_iter, (time.time() - tick) * 1000,
                              self.score(X)))
            if 1 not in self.fixed_factors:
                trainW()
            if 0 not in self.fixed_factors:
                trainH()

            if self.verbose > 0:
                if (it+1)%self.verbose == 0:
                    if 'tick' not in locals():
                        tick = time.time()
                    if 0 not in self.fixed_factors:
                        self.factors_[0] = tH.get_value()
                    if 1 not in self.fixed_factors:
                        self.factors_[1] = tW .get_value()
                    print ('NMF model, iteration %d / %d, duration=%.1fms, cost=%f'
                           % ((it+1), self.n_iter, (time.time() - tick) * 1000,
                              self.score(X)))
                    tick = time.time()

        self.factors_[0] = tH.get_value()
        self.factors_[1] = tW.get_value()
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
        tX = theano.shared(X.astype(theano.config.floatX), name="X")
        tH = theano.shared(self.factors_[0].astype(theano.config.floatX), name="H")
        tW = theano.shared(self.factors_[1].astype(theano.config.floatX), name="W")

        div = theano.function(inputs=[],
                              outputs=beta_div(tX, tW.T, tH, self.beta),
                              name="div")
        return div()




def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


if __name__ == '__main__':


    dataset = load_data(f_name=FILE_NAME)
    x = dataset['x_train']

    beta_nmf = BetaNMF(x.shape, n_components=100, beta=2, n_iter=1000,
                       verbose=20)

    # Fit the model
    tic = time.time()
    beta_nmf.fit(x)
    tic = time.time() - tic
    print 'NMF model trained, duration=%.1fs'% tic
    print 'Resulting score', beta_nmf.score(x)

    tic = time.time()
    beta_nmf1 = BetaNMF(x[0:1500].shape, n_components=100, beta=2, n_iter=100,
                        verbose=20)
    beta_nmf1.fixed_factors = [1]
    beta_nmf1.factors_[1] = beta_nmf.factors_[1]            
    beta_nmf1.fit(x[0:1500])
    tic = time.time() - tic
    print 'NMF model trained, duration=%.1fs'% tic
    print 'Resulting score', beta_nmf.score(x)
    
