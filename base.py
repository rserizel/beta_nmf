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
base.py
~~~~~~~
.. topic:: Contents

    The base module includes the basic functions such as
    beta-divergence, nonnegative random matrices generator or load_data."""

import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from sklearn import preprocessing
import h5py


def beta_div(X, W, H, beta):
    """Compute beta divergence D(X|WH)

    Parameters
    ----------
    X : Theano tensor
        data
    W : Theano tensor
        Bases
    H : Theano tensor
        activation matrix
    beta : Theano scalar


    Returns
    -------
    div : Theano scalar
        beta divergence D(X|WH)"""
    div = ifelse(T.eq(beta, 2),
                 T.sum(1. / 2 * T.power(X - T.dot(H, W), 2)),
                 ifelse(T.eq(beta, 0),
                        T.sum(X / T.dot(H, W) - T.log(X / T.dot(H, W)) - 1),
                        ifelse(T.eq(beta, 1),
                               T.sum(T.mul(X, (T.log(X) -
                                           T.log(T.dot(H, W)))) +
                                     T.dot(H, W) - X),
                               T.sum(1. / (beta * (beta - 1.)) *
                                     (T.power(X, beta) +
                                      (beta - 1.) *
                                      T.power(T.dot(H, W), beta) -
                                      beta * T.power(T.mul(X, T.dot(H, W)),
                                                     (beta - 1)))))))
    return div


def load_data(f_name, scale=True, rnd=True):
    """Get data from H5FS file.

    Parameters
    ----------
    f_name : String
        file name
    scale : Boolean (default True)
        scale data to unit variance (scikit-learn function)
    rnd : Boolean (default True)
        randomize the data along time axis


    Returns
    -------
    data : Dictionnary
        dictionary containing the data

        x_train: numpy array

            train data matrix """

    train_df = h5py.File(f_name, 'r')
    x_train = train_df['x_train'][:]
    train_df.close()
    if scale:
        print "scaling..."
        x_train = preprocessing.scale(x_train, with_mean=False)
    print "Total dataset size:"
    print "n train samples: %d" % x_train.shape[0]
    print "n features: %d" % x_train.shape[1]

    if rnd:
        print "Radomizing..."
        np.random.shuffle(x_train)

    data = dict(
        x_train=x_train,
    )
    return data


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
