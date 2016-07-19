# -*- coding: utf-8 -*-
"""
updates.py
~~~~~~~~~~
.. topic:: Contents

  The update module regroupse the update functions used for the beta NMF"""

import theano.tensor as T
import theano
from theano.ifelse import ifelse


def beta_H(X, W, H, beta):
    """Update activation with beta divergence

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
    H : Theano tensor
        Updated version of the activations
    """
    up = ifelse(
      T.eq(beta, 2),
      (T.dot(X, W)) / (T.dot(T.dot(H, W.T), W)),
      (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X), W)) /
      (T.dot(T.power(T.dot(H, W.T), (beta-1)), W)))
    return T.mul(H, up)


def beta_W(X, W, H, beta):
    """Update bases with beta divergence

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
    W : Theano tensor
        Updated version of the bases
    """
    up = ifelse(
      T.eq(beta, 2),
      (T.dot(X.T, H)) / (T.dot(T.dot(H, W.T).T, H)),
      (T.dot(T.mul(T.power(T.dot(H, W.T), (beta - 2)), X).T, H)) /
      (T.dot(T.power(T.dot(H, W.T), (beta-1)).T, H)))
    return T.mul(W, up)
