import os
import sys
import logging
import optparse
import tarfile
from network import RatioEstimator

def test_training():
    x ='data/dilepton/QSFUP/X_train_10.npy'
    y ='data/dilepton/QSFUP/y_train_10.npy'
    x0='data/dilepton/QSFUP/X0_train_10.npy'
    x1='data/dilepton/QSFUP/X1_train_10.npy'
    print("Loaded existing datasets ")
    
    estimator = RatioEstimator(
        n_hidden=(10,10),
        activation="relu"
    )
    estimator.train(
        method='carl',
        batch_size = 1024,
        n_epochs = 1,
        x=x,
        y=y,
        x0=x0,
        x1=x1,
        scale_inputs = True,
        prune_network = False,
    #    early_stopping = True,
    #    early_stopping_patience = 10
    )
    assert True
