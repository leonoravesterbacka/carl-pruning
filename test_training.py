import os
import sys
import logging
import optparse
import torch
import tarfile
from network import RatioEstimator
from network import Loader


parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='QSFUP', help='variation to derive weights for. default QSF down to QSF up')
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=10, help='specify the number of events do do the training on, default None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/user/m/mvesterb/pmg/', help='path to where the data is stored')
(opts, args) = parser.parse_args()
sample  = 'dilepton'
var     = opts.variation
n       = opts.nentries
p       = opts.datapath
loading = Loader()
#if os.path.exists('tests/data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'):
x='tests/data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'
y='tests/data/'+ sample +'/'+ var +'/y_train_'+str(n)+'.npy'
x0='tests/data/'+ sample +'/'+ var +'/X0_train_'+str(n)+'.npy'
x1='tests/data/'+ sample +'/'+ var +'/X1_train_'+str(n)+'.npy'
print("Loaded existing datasets ")

estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="relu"
)
estimator.train(
    method='carl',
    batch_size = 10,
    n_epochs = 1,
    x=x,
    y=y,
    x0=x0, 
    x1=x1,
    scale_inputs = True,
    prune_network = False,
)
