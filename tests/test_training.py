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
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=1, help='specify the number of events do do the training on, default None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/user/m/mvesterb/pmg/', help='path to where the data is stored')
(opts, args) = parser.parse_args()
sample  = 'dilepton'
var     = opts.variation
n       = opts.nentries
p       = opts.datapath
loading = Loader()
if os.path.exists(p+'/Sh_228_ttbar_'+sample+'_EnhMaxHTavrgTopPT_'+var+'.root'):
    print("Doing training of model with datasets: ",sample, ", generator variation: ", var, " with ", n, " events." )
else:
    print("Trying to do training with datasets: ",sample, ", generator variation: ", var, " with ", n, " events, but they don't exist in ", p )
    print("Try one of the following options:")
    print("Generator variation: -v QSFUP, QSFDOWN, CKKW20, CKKW50")
    print("ttbar sample       : -s dilepton, singleLepton, allHadronic")
    sys.exit()
if os.path.exists('data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'):
    x='data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'
    y='data/'+ sample +'/'+ var +'/y_train_'+str(n)+'.npy'
    x0='data/'+ sample +'/'+ var +'/X0_train_'+str(n)+'.npy'
    x1='data/'+ sample +'/'+ var +'/X1_train_'+str(n)+'.npy'
    print("Loaded existing datasets ")
    if torch.cuda.is_available():
        tar = tarfile.open("data_out.tar.gz", "w:gz")
        for name in ['data/'+ sample +'/'+ var +'/X0_train_'+str(n)+'.npy']:
            tar.add(name)
        tar.close()
else:
    x, y, x0, x1 = loading.loading(
        folder = './data/',
        plot = True,
        var = var,
        do = sample,
        randomize = False,
        save = True,
        correlation = True,
        preprocessing = True,
        nentries = n,
        path = p,
    )
    print("Loaded new datasets ")

estimator = RatioEstimator(
    n_hidden=(10,10,10),
    activation="relu"
)
estimator.train(
    method='carl',
    batch_size = 1024,
    n_epochs = 100,
    x=x,
    y=y,
    x0=x0, 
    x1=x1,
    scale_inputs = True,
    prune_network = True,
)
estimator.save('models/'+ sample +'/'+ var +'_carl_'+str(n), x=x, export_model = True)
