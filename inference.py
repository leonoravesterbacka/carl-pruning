import os
import sys
import optparse
from network import RatioEstimator
from network.utils.loading import Loader

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='QSFUP', help='variation to derive weights for. default QSF down to QSF up')
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=0, help='specify the number of events do do the training on, default None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/user/m/mvesterb/pmg/', help='path to where the data is stored')

(opts, args) = parser.parse_args()
sample = 'dilepton'
var    = opts.variation
n      = opts.nentries
p      = opts.datapath
if os.path.exists('data/'+ sample +'/'+ var +'/X_train_'+str(n)+'.npy'):
    print("Doing evaluation of model trained with datasets: ",sample, ", generator variation: ", var, " with ", n, " events." )
else:
    print("No datasets available for evaluation of model trained with ",sample, ", generator variation: ", var, " with ", n, " events." )
    print("ABORTING")
    sys.exit()
    
loading = Loader()
carl = RatioEstimator()
carl.load('models/'+ sample + '/' + var + '_carl_'+str(n))
evaluate = ['train', 'val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x='data/'+ sample + '/' + var + '/X0_'+i+'_'+str(n)+'.npy')
    w = 1./r_hat
    loading.load_result(x0='data/'+ sample + '/' + var + '/X0_'+i+'_'+str(n)+'.npy',     
                        x1='data/'+ sample + '/' + var + '/X1_'+i+'_'+str(n)+'.npy',
                        weights=w, 
                        label = i,
                        do = sample,
                        var = var,
                        plot = True,
                        n = n,
                        path = p,
    )
carl.evaluate_performance(x='data/'+ sample + '/' + var + '/X_val_'+str(n)+'.npy',y='data/' + sample + '/' + var +'/y_val_'+str(n)+'.npy')
