import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys
import os
#sys.path.insert(0, '/home/syazdani/caffe/python')
import caffe
#import cv2
import numpy as np
import shutil
import mlab


#**************Define hyper parameters**************************

def uniform_int(name, lower, upper):
    # `quniform` returns:
    # round(uniform(low, high) / q) * q
    return hp.quniform(name, lower, upper, q=1)

def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

def loguniform(name, lower, upper):
    # Do not forget to make a logarithm for the
    # lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)

parameter_space = {

    'base_lr': hp.loguniform('base_lr', 0.001, 0.5),
    'momentum': hp.loguniform('momentum', 0.01, 2),
    'dropout': hp.uniform('dropout', 0, 0.5),
    'weight_decay': hp.uniform('weight_decay', 0.0001, 0.0009),
    'batch_size': loguniform_int('batch_size', 16, 512)
}

tmp = sys.stdout
sys.stdout = open('caffe-output_cifar10.txt', 'wt')



#*****define the objective function********************

from pprint import pprint

def objective(parameters):
    #print("Parameters:")
    #pprint(parameters)
    #print()


    base_lr = parameters['base_lr']
    momentum = parameters['momentum']
    batch_size = int(parameters['batch_size'])
    weight_decay = parameters['weight_decay']
    proba = parameters['dropout']

    solver_options = {}
    solver_options["base_lr"] = base_lr
    solver_options["momentum"] = momentum
    solver_options["weight_decay"] = weight_decay
    mlab.update_solver("/home/sanand2/hypercaffe/caffe/examples/cifar10/cifar10_full_solver.prototxt", solver_options)
    solver_options = {}
    solver_options["batch_size"] = batch_size
    mlab.update_solver("/home/sanand2/hypercaffe/caffe/examples/cifar10/cifar10_full_solver.prototxt", solver_options)

    net = caffe.Net('/home/sanand2/hypercaffe/caffe/examples/cifar10/cifar10_full_train_test.prototxt', caffe.TRAIN)

    solver = caffe.get_solver('/home/sanand2/hypercaffe/caffe/examples/cifar10/cifar10_full_solver.prototxt')
    solver.solve()
    min = solver.net.blobs['loss'].data
    print min

    #loss_val = np.zeros(10)
    #min = 0
    #for it in range(10):
    #    solver.solve()
    #    loss_val[it] = solver.net.blobs['loss'].data
#	if min<loss_val[it] :
#	    min = loss_val[it]
        #print "loss value is " + str(loss_val[it])
#    print min
    #print net.blobs['mnist'].batch_size
    #min = 0
    #for i in range(10):
        #print str(loss_val[i]) + " "
        #sum += int(loss_val[i])
    #sum = sum /10
    #print "loss value is " + loss_val
    #os.system("cd /home/syazdani/caffe")
    #os.system("./examples/mnist/train_lenet.sh")
    #os.system("cd hyperopt")
    #sys.stdout.close()
    #sys.stdout = tmp
    return {
        'loss': min,
        'status': STATUS_OK
        }

import hyperopt
from functools import partial

trials = Trials()

tpe = partial(
    hyperopt.tpe.suggest,

    # Sample 1000 candidate and select candidate that
    # has highest Expected Improvement (EI)
    n_EI_candidates=1000,

    # Use 20% of best observations to estimate next
    # set of parameters
    gamma=0.2,

    # First 20 trials are going to be random
    n_startup_jobs=2,
)



#********* Define Minimization function***********
best = fmin(objective,
    space=parameter_space,
    algo=tpe,
    max_evals=5,
    trials=trials)

print (best)
