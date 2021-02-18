#import sys, os, math
#sys.path.append("../../TFA2")

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus : 
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)] )

import amplitf.interface as atfi

atfi.set_single_precision()

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

from DistributionModel import parameters_list, observables_phase_space, observables_toys, observables_titles

variables = observables_toys + [ i[0] for i in parameters_list ]
titles = observables_titles + [ i[1] for i in parameters_list ]
parameters_bounds = [ i[2] for i in parameters_list ]

parameters_phase_space = RectangularPhaseSpace( parameters_bounds )
bounds = observables_phase_space.bounds() + parameters_bounds

phsp = CombinedPhaseSpace( observables_phase_space, parameters_phase_space )

data = atfi.const(tfr.read_tuple("toy_tuple.root", variables))

print(data)

data = phsp.filter(data)

print(data)

# Open matplotlib window
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=4, usetex=False)
fig, ax = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(12, 9))

#atfi.set_seed(4)

# Perform density estimation using ANNs
tfn.estimate_density(
    phsp,
    data,
    ranges=bounds,  # list of ranges for observables
    labels=titles,
    learning_rate=0.0002,  # Tunable meta-parameter
    weight_penalty=1.,  # Tunable meta-parameter (larger for smoother distribution)
    n_hidden=[32, 64, 32, 8],  # Structure of hidden layers (4 layers, 64, 64, 32 and 8 neurons)
    training_epochs=50000,  # Number of training iterations (epochs)
    norm_size=4000000,  # Size of the normalisation sample
    print_step=50,  # Print cost function every 50 epochs
    display_step=500,  # Display status of training every 500 epochs
    initfile="init.npy",  # Init file (e.g. if continuing previously interrupted training)
    outfile="train",  # Name prefix for output files (.pdf, .npy, .txt)
    seed=4,  # Random seed
    fig=fig,  # matplotlib window references
    axes=ax,
)
