import tensorflow as tf

# Limit the maximum VRAM used to 12 Gb
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus : 
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)] )

import amplitf.interface as atfi

# Training is performed in single FP precision for speed
atfi.set_single_precision()

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

# Import model-spacific information (phase spaces, observables names etc.)
from DistributionModel import parameters_list, observables_phase_space, observables_toys, observables_titles, exp_phase_space

variables = observables_toys       # List of names of variables 
titles = observables_titles        # Titles for plots
bounds = exp_phase_space.bounds()

# Read training data
data = atfi.const(tfr.read_tuple("test_tuple.root", variables))
data = exp_phase_space.filter(data)

print(data)

# Open matplotlib window
import matplotlib.pyplot as plt
tfp.set_lhcb_style(size=8, usetex=False)
fig, ax = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(10, 7))

mass_penalty_fraction = 10.  # Penalty fraction for neuron weights corresponding to D mass variable
weight_penalty = 1.          # General L2 weight penalty parameter

def regularisation(weights) : 
    """
      Custom regularisation function that is a standard L2, but additionally penalises 
      the weight in the 1s layer corresponding to "D mass" variable, 
      to make Dalitz features smoothly dependent on D mass
    """
    penalty = 0.
    for w in weights : 
        penalty += tf.reduce_sum(tf.square(w))
    penalty += mass_penalty_fraction*tf.reduce_sum(tf.square(weights[0][2,:]))
    return weight_penalty*penalty

tfn.estimate_density(
    exp_phase_space, # Phase space
    data,            # Data to train with
    ranges=bounds,   # list of ranges for observables
    labels=titles,   # Ovservables' titles
    learning_rate=0.0002,  # Tunable meta-parameter
    n_hidden=[32, 64, 32, 8],  # Structure of hidden layers (4 layers, 64, 64, 32 and 8 neurons)
    training_epochs=50000,  # Number of training iterations (epochs)
    norm_size=4000000,   # Size of the normalisation sample
    print_step=50,       # Print cost function every 50 epochs
    display_step=500,    # Display status of training every 500 epochs
    initfile="init_3d.npy",  # Init file (e.g. if continuing previously interrupted training)
    outfile="train_3d",  # Name prefix for output files (.pdf, .npy, .txt)
    seed=4,   # Random seed
    fig=fig,  # matplotlib window references
    axes=ax,
)
