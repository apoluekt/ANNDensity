import amplitf.interface as atfi

atfi.set_single_precision()

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

from DistributionModel import observables_phase_space, observables_toys, observables_titles

bounds = observables_phase_space.bounds()

data = atfi.const(tfr.read_tuple("test_tuple.root", observables_toys))

# Open matplotlib window
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=9, usetex=False)
fig, ax = plt.subplots(nrows=len(observables_toys), ncols=len(observables_toys), figsize=(8, 6))

# Perform density estimation using ANNs
tfn.estimate_density(
    observables_phase_space,
    data,
    ranges=bounds,  # list of ranges for observables
    labels=observables_titles,
    learning_rate=0.001,  # Tunable meta-parameter

    #weight_penalty=0.02,  # Tunable meta-parameter (larger for smoother distribution)
    #n_hidden=[32, 64, 32, 8],  # Structure of hidden layers (2 layers, 32 and 8 neurons)
    weight_penalty=0.1,  # Tunable meta-parameter (larger for smoother distribution)
    n_hidden=[16, 32, 8],  # Structure of hidden layers (2 layers, 32 and 8 neurons)

    training_epochs=30000,  # Number of training iterations (epochs)
    norm_size=1000000,  # Size of the normalisation sample
    print_step=50,  # Print cost function every 50 epochs
    display_step=200,  # Display status of training every 500 epochs
    initfile="init_2d.npy",  # Init file (e.g. if continuing previously interrupted training)
    outfile="eff_train_2d",  # Name prefix for output files (.pdf, .npy, .txt)
    seed=2,  # Random seed
    fig=fig,  # matplotlib window references
    axes=ax,
)
