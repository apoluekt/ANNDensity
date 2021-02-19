import array
import sys
import math
import numpy as np
import tensorflow as tf

import amplitf.interface as atfi
import amplitf.likelihood as atfl
import tfa.rootio as tfr
import tfa.neural_nets as tfn
import tfa.toymc as tft

from DistributionModel import parameters_list, exp_phase_space, observables_data, observables_titles, observables_phase_space, sqdlz_phsp, m_phsp, md

initfile = "train_3d.npy"           # file with the trained parameters of the NN

init_w = np.load(initfile, allow_pickle = True)       # Load NN parameters 

# Initialise NN weights and biases from the loaded file
scale, ranges = init_w[:2]
weights, biases = tfn.init_fixed_weights_biases(init_w[2:])

ndim = exp_phase_space.dimensionality()
observables_bounds = exp_phase_space.bounds()

# Density model as a multilayer perceptron
def model(x) : 
  return scale*tfn.multilayer_perceptron(x, ranges, weights, biases)

fit_sample = tft.run_toymc(model, observables_phase_space, 8000000, 1e-20, chunk = 1000000)

tfr.write_tuple("fit_result_3d.root", fit_sample, observables_data)

norm_sample = sqdlz_phsp.rectangular_grid_sample( ( 50, 50 ) ).numpy()
norm_sample = np.concatenate([norm_sample, np.array(2500*[md])[:,np.newaxis]], axis = 1)
fit = model(norm_sample).numpy().reshape( (50, 50) )

toy_sig = tfr.read_tuple_filtered("test_tuple.root", branches = ["mprime", "thetaprime"], 
                                  sel_branches = ["md"], selection = "abs(md-1.97)<0.05")

hist = np.histogram2d(toy_sig[:,0], toy_sig[:,1], bins=(50, 50), range=observables_bounds[:2])
fit = fit/atfl.integral(fit)*atfl.integral(hist[0])

chi2 = np.sum((fit-hist[0])**2/fit)

print(fit)
print(hist[0])
print(f"Chi2={chi2}")
