import amplitf.interface as atfi
import amplitf.likelihood as atfl

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.toymc as tft
import tfa.neural_nets as tfn

import tensorflow as tf
import numpy as np

from DistributionModel import observables_data, observables_titles, observables_phase_space

atfi.set_seed(1)

norm_size = 1000000    # size of the normalisation sample (random uniform)
initfile = "eff_train_2d.npy"           # file with the trained parameters of the NN

ann = np.load(initfile, allow_pickle=True)       # Load NN parameters 

# Initialise NN weights and biases from the loaded file
(scale, ranges) = ann[:2]
(weights, biases) = tfn.init_fixed_weights_biases(ann[2:])

ndim = observables_phase_space.dimensionality()
observables_bounds = observables_phase_space.bounds()

# Density model as a multilayer perceptron
def model(x) : 
  return scale*tfn.multilayer_perceptron(x, ranges, weights, biases)

# Generate the sample corresponding to fit result and save it to ROOT file
fit_sample = tft.run_toymc(model, observables_phase_space, 1000000, 1e-10, chunk = 1000000)
tfr.write_tuple("eff_fit_result_2d.root", fit_sample, observables_data)

# Calculate PDF density on a 50x50 grid for chi2 evaluation
norm_sample = observables_phase_space.rectangular_grid_sample( ( 50, 50 ) )
fit = model(norm_sample).numpy().reshape( (50, 50) )

# Read test sample for chi2 evaluation
toy_sig = tfr.read_tuple("test_tuple.root", branches = ["mprime", "thetaprime"])
hist = np.histogram2d(toy_sig[:,0], toy_sig[:,1], bins=(50, 50), range=observables_bounds)

# Normalise fit result 
fit = fit/atfl.integral(fit)*atfl.integral(hist[0])

# Chi2
chi2 = np.sum((fit-hist[0])**2/fit)

print(fit)
print(hist[0])
print(f"Chi2={chi2}")
