import amplitf.interface as atfi
import amplitf.likelihood as atfl

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.plotting as tfp
import tfa.optimisation as tfo
import tfa.rootio as tfr
import tfa.toymc as tft
import tfa.neural_nets as tfn

import tensorflow as tf
import numpy as np
import sys

import matplotlib.pyplot as plt

from DistributionModel import parameters_list, exp_phase_space, observables_data, observables_titles, observables_phase_space, sqdlz_phsp, m_phsp, md

print_step = 50        # print statistics every 50 epochs
norm_size = 4000000    # size of the normalisation sample (random uniform)
path = "./"
initfile = "train.npy"           # file with the trained parameters of the NN
calibfile = "test_tuple.root"    # Sample to fit
outfile = "result"               # Prefix for output files (text and pdf)
seed = 1                         # initial random seed
selection = ""

if len(sys.argv)>1 : calibfile = sys.argv[1]  # file to fit is the first command line parameter
if len(sys.argv)>2 : outfile = sys.argv[2]    # output prefix is the second parameter
if len(sys.argv)>3 : seed = int(sys.argv[3])  # optionally, initial seed

ann = np.load(initfile, allow_pickle = True)       # Load NN parameters 

data_sample = tfr.read_tuple(path + calibfile, branches = observables_data)

# Initialise NN weights and biases from the loaded file
(scale, ranges) = ann[:2]
(weights, biases) = tfn.init_fixed_weights_biases(ann[2:])

ndim = exp_phase_space.dimensionality()
observables_bounds = exp_phase_space.bounds()

# Density model as a multilayer perceptron
def model(x, pars) : 
  # Constant vectors of fit parameters (the same for every data point)
  vec = tf.reshape( tf.concat( [ tf.constant(ndim*[0.], dtype = atfi.fptype() ), pars ], axis = 0 ), [ 1, ndim + len(pars) ] )
  # Input tensor for MLP, 5 first variables are data, the rest are constant optimisable parameters 
  x2 = tf.pad( x, [[0, 0], [0, len(pars)]], 'CONSTANT') + vec
  return scale*tfn.multilayer_perceptron(x2, ranges, weights, biases)

# Initialise random seeds
atfi.set_seed(seed)

# Declare fit parameters 
pars = [ tfo.FitParameter(par[0], (par[2][0]+par[2][1])/2., par[2][0], par[2][1]) for par in parameters_list ]

# Unbinned negative log likelihood
@atfi.function
def nll(pars) : 
  parslist = [ pars[i[0]] for i in parameters_list ]
  return atfl.unbinned_nll( model(data_sample, parslist), atfl.integral( model(norm_sample, parslist) ) )

# Normalisation sample is a uniform random sample in 5D phase space
norm_sample = exp_phase_space.uniform_sample( norm_size )

# Data sample, run through phase space filter just in case
data_sample = exp_phase_space.filter(data_sample)

bins = ndim*[50]

tfp.set_lhcb_style(size=9, usetex=False)
fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(8, 6))

# Initialise multidimensional density display object
display = tfp.MultidimDisplay(data_sample, norm_sample, bins, observables_bounds, observables_titles, fig, ax)

print("Normalisation sample size = {len(norm_sample)}")
print(norm_sample)
print("Data sample size = {len(data_sample)}")
print(data_sample)

# Run minimisation 10 times, choose the best NLL value
best_nll = 1e10
for i in range(0, 3) : 
    for p in pars :
        p.update(np.random.uniform(p.lower_limit, p.upper_limit))
    result = tfo.run_minuit(nll, pars)
    print(result)
    parslist = [ result["params"][i[0]][0] for i in parameters_list ]
    if result['loglh'] < best_nll :   # If we got the best NLL so far
        best_nll = result['loglh']
        best_result = result
        norm_pdf = model(norm_sample, parslist)  # Calculate PDF
        display.draw(norm_pdf)         # Draw PDF
        plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
        plt.draw()
        plt.pause(0.1)


print("Optimization Finished!")

parslist = [ best_result["params"][i[0]][0] for i in parameters_list ]
norm_pdf = model(norm_sample, parslist)  # Calculate PDF
display.draw(norm_pdf)

def fit_model(x) :
  return model(x, parslist)

fit_sample = tft.run_toymc(fit_model, observables_phase_space, 1000000, 1e-20, chunk = 1000000)
tfr.write_tuple("eff_fit_result.root", fit_sample, observables_data)

norm_sample_2 = sqdlz_phsp.rectangular_grid_sample( ( 50, 50 ) ).numpy()
norm_sample_2 = np.concatenate([norm_sample_2, np.array(2500*[md])[:,np.newaxis]], axis = 1)
fit = fit_model(norm_sample_2).numpy().reshape( (50, 50) )

toy_sig = tfr.read_tuple_filtered("test_tuple.root", branches = ["mprime", "thetaprime"], 
                                  sel_branches = ["md"], selection = "abs(md-1.97)<0.05")

hist = np.histogram2d(toy_sig[:,0], toy_sig[:,1], bins=(50, 50), range=observables_bounds[:2])
fit = fit/atfl.integral(fit)*atfl.integral(hist[0])

chi2 = np.sum((fit-hist[0])**2/fit)

print(fit)
print(hist)
print(norm_sample_2)
print(f"Chi2={chi2}")
