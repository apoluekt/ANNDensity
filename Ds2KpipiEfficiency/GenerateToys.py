import array
import sys
import math
import numpy as np

import tensorflow as tf

import amplitf.interface as atfi
import tfa.rootio as tfr

from DistributionModel import parameters_list, observables_toys, observables_phase_space, selection_with_random_cuts, true_cuts, random_array_size

import sys, os

def main() : 

  nev = 2000000
  outfile = "toy_tuple.root"

  atfi.set_seed(nev+1)

  chunk_size = 1000000  # Events will be generated in chunks of this size

  bounds = { i[0] : (i[2], i[3]) for i in parameters_list }  # Bounds and exponential factor for generation of cuts

  branches = observables_toys + [ i[0] for i in parameters_list ]

  n = 0   # Current tuple size

  arrays = []

  while(True) : 

    # Create Dalitz plot sample
    unfiltered_sample = observables_phase_space.unfiltered_sample(chunk_size)  # Unfiltered array
    sample = observables_phase_space.filter(unfiltered_sample)

    size = sample.shape[0]
    print(f"Filtered chunk size = {size}")

    # Generate final state momenta from Dalitz plot and run through selection
    rnd = tf.random.uniform([size, random_array_size], dtype = atfi.fptype() )    # Auxiliary random array
    array = atfi.stack(selection_with_random_cuts(sample, rnd), axis = 1)
    arrays += [ array ]

    # Increment counters and check if we are done
    size = array.shape[0]
    n += size
    if n > nev : break
    print(f"Selected size = {n}, last = {size}")

  tfr.write_tuple(outfile, atfi.concat(arrays, axis = 0)[:nev,:], branches)

if __name__ == "__main__" : 
  main()
