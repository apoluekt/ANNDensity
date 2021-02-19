# ANNDensity
ANNs for density estimation in flavour physics amplitude analyses. Code used for feasibility studies reported in https://arxiv.org/abs/1902.01452

## Prerequisites

The code needs TensorFlow environment to run, together with two libraries, AmpliTF (https://github.com/apoluekt/AmpliTF) and TFA2 (https://github.com/apoluekt/TFA2). The latter includes instructions to set up the full environment in `conda`: 
  https://github.com/apoluekt/TFA2/blob/master/doc/01_installation.md
  
## Content of the package

The two subdirectories contain examples of density estimation for the efficiency shape and background over the square Dalitz plot phase space of the Ds->K+pi+pi- decay. 

* Ds2KpipiEfficiency: Estimation of efficiency (Sections 6 and 8.2 of arXiv:1902.01452)
* Ds2KpipiBackground: Estimation of background (Sections 7.2 and 8.3 of arXiv:1902.01452)

## Running the code

Each subdirectory contains the file `runme.sh` which can be executed to perform all the steps (generation ot toy MC samples, ANN training, fitting etc.). It's best to run them on a machine with good GPU. The whole cycle takes a few hours on a machine with Tesla p100 card. 
