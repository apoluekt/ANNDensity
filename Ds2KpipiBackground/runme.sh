#####################################################################
##################### "Simple" ANN density estimation ###############
#####################################################################

date >> timestamps.txt
# Generate the sample to be used for density parametrisation
python GenerateTestSample.py 100000 test_tuple.root

date >> timestamps.txt
# Generate the high-statistics reference sample with the same model
python GenerateTestSample.py 4000000 ref_tuple.root

date >> timestamps.txt
# Parametrise density with ANN directly from the test sample
# with special regularisation to allow for interpolation of 
# background density from the sidebands
python TrainNNSpecial3D.py

date >> timestamps.txt
# Generate high-statistics sample corresponding to the result of the density estimation
# Store output with the chi2 value to the log file
python GenerateFitResult3D.py | tee ann.log



######################################################################
############# Model-assisted ANN density estimation ##################
######################################################################

date >> timestamps.txt
# Generate toy sample with ransomised density parameters
python GenerateToys.py

date >> timestamps.txt
# Train ANN model for the density as a function of model paramaters
python TrainNN.py

date >> timestamps.txt
# Fit the test sample with the trained model
# Store the output with the chi2 value to the log file
python FitSample.py | tee ma_ann.log

date >> timestamps.txt
#(Plotting scripts that need ROOT and rootpy)
#python PlotFit.py
#python PlotFit3D.py
