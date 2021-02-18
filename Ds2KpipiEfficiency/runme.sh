#####################################################################
##################### "Simple" ANN density estimation ###############
#####################################################################

# Generate the sample to be used for density parametrisation
python GenerateTestSample.py 100000 test_tuple.root

# Generate the high-statistics reference sample with the same model
python GenerateTestSample.py 4000000 ref_tuple.root

# Parametrise density with ANN directly from the test sample
python TrainNN2D.py

# Generate high-statistics sample corresponding to the result of the density estimation
python GenerateFitResult2D.py



######################################################################
############# Model-assisted ANN density estimation ##################
######################################################################

# Generate toy sample with ransomised density parameters
python GenerateToys.py

# Train ANN model for the density as a function of model paramaters
python TrainNN.py

# Fit the test sample with the trained model
python FitSample.py



#(Plotting scripts that need ROOT and rootpy)
#python PlotFit.py
#python PlotFit2D.py
