Emily Mulhall, 5677176, mulha024@umn.edu

#2. Run 'python3 neural_net.py'  For this problem, though not specified, I used 20% of the training data as a validation set.  This was used with early stopping in order to stop training when the validation loss no longer decreases for more than 2 epochs.  Using a validation set prevents overfitting to the training data.

If you would not like your own plots please comment out the matplotlib code.

#3. Run 'python3 cnn.py'  Again, a validation set was used in order to determine convergence and prevent overfitting.  A file named TimeHistory.py is included in the zip file and must be in the same directory you are running cnn.py from.  This file is a callback that tracks the training time and is used for the plots of batch size vs. run size when comparing different optimizers.  

If you would not like your own plots please comment out the matplotlib code.