Emily Mulhall, 5677176, mulha024@umn.edu

Question #3:
 Part a: To run: type 'python3 LDA1dThres.py 10' into the terminal.
 	10 is the number of cross-validation folds and can be any other integer.
 Part b: To run: type 'python3 LDA2dGaussGM.py 10' into the terminal.
 	10 is the number of cross-validation folds and can be any other integer.

 Question #4:
 Part a: To run: type 'python3 logisticRegression.py 10 [10 25 50 75 100]' into the terminal.
 	10 is the number of splits we will perform.  In this case we will split the data into 80% training data and 20% testing data randomly.  We will then do this 9 more times and run our experiments on the data.  [10 25 50 75 100] is our training data percentages vector.  These are the percentages by which we will split our training data and run experiments.  Thus, the first time we run our experiment it will use 10% of the training data split out, the second time will use 25%, and so on.  Note that the brackets are not necessary, but it will not work properly with a different type of bracket marker, such as {, (, etc.
 Part b: To run: type 'python3 naiveBayesGaussian.py 10 [10 25 50 75 100]' into the terminal.
  	10 is the number of splits we will perform.  In this case we will split the data into 80% training data and 20% testing data randomly.  We will then do this 9 more times and run our experiments on the data.  [10 25 50 75 100] is our training data percentages vector.  These are the percentages by which we will split our training data and run experiments.  Thus, the first time we run our experiment it will use 10% of the training data split out, the second time will use 25%, and so on.  Note that the brackets are not necessary, but it will not work properly with a different type of bracket marker, such as {, (, etc.