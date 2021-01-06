Name: Emily Mulhall
Student ID: 5677176
Email: mulha024@umn.edu
Running the code:

1. Run 'python3 SVM_dual.py'
2. Run 'python3 kernel_SVM.py'
3. This problem requires combining 6 files of features and creating a dataset.  There's two options to run this code:
	a. Run 'python3 multi_SVM.py' with the 'mult_SVM_dataset.csv' in the same folder that you are running the code.  This csv has been included in the zip file with the code and this text file.
	b. Uncomment the 'createDataset()' command in the main method and then run 'python3 multi_SVM.py.'  You must first make sure to unzip the 6 feature files that are included in my submission.  This command will create a 'mult_SVM_dataset.csv' file in your current directory.

	Note: There are a few issues with multi-class SVM classification.  The first arises when a class is never classified.  In this case the default class is class 0.  If a class is classified into multiple classes it will be classified as the largest of the classes.  For example, if a class is classified into 6 and 8, it will be classified as class 8.