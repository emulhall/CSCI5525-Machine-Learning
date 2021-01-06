Emily Mulhall, 5677176, mulha024@umn.edu

#1. Run 'python3 adaboost.py'

#2. Run 'python3 rf.py'

#3. Run 'python3 kmeans.py'

If you do not want to generate your own plots please be sure to comment out the matplotlib lines in all three.  Additionally, if you do not want your own compressed images, comment out the plt.imsave(...) line.

For both #1 and #2, the customer ID has been removed as a feature because, while it has high information gain, it does not generalize well to testing.  Additionally, column 6 has missing data replaced with the mode of that column.

Please make sure that the dataset for #1 and #2, and the image for #3 are all in the same directory as you are running this code from.