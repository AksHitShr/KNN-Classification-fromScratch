### KNN Implementation from scratch in Python and Sklearn Decision Trees
### Akshit Sharma

#### KNN Classification Tasks

> Standard Ratio for Train-Test split taken as 80:20.

> Plot of k vs accuracy done for Distance Metric as Euclidean, VIT Encoding and 80:20 Train-Test Split, for k values ranging from 1 to 38 as k should be less tan sqrt(n), where n is the train set size.

> Testing is done using a bash script named 'eval.sh', by executing command:
`
./eval.sh <filename>
`
where filename is the name of .npy file containg the test data samples, just like the train data samples taken from data.npy file given.

> Proper error handling done for testing like testing if the test samples file exists in the same directory, whether it has the .npy extension, whether it is passed as an argument while running the script,etc.

> Assuming that only testing needed for the eval scipt is to list the top 20 set of hyperparameters and their correspondiing metrices when used for running KNN model.

> The execution time can be improved by using vectorization (using numpy functions like np.sqrt, np.dot, np.sum, etc. and by also reshaping the train and test sets to avoid the use of multiple for loops to calculate distances seperately).

> Plot for inference time made for comparing the times of best KNN implementation and the inbuilt KNN implementation in Sklearn.

> Plot for inference time for different train-test splits done for train set ratios of 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 for best KNN implementation and the inbuilt Sklearn implementation for KNN.

### Decision Tree Classification Tasks

> Data Visualization: plotted graphs to show gender distribution, frequency of industries, number of people with given no. of children, frequency of different occupations, frequency of most bought items int the given dataset.

> Standard Ratio for Train:Test split taken as 80:20.

> K-Fold validation done for model with highest F1-Score (Macro) for both settings.

> Best K value chosen on the basis of highest F1-Score (Macro) (mean of k F1-Scores (macro) after taking each of the k parts as test set and performing classification).
