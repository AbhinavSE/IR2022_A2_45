# IR2022_A2_45

## Question 1 - [40 Points] Scoring and Term-Weighting

### Jaccard Coefficient
First we preprocess the data using the preprocessing.py file.
Then we take the query from the user and filter it.
Then we calculate the jaccard coefficient for each document and sort them in descending order.
Finally we report the top 5 results.

### TF-IDF


## Question 2 - [25 points] Ranked-Information Retrieval and Evaluation

First we pickle the data, then we load it and get the DCG of the data.
We then generate 5 random permutations of the data and get the DCG of each of them to show that sorting the relevances gives the max DCG.
After that we sort the data and get the DCG of the sorted data (Max DCG).
Then we output the number of files possible with max DCG.
We then get the nDCG of the first 50 documents and then the whole dataset.
Finally that we plot the precision-recall curve for the TF-IDF sorted data.

## Question 3 - [ 35 points ] Naive Bayes Classifier

First we preprocess the data with the same steps as assignment-1.
Then we split the data into training and test sets using the three ratios.
We create a tf-icf matrix for the training data and save it to a pickle file.
We create a traing and test dataset for the top-k features and save it to a pickle file.
Then we train a Naive Bayes classifier on the training dataset and predict the labels of the test dataset.
We print the accuracy of the classifier and the confusion matrix.
