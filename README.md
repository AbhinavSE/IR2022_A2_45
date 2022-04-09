# IR2022_A2_45

## Question 1 - [40 Points] Scoring and Term-Weighting

### Jaccard Coefficient
1. First we preprocess the data using the preprocessing.py file.
2. Then we take the query from the user and filter it.
3. Then we calculate the jaccard coefficient for each document and sort them in descending order.
4. Finally we report the top 5 results.

### TF-IDF
1. load data: load all documents
2. get postings list: create a dictionary of all terms and their document frequency with token count in each document
3. calculate tf-idf vector for docs: calculate tf using binary, raw_count, term_frequency, log_normalization, double_normalization and idf for all documents and save it in a file
4. query: calculate tfidf vector for query and calculate cosine similarity with tfidf vector of all documents


## Question 2 - [25 points] Ranked-Information Retrieval and Evaluation

1. First we pickle the data, then we load it and get the DCG of the data.
2. We then generate 5 random permutations of the data and get the DCG of each of them to show that sorting the relevances gives the max DCG.
3. After that we sort the data and get the DCG of the sorted data (Max DCG).
4. Then we output the number of files possible with max DCG.
5. We then get the nDCG of the first 50 documents and then the whole dataset.
6. Finally that we plot the precision-recall curve for the TF-IDF sorted data.

## Question 3 - [ 35 points ] Naive Bayes Classifier

1. First we preprocess the data with the same steps as assignment-1.
2. Then we split the data into training and test sets using the three ratios.
3. We create a tf-icf matrix for the training data and save it to a pickle file.
4. We create a traing and test dataset for the top-k features and save it to a pickle file.
5. Then we train a Naive Bayes classifier on the training dataset and predict the labels of the test dataset.
6. We print the accuracy of the classifier and the confusion matrix.
