import glob
import os
from tqdm import tqdm
from Utils.preprocessing import *
import joblib
import argparse
import math
import numpy as np
import random
random.seed(69)
data_loc = os.path.join(os.getcwd(), "Data/20_newsgroups/")

class Naive_Bayes():
    '''
    A Naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions.
    '''
    def fit(self, X, y):
        '''
        Fit the model according to the given training data.
        X: array-like, shape = [n_samples, n_features]
        y: array-like, shape = [n_samples]
        output: None
        '''
        self.labels, self.counts = np.unique(y, return_counts=True)
        total = sum(self.counts)
        self.class_prob = self.counts / total
        self.prob_matrix = np.zeros((len(self.labels), X.shape[1], 2))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                label_index = list(self.labels).index(y[i])
                self.prob_matrix[label_index, j, X[i, j]] += 1 / self.counts[label_index]
        
    def predict(self, X):
        '''
        Predict class labels for samples in X.
        X: array-like, shape = [n_samples, n_features]
        output: array-like, shape = [n_samples]
        '''
        pred = []
        for i in range(X.shape[0]):
            prob = np.zeros(len(self.labels))
            for j in range(X.shape[1]):
                for k in range(len(self.labels)):
                    prob[k] += self.prob_matrix[k, j, X[i, j]]
            pred.append(self.labels[np.argmax(prob)])
        return np.array(pred)

def preprocess_data():
    '''
    Preprocess the data and save it to a pickle file.
    '''
    data = []
    for subdir in glob.glob(data_loc + "*"):
        for filename in glob.glob(subdir + "/*"):
            with open(filename, 'r', encoding='latin-1') as f:
                name = str(filename).split("/")[-1]
                content = f.read()
                data.append({'file': name, 'content': content, 'label': subdir.split("/")[-1]})

    # filters the data
    for files in data:
        files['filtered_content'] = filter(files['content'])

    joblib.dump(data, 'Data/Q3_docs.pkl')

def split(split_ratio=0.8):
    '''
    Split the dataset into training and testing data and save it to a pickle file.
    split_ratio: float, ratio of training data to the whole dataset
    '''

    data = joblib.load('Data/Q3_docs.pkl')
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    joblib.dump(train_data, 'Data/Q3_train_data.pkl')
    joblib.dump(test_data, 'Data/Q3_test_data.pkl')

def create_tf_icf():
    '''
    Creates a tf-icf matrix for the training data and saves it to a pickle file.
    '''
    train_data = joblib.load('Data/Q3_train_data.pkl')
    labels = [dir.split('/')[-1] for dir in glob.glob(data_loc + "*")]
    tokens = dict()
    for file in train_data:
        for token in file['filtered_content']:
            if token not in tokens:
                tokens[token] = [0] * len(labels)
            tokens[token][labels.index(file['label'])] += 1
    
    tf_icf_matrix = [[] for _ in range(len(labels))]
    for token in tokens.keys():
        cf = sum([1 if count > 0 else 0 for count in tokens[token]])
        icf = math.log(len(labels)/ cf)
        for i in range(len(labels)):
            tf = tokens[token][i]
            tf_icf = tf * icf
            tf_icf_matrix[i].append((token, tf_icf))

    for i in range(len(labels)):
        tf_icf_matrix[i].sort(key=lambda x: x[1], reverse=True)
        tf_icf_matrix[i] = tf_icf_matrix[i]

    joblib.dump(tf_icf_matrix, 'Data/Q3_tf_icf_matrix.pkl')

def create_dataset(k):
    '''
    Creates a training and test dataset for top-k features and saves it to a pickle file.
    k: int, number of top-k features to be used
    '''
    tf_icf_matrix = joblib.load('Data/Q3_tf_icf_matrix.pkl')
    features = set()
    for i in range(len(tf_icf_matrix)):
        features.update([token for token, _ in tf_icf_matrix[i][:k]])
    features = list(features)
    train_data = joblib.load('Data/Q3_train_data.pkl')
    test_data = joblib.load('Data/Q3_test_data.pkl')
    X_train = [[0]*len(features) for _ in range(len(train_data))]
    y_train  = []
    for i, file in enumerate(train_data):
        y_train.append(file['label'])
        for j in range(len(features)):
            if features[j] in file['filtered_content']:
                X_train[i][j] = 1

    X_test = [[0]*len(features) for _ in range(len(test_data))]
    y_test  = []
    for i, file in enumerate(test_data):
        y_test.append(file['label'])
        for j in range(len(features)):
            if features[j] in file['filtered_content']:
                X_test[i][j] = 1
        
    joblib.dump((np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)), f'Data/Q3_cleaned_dataset_{k}.pkl')

def confusion_matrix(y_pred, y_true):
    '''
    Returns the confusion matrix for the predicted and true labels.
    y_pred: array-like, shape = [n_samples]
    y_true: array-like, shape = [n_samples]
    output: array-like, shape = [n_classes, n_classes]
    '''
    labels = [dir.split('/')[-1] for dir in glob.glob(data_loc + "*")]
    matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(y_pred)):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1
    return matrix
    
if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", help="Preprocess the data", action="store_true")
    parser.add_argument("-k", "--num_features", type=int, default=5, help="The number of games to simulate")
    args = parser.parse_args()

    k = args.num_features
    print("Setting number of features as {}".format(k))

    if args.preprocess:
        print("Preprocessing the data...")
        preprocess_data()
    
    for split_ratio in [0.5, 0.7, 0.8]:
        print(f"Splitting the data in ratio {split_ratio}...")
        split(split_ratio)
        print("Creating tf-icf matrix...")
        create_tf_icf()
        print("Creating processed dataset...")
        create_dataset(k)

        dataset = joblib.load(f'Data/Q3_cleaned_dataset_{k}.pkl')
        X_train, y_train, X_test, y_test = dataset

        nb = Naive_Bayes()
        nb.fit(X_train, y_train)
        pred = nb.predict(X_test)
        print("Accuracy: {}".format(np.mean(pred == y_test)))
        print(confusion_matrix(pred, y_test))
        print("------------------------------------------------------")