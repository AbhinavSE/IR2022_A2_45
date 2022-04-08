import glob
import os
from tqdm import tqdm
from Utils.preprocessing import *
import joblib
import argparse
import math

data_loc = os.path.join(os.getcwd(), "Data/20_newsgroups/")

def preprocess_data():
    data = []
    error_files = []
    for subdir in glob.glob(data_loc + "*"):
        for filename in glob.glob(subdir + "/*"):
            with open(filename, 'r', encoding='latin-1') as f:
                name = str(filename).split("/")[-1]
                content = f.read()
                data.append({'file': name, 'content': content, 'label': subdir.split("/")[-1]})

    # filters the data
    for files in tqdm(data):
        files['filtered_content'] = filter(files['content'])

    joblib.dump(data, 'Data/Q3_docs.pkl')

def create_tf_icf():
    data = joblib.load('Data/Q3_docs.pkl')
    labels = [dir.split('/')[-1] for dir in glob.glob(data_loc + "*")]
    tokens = dict()
    for file in tqdm(data):
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

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", help="Preprocess the data", action="store_true")
    parser.add_argument("-c", "--create_tf_icf", help="Create tf-icf matrix", action="store_true")
    args = parser.parse_args()

    if args.preprocess:
        print("Preprocessing the data...")
        preprocess_data()

    if args.create_tf_icf:
        print("Creating tf-icf matrix...")
        create_tf_icf()

    tf_icf = joblib.load('Data/Q3_tf_icf_matrix.pkl')
    for i in range(len(tf_icf)):
        print(f"Top 10 words for {tf_icf[i][0]}")
        for j in range(10):
            print(f"{tf_icf[i][j][0]} : {tf_icf[i][j][1]}")