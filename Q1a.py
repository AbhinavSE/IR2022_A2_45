import time
import numpy as np
from Utils.preprocessing import *
import os
import glob
from tqdm import tqdm
import joblib
import argparse
import json
import pickle as pkl

def preprocess_data():
    data_loc = os.path.join(os.getcwd(), "Data/Humor,Hist,Media,Food/")
    data = []
    error_files = []
    for filename in glob.glob(data_loc + "*"):
        with open(filename, 'r', encoding='latin-1') as f:
            name = str(filename).split("/")[-1]
            content = f.read()
            data.append({'file': name, 'content': content})

    # filters the data
    for files in tqdm(data):
        files['filtered_content'] = filter(files['content'])

    for file in data:
        print(f"Text for file- {file['file']}")
        print(file['content'][:20])
        print(f"Cleaned text for file- {file['filtered_content']}")
        print(file['filtered_content'][:20])
        break

    data.sort(key=lambda x: x['file'])

    joblib.dump(data, 'Data/docs.pkl')


def jaccard_coefficient(query, doc):
    query_set = set(query)
    doc_set = set(doc)
    intersection = len(query_set.intersection(doc_set))
    union = len(query_set.union(doc_set))
    return intersection / union


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", help="Preprocess the data", action="store_true")
    args = parser.parse_args()

    if args.preprocess:
        print("Preprocessing the data...")
        preprocess_data()

    start = time.perf_counter()
    data = joblib.load('Data/docs.pkl')
    print(f'Time to load: {time.perf_counter() - start :.2f}s')

    query = input("Enter Query: ")
    filtered_query = filter(query)
    print(filtered_query)

    jaccard_scores = []
    print("Calculating Scores")
    for doc in tqdm(data):
        jaccard_scores.append({"Name": doc['file'], "Score": jaccard_coefficient(filtered_query, doc['filtered_content'])})
    jaccard_scores.sort(key=lambda x: x['Score'], reverse=True)

    print("Top 5 results")
    print(jaccard_scores[:5])

'''
First we preprocess the data using the preprocessing.py file.
Then we take the query from the user and filter it.
Then we calculate the jaccard coefficient for each document and sort them in descending order.
Finally we report the top 5 results.
'''
