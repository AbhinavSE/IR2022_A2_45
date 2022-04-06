import math
import joblib
import random
from collections import Counter
import matplotlib.pyplot as plt
random.seed(69)

def pickle_data(path):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        split = line.split()
        relevance = split[0]
        query = split[1]
        url = split[2:]
        if query.split(":")[1] == "4":
            data.append((relevance, query, url))
    joblib.dump(data, "Data/IR-assignment-2-data.pkl")

def get_relevances(data):
    return [int(line[0]) for line in data]

def get_DCG(relevances):
    dcg = 0
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg

def get_nDCG(relevances):
    ndcg = 0
    for i, rel in enumerate(relevances):
        ndcg += (2 ** rel - 1) / math.log2(i + 2)
    return ndcg/get_DCG(sorted(relevances, reverse=True))

def plot_precision_recall(relevances):
    precision = []
    recall = []
    relevant_docs= [1 if rel>0 else 0 for rel in relevances]
    relevant_till_now = 0
    total_relevant = sum(relevant_docs)
    for i in range(1, len(relevant_docs)+1):
        relevant_till_now += relevant_docs[i-1]
        precision.append(relevant_till_now/i)
        recall.append(relevant_till_now/total_relevant)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

path = "Data/IR-assignment-2-data.pkl"
data = joblib.load(path)

relevances = get_relevances(data)
print("DCG of original data:", get_DCG(relevances))

random_permutation1_relevances = random.sample(relevances, len(relevances))
print("DCG of random permutation 1:", get_DCG(random_permutation1_relevances))
random_permutation2_relevances = random.sample(relevances, len(relevances))
print("DCG of random permutation 2:", get_DCG(random_permutation2_relevances))
random_permutation3_relevances = random.sample(relevances, len(relevances))
print("DCG of random permutation 3:", get_DCG(random_permutation3_relevances))
random_permutation4_relevances = random.sample(relevances, len(relevances))
print("DCG of random permutation 4:", get_DCG(random_permutation4_relevances))
random_permutation5_relevances = random.sample(relevances, len(relevances))
print("DCG of random permutation 5:", get_DCG(random_permutation5_relevances))
random_permutation6_relevances = random.sample(relevances, len(relevances))

sorted_relevances = sorted(relevances, reverse=True)
print("DCG of sorted data:", get_DCG(sorted_relevances))

rel_counter = Counter(relevances)
number_of_files_with_max_dcg = 1
for key, val in rel_counter.items():
    number_of_files_with_max_dcg *= math.factorial(val)

print("Number of files with max DCG:", number_of_files_with_max_dcg)
print()

print("nDCG of original data at 50:", get_nDCG(relevances[:50]))
print("nDCG of whole dataset:", get_nDCG(relevances))

sorted_relevances_tfidf = get_relevances(sorted(data, reverse=True, key=lambda x: x[2][74].split(":")[1]))
plot_precision_recall(sorted_relevances_tfidf)