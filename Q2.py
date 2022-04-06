import math
import joblib
import random
from collections import Counter
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