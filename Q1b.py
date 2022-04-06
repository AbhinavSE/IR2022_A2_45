import joblib
from collections import Counter
import numpy as np

from tqdm import tqdm


def getWordCount(words):
    """
    This function counts the number of times a word appears in a document.
    """
    return Counter(words)


def getTermFrequency(term: str, doc: dict, postingsList: dict[dict[str]], method: str):
    """
    This function calculates the term frequency of a word in a document.
    """
    if method == 'binary':
        return 1 if term in doc['filtered_content'] else 0
    elif method == 'raw_count':
        return postingsList[term][doc['file']]
    elif method == 'term_frequency':
        return postingsList[term][doc['file']] / len(doc['filtered_content'])
    elif method == 'log_normalization':
        return np.log(1 + postingsList[term][doc['file']])
    elif method == 'double_normalization':
        return 0.5 + 0.5 * (postingsList[term][doc['file']] / Counter(doc['filtered_content']).most_common(1)[0][1])


def getInverseDocumentFrequency(term: str, data: list[dict], docName: str, postingsList: dict[dict[str]]) -> float:
    """
    This function calculates the inverse document frequency of a word in a document.
    IDF(word) = log(total no. of documents / document frequency(word)+1)
    """
    return np.log(1 + len(data) / postingsList[term][docName])


def getPostingsList(data: list[dict]):
    print("Getting Postings List")
    postingsList = {}
    for doc in tqdm(data):
        for term in doc['filtered_content']:
            if term not in postingsList:
                postingsList[term] = {doc['file']: doc['filtered_content'].count(term)}
            else:
                postingsList[term][doc['file']] = doc['filtered_content'].count(term)
    return postingsList


def getTFIDFVector(data: list[dict], postingsList: dict[dict[str]]) -> np.array:
    """
    This function calculates the TF-IDF Vector of all documents.
    """
    tfidfVector = np.zeros((len(data), len(postingsList)))
    for i, doc in enumerate(tqdm(data)):
        for j, term in enumerate(postingsList):
            if term in doc['filtered_content']:
                tfidfVector[i, j] = getTermFrequency(term, doc, postingsList) * getInverseDocumentFrequency(term, data, doc['file'], postingsList)
    return tfidfVector


if __name__ == "__main__":
    data = joblib.load('Data/docs.pkl')

    # Create the Postings List
    # postingsList = getPostingsList(data)
    # joblib.dump(postingsList, 'Data/postingsList.pkl', compress=9)

    postingsList = joblib.load('Data/postingsList.pkl')
    for method in ['binary', 'raw_count', 'term_frequency', 'log_normalization', 'double_normalization']:
        tfidfVector = getTFIDFVector(data, postingsList)
        joblib.dump(tfidfVector, 'Data/tfidfVector_{}.pkl'.format(method), compress=9)
