from multiprocessing import Pool, current_process, freeze_support
import os
import joblib
from collections import Counter
import numpy as np
from tqdm import tqdm
from Utils.preprocessing import filter
from sklearn.metrics.pairwise import cosine_similarity


class TFIdfVectorizer:
    METHODS = ['binary', 'raw_count', 'term_frequency', 'log_normalization', 'double_normalization']

    def __init__(self, data):
        """
        TF-IDF Vectorizer
        """
        freeze_support()
        self.data = data
        # Get the Postings List
        self.postingsList = self.getPostingsList(data)
        # Paralellize and Calculate TF-IDF Vector for each type of method
        self.storeTFIDFVector()

    def storeTFIDFVector(self):
        """
        Store TF-IDF Vector for each method
        """
        with Pool(processes=4) as pool:
            pool.starmap_async(self.saveTFIDFVector, [(data, self.postingsList, method) for method in self.METHODS]).get()

    def getTermFrequency(self, term: str, doc: dict, postingsList: dict[dict[str]], method: str):
        """
        This function calculates the term frequency of a word in a document.
        """
        if method == 'binary':
            return 1 if doc['file'] in postingsList[term] else 0
        elif method == 'raw_count':
            return postingsList[term][doc['file']]
        elif method == 'term_frequency':
            return postingsList[term][doc['file']] / len(doc['filtered_content'])
        elif method == 'log_normalization':
            return np.log(1 + postingsList[term][doc['file']])
        elif method == 'double_normalization':
            return 0.5 + 0.5 * (postingsList[term][doc['file']] / Counter(doc['filtered_content']).most_common(1)[0][1])

    def getInverseDocumentFrequency(self, term: str, data: list[dict], postingsList: dict[dict[str]]) -> float:
        """
        This function calculates the inverse document frequency of a word in a document.
        IDF(word) = log(total no. of documents / document frequency(word)+1)
        """
        return np.log(1 + len(data) / (len(postingsList[term]) + 1))

    def getPostingsList(self, data: list[dict]):
        print("Loading Postings List")
        if os.path.exists('Data/postingsList.pkl'):
            return joblib.load('Data/postingsList.pkl')
        postingsList = {}
        # tqdm inside starmap_async
        for doc in tqdm(data):
            for term in doc['filtered_content']:
                if term not in postingsList:
                    postingsList[term] = {doc['file']: doc['filtered_content'].count(term)}
                else:
                    postingsList[term][doc['file']] = doc['filtered_content'].count(term)
        return postingsList

    def saveTFIDFVector(self, data: list[dict], postingsList: dict[dict[str]], tfMethod: str, save: bool = False) -> np.array:
        """
        This function calculates the TF-IDF Vector of all documents.
        """
        if not os.path.exists(f'Data/tfidfVector_{tfMethod}.pkl'):
            print(f'Calculating TF-IDF Vector with TF method {tfMethod}')
            tfidfVector = np.zeros((len(data), len(postingsList)))
            current = current_process()
            pos = current._identity[0] - 1

            # Calculate TF-IDF Vector for each document
            with tqdm(total=len(data), desc=tfMethod, position=pos) as pbar:
                for i, doc in enumerate(data):
                    for j, term in enumerate(postingsList):
                        if term in doc['filtered_content']:
                            tfidfVector[i, j] = self.getTermFrequency(term, doc, postingsList, tfMethod) * self.getInverseDocumentFrequency(term, data, postingsList)
                    pbar.update(1)
            joblib.dump(tfidfVector, f'Data/tfidfVector_{tfMethod}.pkl', compress=9)
        else:
            print(f'TF-IDF Vector with TF method {tfMethod} already exists')

    def query(self, queryText: str, method: str, similarity: str = 'cosine', topk: int = 5) -> list[tuple]:
        """
        Query using cosine similarity
        """
        if method not in ['binary', 'raw_count', 'term_frequency', 'log_normalization', 'double_normalization']:
            raise ValueError(f'Method {method} not supported')
        tfidfVector = joblib.load(f'Data/tfidfVector_{method}.pkl')
        queryTokens = filter(queryText)
        queryPostingsList = {}
        for term in queryTokens:
            if term not in queryPostingsList:
                queryPostingsList[term] = {'query': 1}
            queryPostingsList[term]['query'] += 1
        queryVector = np.zeros(len(self.postingsList))
        for term in queryTokens:
            if term in self.postingsList:
                tf = self.getTermFrequency(term, {'file': 'query', 'filtered_content': queryTokens}, queryPostingsList, method)
                idf = self.getInverseDocumentFrequency(term, self.data, self.postingsList)
                queryVector += tf * idf
        sim = None
        if similarity == 'cosine':
            sim = cosine_similarity(tfidfVector, queryVector.reshape(1, -1)).reshape(-1)

        # Return name and sim of top k similar documents
        docNames = [doc['file'] for doc in self.data]
        return sorted(zip(docNames, sim), key=lambda x: x[1], reverse=True)[:topk]


if __name__ == "__main__":
    data = joblib.load('Data/docs.pkl')
    tfidf = TFIdfVectorizer(data)
    res_docs = tfidf.query("Children don't like reading books", 'term_frequency')
    print(*res_docs, sep='\n')
