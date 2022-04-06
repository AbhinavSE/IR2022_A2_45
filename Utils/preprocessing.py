import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

from wordcloud import STOPWORDS
STOPWORDS = set(stopwords.words('english'))


def remove_article_connector(tokens):
    '''Removes the article connector from the text'''
    article = ["BUT", "YET", "YOU", "THE", "WAS", "FOR", "ARE", "THEY", "THIS", "THAT", "WERE", "WITH", "YOUR", "JUST", "WILL", "WHO", "ABOUT", "THEIR", "OUR",
               "HAS", "WHO", "GET", "THEM", "WHAT", "CAN", "IS", "HIS", "MORE", "OUT", "FROM", "HAVE", "HERE", "WE", "ALL", "THERE", "TO", "ALSO", "AND", "AS", "NOT"]
    pre_words = [token for token in tokens if token.strip() not in article]
    return pre_words


def stemming(tokens):
    '''Stem the words'''
    ps = PorterStemmer()
    return [ps.stem(i) for i in tokens]


def remove_punctuation(tokens):
    '''Removes the punctuation from the text'''
    table = string.punctuation
    tokens = [token for token in tokens if token not in table and token]
    tokens = [re.sub(r"[\n\t]+", " ", s) for s in tokens]
    return tokens


def keep_letters_numbers(text):
    '''Keeps only letters and numbers'''
    return re.sub(r'[^a-z0-9\-\s\\]+', '', text, re.IGNORECASE)


def remove_stopwords(tokens):
    '''Removes the stopwords from the text'''
    ex = {"shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "shouldn't": 'should not', "that's": 'that is', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "we'd": 'we would', "we're": 'we are', "weren't": 'were not', "we've": 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where's": 'where is', "who'd": 'who would', "who'll": 'who will', "who're": 'who are', "who's": 'who is', "who've": 'who have', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', "'re": ' are', "wasn't": 'was not', "we'll": 'we will', "'cause": 'because', "could've": 'could have', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd've": 'i would have', "i'll've": 'i will have', "it'd": 'it would', "it'd've": 'it would have', "it'll've": 'it will have', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't've": 'might not have', "must've": 'must have', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "sha'n't": 'shall not',
          "shan't've": 'shall not have', "she'd've": 'she would have', "she'll've": 'she will have', "should've": 'should have', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "this's": 'this is', "that'd": 'that would', "that'd've": 'that would have', "there'd": 'there would', "there'd've": 'there would have', "here's": 'here is', "they'd've": 'they would have', "they'll've": 'they will have', "to've": 'to have', "we'd've": 'we would have', "we'll've": 'we will have', "what'll've": 'what will have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where've": 'where have', "who'll've": 'who will have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't've": 'will not have', "would've": 'would have', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd've": 'you would have', "aren't": 'are not', "can't": 'cannot', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'll": 'he will', "he's": 'he is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "isn't": 'is not', "it's": 'it is', "it'll": 'it will', "i've": 'i have', "let's": 'let us', "mightn't": 'might not', "mustn't": 'must not', "n't": 'not', "you'll've": 'you will have'}
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [ex.get(token, token) for token in tokens]
    return tokens


def filter(item):
    '''Combines the above functions and filters text'''
    item = item.lower()
    item = re.sub(r'\\N', '', item)
    tokens = word_tokenize(item)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_article_connector(tokens)
    item = keep_letters_numbers(' '.join(tokens))
    return tokens
