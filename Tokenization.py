import re
import tokenize
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# defined stop words
cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 0
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = (word for word in words if word not in cachedStopWords)
#    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    tokens = (PorterStemmer().stem(token) for token in words)
    p = re.compile('[a-zA-Z]+')
#    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    filtered_tokens = [token for token in tokens if p.match(token) and len(token) >= min_length]
    return filtered_tokens
