import re
import string
from typing import Dict, List, Tuple
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt_tab')
nltk.download('stopwords')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

SMALL_SAMPLE_DATASET = "../data/guess-query/training.tsv" #'training.txt'


def read_data(path: str) -> List[Dict[str, str]]:
    """
    Read data from a TSV file.
    """
    data = list()
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            line_split = line.split("\t")
            row = dict()
            row["query"] = line_split[1]
            row["answer"] = line_split[0]
            data.append(row)

    return data

def preprocess_text(data: List[Dict[str, str]]) -> List[Tuple[List[str], List[str]]]:
    """
    Lower casting, tokenization, stopwords removal.
    Note: stemming or lemmatization is applied.
    """
    out = list()
    for row in data:
        row["query"] = row["query"].lower()
        row["answer"] = row["answer"].lower()

        row["query"] = word_tokenize(row["query"])
        row["answer"] = word_tokenize(row["answer"])

        stopwords_tokens = stopwords.words("english")
        answer  = [
            word for word in row["answer"] if (word not in stopwords_tokens) and (word not in string.punctuation)
        ]
        query = [
            token for word in row["query"] for token in re.split(r"-|/|_", word) if (token not in stopwords_tokens) and (token not in string.punctuation)
        ]

        out.append((query, answer))

    return out

def create_vocab(query_tokens: List[List[str]]):
    """
    Create a vocabulary from the tokens.
    """
    return list(set([item for sublist in query_tokens for item in sublist]))


def construct_features(answer_tokens: List[str], vocab: List[str]) -> np.ndarray:
    feature_matrix = np.zeros([len(answer_tokens), len(vocab)])
    for idx, item_answer_tokens in enumerate(answer_tokens):
        for token in item_answer_tokens:
            vocab_token_index = vocab.index(token) if token in vocab else -1
            if  vocab_token_index >= 0:
                feature_matrix[idx, vocab_token_index] += 1

    return feature_matrix


if __name__=="__main__":
    raw_content = read_data(SMALL_SAMPLE_DATASET)
    answer_query_tuples = preprocess_text(raw_content)

    query_tokens, answer_tokens = zip(*answer_query_tuples)
    vocabulary = create_vocab(query_tokens)

    X_train = construct_features(answer_tokens, vocabulary)
    #print("X_train 1st:", X_train[0, :], "shape: ", X_train.shape)

    # Create frequency matrix
    count_vectorizer = CountVectorizer(vocabulary=vocabulary)
    answers_string = [" ".join(tokens) for tokens in answer_tokens]
    X_train_sklearn = count_vectorizer.fit_transform(answers_string)
    #print("Vocabulary:", count_vectorizer.vocabulary_)
    #print("X_train_sklearn 1st:", X_train_sklearn[0, :].toarray(), "\n shape: ", X_train_sklearn.toarray().shape)

    # Create TF-IDF matrix: this makes sure that we are also accounting for the freq of terms in each doc,
    # so if a token is met often in a doc but also term is not freq in other docs, it will have a higher weight
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_sklearn)
    #print("X_train_tfidf 1st:", X_train_tfidf.toarray()[0, :], "\n shape: ", X_train_tfidf.toarray().shape)

    y_train = [" ".join(tokens) for tokens in query_tokens]
    #print("y_train 1st:", y_train[0], "\n length: ", len(y_train))

    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)
    y_pred_proba = log_reg.predict_proba(X_train_tfidf)
    #print("y_pred_proba 1st:", y_pred_proba[0], "\n length: ", len(y_pred_proba))
    y_pred = log_reg.predict(X_train_tfidf)
    #print("y_pred 1st:", y_pred[0], "\n length: ", len(y_pred))

    test = list()
    for i in range(int(input())):
        x = input()
        raw_text = dict(query="", answer=x)
        test.append(raw_text)

    answer_query_tuples = preprocess_text(test)
    query_tokens, answer_tokens = zip(*answer_query_tuples)

    answers_string = [" ".join(tokens) for tokens in answer_tokens]
    X_test = count_vectorizer.transform(answers_string)
    X_test_tfidf = tfidf_transformer.transform(X_test)
    y_pred = log_reg.predict(X_test)

    for i in y_pred:
        print(i)







