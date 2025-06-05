import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(path):
    with open(path, "r") as file:
        return file.read()

def preprocess_text(text_raw: str) -> str:
    # Lowercase and simple whitespace split
    tokens_raw = [token.strip().lower() for token in re.split(r" |\n|\t", text_raw)]

    stop_words = {"the", "a", "an", "in", "to", "for", "will", "be", "is", "are"}

    # Split further on dashes, underscores, and slashes
    tokens = [
        sub_token
        for token in tokens_raw
        for sub_token in re.split(r"-|_|/", token)
        if sub_token not in stop_words and sub_token not in string.punctuation and sub_token != ""
    ]

    return " ".join(tokens)

def determine_class(x_in, x_computers, x_fruits):
    sim_computers = cosine_similarity(x_in, x_computers)[0][0]
    sim_fruits = cosine_similarity(x_in, x_fruits)[0][0]

    return "computer-company" if sim_computers > sim_fruits else "fruit"

def main():
    apple_computers = "apple-computers.txt"
    apple_fruit = "apple-fruit.txt"

    text_computers = preprocess_text(read_file(apple_computers))
    text_fruit = preprocess_text(read_file(apple_fruit))

    # Build vocabulary from both base documents
    vocabulary = list(set(text_computers.split()).union(set(text_fruit.split())))

    count_vectorizer = CountVectorizer(vocabulary=vocabulary)
    X_counts = count_vectorizer.fit_transform([text_computers, text_fruit])

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)

    x_computers = X_tfidf[0]
    x_fruits = X_tfidf[1]

    n = int(input("Enter number of test documents: "))
    test_docs = [input(f"Enter test document {i+1}: ") for i in range(n)]

    test_docs_preprocessed = [preprocess_text(doc) for doc in test_docs]
    X_test_counts = count_vectorizer.transform(test_docs_preprocessed)
    X_test_tfidf = tfidf.transform(X_test_counts)

    for i, x in enumerate(X_test_tfidf):
        label = determine_class(x, x_computers, x_fruits)
        print(label)

if __name__ == "__main__":
    main()
