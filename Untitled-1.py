# %%
import csv
from sklearn.model_selection import train_test_split


def read_csv(filename):
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # skip one row
        for row in reader:
            text, label = row
            yield text, label


texts = []
labels = []
for text, label in read_csv("input/email_classification.csv"):
    texts.append(text)
    labels.append(1 if label == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.20, random_state=42
)

# %%
from typing import List, Tuple, Dict, Iterable
import math
import string

from collections import defaultdict


class NaiveBayes:
    def __init__(self, k: float) -> None:
        self.k = k
        self.count_spam = 0
        self.count_ham = 0
        self.vocab = set()
        self.token_spam_counts = defaultdict(int)
        self.token_ham_counts = defaultdict(int)

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
        tokens = text.split()
        return tokens

    def fit(self, xs, ys) -> None:

        # iterate through each sample
        for x, y in zip(xs, ys):
            if y == 1:  # if spam
                self.count_spam += 1
            elif y == 0:  # if ham
                self.count_ham += 1

            # iterate through each token in sample
            x_tokenized = self.tokenize(x)
            for token in x_tokenized:
                self.vocab.add(token)
                if y == 1:  # if spam
                    self.token_spam_counts[token] += 1
                elif y == 0:  # if ham
                    self.token_ham_counts[token] += 1

    def _proba_single_token(self, token):
        """
        P(token|spam) : count token in spam messages / count spam messages
        P(token|ham) : count token in ham messages / count ham messages
        """
        count_token_in_spam = self.token_spam_counts[token]
        count_token_in_ham = self.token_ham_counts[token]
        p_token_spam = (count_token_in_spam + self.k) / (self.count_spam + self.k * 2)
        p_token_ham = (count_token_in_ham + self.k) / (self.count_ham + self.k * 2)
        return p_token_spam, p_token_ham

    def _predict_single_sample(self, x):
        x_tokenized = self.tokenize(x)
        log_prob_if_spam = 0
        log_prob_if_ham = 0
        for token in self.vocab:
            p_token_spam, p_token_ham = self._proba_single_token(token)
            # if token appear in text
            # add log proba of seeing it
            if token in x_tokenized:
                log_prob_if_spam += math.log(p_token_spam)
                log_prob_if_ham += math.log(p_token_ham)
            # if token do not appear in text
            # add log proba of not seeing it
            elif token not in x_tokenized:
                log_prob_if_spam += math.log(1 - p_token_spam)
                log_prob_if_ham += math.log(1 - p_token_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)

    def predict(self, xs):
        return [self._predict_single_sample(x) for x in xs]


# texts = [
#     "spam rules",
#     "ham rules",
#     "hello ham",
# ]
# labels = [1, 0, 0]

# bayes = NaiveBayes(k=0.5)
# bayes.fit(texts, labels)

# assert bayes.vocab == {"spam", "ham", "rules", "hello"}
# assert bayes.count_spam == 1
# assert bayes.count_ham == 2
# assert bayes.token_spam_counts == {"spam": 1, "rules": 1}
# assert bayes.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

# texts = ["hello spam", "hello ham"]
# bayes.predict(texts)

# %%
bayes = NaiveBayes(k=1)
bayes.fit(X_train, y_train)

x = X_test[0]
bayes.predict(x)
