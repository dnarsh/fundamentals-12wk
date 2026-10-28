"""Week 09: logistic regression with batch gradient descent, plain Python. Write by hand."""


def sigmoid(z):
    raise NotImplementedError


def log_loss(y_true, p):
    raise NotImplementedError


class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        raise NotImplementedError
