"""Week 07: TF-IDF and BM25 from scratch. Write by hand (stdlib only; no sklearn, no rank_bm25)."""


def tokenize(text):
    raise NotImplementedError


def tf(term, doc):
    raise NotImplementedError


def idf(term, docs):
    raise NotImplementedError


def tfidf_vector(doc, docs):
    raise NotImplementedError


def bm25_idf(term, docs):
    raise NotImplementedError


def bm25_score(query, doc, docs, k1=1.5, b=0.75):
    raise NotImplementedError


def rank_bm25(query, docs, k1=1.5, b=0.75):
    raise NotImplementedError
