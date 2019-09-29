import itertools
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
import sys

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

number_doc = 3400


# File IO and processing
class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file, encoding='utf-8') as f:
        return set([x.strip() for x in f.readlines()])


stopwords = read_stopwords('common_words')
stemmer = SnowballStemmer('english')


def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]


# Term-Document Matrix
class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float


def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq


def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict


def compute_tfidf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    tf = term_num_list(doc)
    total_term = len(doc.author) + len(doc.keyword) + len(doc.title) + len(doc.abstract)
    for word, freq in tf.items():
        tf[word] = tf[word]/total_term
    idf = defaultdict(float)
    for str, freq in doc_freqs.items():
        idf[str] = np.log(number_doc/(1+freq))
    tfidf = defaultdict(float)
    for word, freq in tf.items():
        tfidf[word] = freq * idf[word]
    return dict(tfidf)


def compute_boolean(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        if vec[word] == 0:
            vec[word] += weights.author
    for word in doc.keyword:
        if vec[word] == 0:
            vec[word] += weights.keyword
    for word in doc.title:
        if vec[word] == 0:
            vec[word] += weights.title
    for word in doc.abstract:
        if vec[word] == 0:
            vec[word] += weights.abstract
    return dict(vec)


def term_num_list(doc: Document):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += 1
    for word in doc.keyword:
        vec[word] += 1
    for word in doc.title:
        vec[word] += 1
    for word in doc.abstract:
        vec[word] += 1
    return vec


def num_doc() :
    if number_doc == 0:
        number_doc == read_docs("cacm.raw")
    return number_doc

### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)


def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


def dice_sim(x, y):
    num = 2 * dictdot(x, y)
    sumx = sum_weight(x)
    sumy = sum_weight(y)
    return num/(sumx + sumy)


def jaccard_sim(x, y):
    min = 0
    max = 0
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    for key in keys:
        if x.get(key, 0) >= y.get(key, 0):
            min += y.get(key, 0)
            max += x.get(key,0)
        else:
            min += x.get(key, 0)
            max += y.get(key, 0)
    return min/max


def overlap_sim(x, y):
    num = dictdot(x, y)
    sumx = sum_weight(x)
    sumy = sum_weight(y)
    small = min(sumx, sumy)
    return num/small


def sum_weight(x: Dict[str, float]):
    sum = 0
    for word, freq in x.items():
        sum += freq
    return sum


# Precision/Recall
def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b


def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''''
    total_rel = len(relevant)
    length = len(results)
    rec_prec = defaultdict(float)
    rec_prec[0] = 1
    x: int = 0
    count = 0
    while x < length:
    # print(x)
        if results[x] in relevant:
            count += 1
            rec = count/total_rel
            pre = count/(x + 1)
            if rec == recall:
                return pre
            rec_prec[rec] = pre
        x += 1

    keylist = list(rec_prec)
    index = 0
    while index < len(keylist):
        if (keylist[index] < recall) and (keylist[index + 1] > recall):
            return interpolate(keylist[index], rec_prec[keylist[index]], keylist[index + 1], rec_prec[keylist[index + 1]], recall)
            break
        index += 1

    return 1


def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3


def mean_precision2(results, relevant):
    index = 1
    sum = 0
    while index <= 10:
        sum += precisionormn_at(index/10, results, relevant)
        index += 1
    return sum/10

def norm_recall(results, relevant):
    sum_rank = 0
    for rel in relevant:
        sum_rank += results.index(rel) + 1
    index = 1
    sum_i = 0
    while index <= len(relevant):
        sum_i += index
        index += 1

    return 1 - ((sum_rank - sum_i) / (len(relevant) * (len(results) - len(relevant))))


def norm_precision(results, relevant):
    N = len(results)
    Rel = len(relevant)
    sum_rank = 0
    for rel in relevant:
        sum_rank += np.log(results.index(rel) + 1)
    index = 1
    sum_i = 0
    while index <= len(relevant):
        sum_i += np.log(index)
        index += 1
    numerator = sum_rank - sum_i
    denominator = N*math.log(N) - (N - Rel)*math.log(N - Rel) - Rel*math.log(Rel)
    return 1 - (numerator / denominator)


### Extensions

# TODO: put any extensions here


### Search

def experiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

         # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()


def vector_mean(vec: List[Dict]):
    index = 0
    pos = 0
    dimension = 0
    while index < len(vec):
        if len(vec[index]) > dimension:
            dimension = len(vec[index])
            pos = index
        index += 1
    mean = defaultdict()
    for d in vec:
        for k in d.keys():
            if k not in mean.keys():
                sum = 0
                for v in vec:
                    if k in v.keys():
                        sum += v[k]
                mean[k] = sum/len(vec)
    return dict(mean)


def sum_vector(x: Dict, y: Dict):
    vec = dict()
    for k in set(x) | set(y):
        vec[k] = x.get(k, 0) + y.get(k, 0)
    return vec


def minus_vector(x: Dict, y: Dict):
    vec = dict()
    for k in set(x) | set(y):
        vec[k] = x.get(k, 0) - y.get(k, 0)
    return vec


def input_query():
    docs = read_docs('cacm.raw')
    stopwords = read_stopwords('common_words')
    term_weights = TermWeights(author=1, title=1, keyword=1, abstract=1)
    query = ''
    while query != 'exit':
        query = input("Enter query, or enter exit\n")
        if query == 'exit':
            break
        query_doc = Document(1, [],[],[],query.split(" "))
        temp_list = list()
        temp_list.append(query_doc)
        processed_docs, processed_query = process_docs_and_queries(docs, temp_list, True, True, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        query_vector = compute_tfidf(processed_query[0], doc_freqs, term_weights)
        doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
        results = search(doc_vectors, query_vector, cosine_sim)
        for i in range(len(results)):
            if i <= 19:
                print(results[i])
            else:
                break


def relevance():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')
    term_weights = TermWeights(author=1, title=1, keyword=1, abstract=1)

    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, True, stopwords)
    query = queries[0]
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
    query_vector = compute_tfidf(query, doc_freqs, term_weights)
    results = search(doc_vectors, query_vector, cosine_sim)
    top_20 = list()
    for i in range(len(results)) :
        if i <= 19:
            top_20.append(results[i])
        else:
            break
    relevant_id = rels[1]
    irrelevant_id = list()
    for r in top_20:
        if r not in relevant_id:
            irrelevant_id.append(r)

    relevant_vector = list()
    irrelevant_vector = list()
    for d in relevant_id:
        relevant_vector.append(doc_vectors[d - 1])
    for d in irrelevant_id:
        irrelevant_vector.append(doc_vectors[d - 1])

    new_vec = minus_vector(sum_vector(vector_mean(relevant_vector), vector_mean(relevant_vector)),
                           vector_mean(irrelevant_vector))

    new_result = search(doc_vectors, new_vec, cosine_sim)
    new_top_20 = list()
    for i in range(len(new_result)):
        if i <= 19:
            new_top_20.append(new_result[i])
        else:
            break
    print("old")
    print(top_20)
    print("new")
    print(new_top_20)
    print("right answer")
    print(relevant_id)


if __name__ == '__main__':
    if sys.argv[1] == "relevance":
        relevance()
    elif sys.argv[1] == "in":
        input_query()
    else:
        experiment()