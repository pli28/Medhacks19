import hw2, hw4
import util
import mechanize
import sqlite3
from queue import Queue, PriorityQueue
from bs4 import BeautifulSoup
from string import ascii_lowercase
from urllib import parse, request
from urllib.parse import urlparse
from collections import defaultdict
import csv
import operator
import json
from datetime import date, datetime


medical_terms = hw2.read_stopwords('medical_terms')


def renew_data():
    base = 'https://www.cdc.gov/diseasesconditions/az/'
    disease_pages_alpha = []
    disease_pages = []
    for a in ascii_lowercase:
        link = base + a + '.html'
        disease_pages_alpha.append(link)
        res = request.urlopen(link)
        html = res.read()
        soup = BeautifulSoup(html, 'html.parser')
        target = soup.find("div", {"class": "az-content"})
        for link in target.findChildren("a"):
            disease = [link.text, link.get('href')]
            disease_pages.append(disease)
    return disease_pages
    # store_csv('disease_pages.csv', disease_pages)


# create vector according to the first three layer of descriptions
# return the vector representation of a disease given its link
def create_vector(root):
    queue = Queue()
    queue.put((0, root))
    visited = []
    word_list = []
    depth = 0
    domain = urlparse(root).netloc
    while not queue.empty() and depth < 2:
        obj = queue.get()
        url = obj[1]
        depth = obj[0] + 1
        word_list = word_list + util.word_list(url)
        print(url)
        try:
            req = request.urlopen(url)
            html = req.read()

            visited.append(url)
            hw4.visitlog.debug(url)

            for link, title in hw4.parse_links(url, html):
                if (urlparse(link).netloc in domain) or (domain in urlparse(link).netloc):
                    if hw4.not_self_reference(link, url) and (link not in visited):
                            queue.put((depth, link))

        except Exception as e:
            print(e, url)
    word_list = [x for x in word_list if x != '']
    return compute_term_frequency(word_list)


def create_database():
    conn = sqlite3.connect('diseases.db')
    c = conn.cursor()
    try:
        c.execute('''DROP TABLE Diseases''')
    except:
        print('CANNOT DROP TABLE')
    c.execute('''CREATE TABLE Diseases(id INTEGER PRIMARY KEY, name TEXT, vec TEXT)''')
    today = date.today()
    disease_link = renew_data()
    id = 0
    for d in disease_link:
        id += 1
        name = d[0]
        link = d[1]
        vec = json.dumps(create_vector(link))
        try:
            c.execute('''INSERT INTO Diseases(id, name, vec)VALUES (?,?,?)''', (id, name, vec))
        except Exception as e:
            print(e)
            print(name)
            conn.rollback()
    conn.commit()
    conn.close()


def engine(query: str):
    words = util.process_string(query)
    vec = compute_term_frequency(words)
    conn = sqlite3.connect('diseases.db')
    c = conn.cursor()
    c.execute('''SELECT name, vec FROM Diseases''')
    target = defaultdict()
    for row in c:
        name = row[0]
        d_vec = json.loads(row[1])
        sim = hw2.cosine_sim(d_vec, vec)
        target[sim] = name
    idx = 0
    for i in sorted(target.keys(), reverse=True):
        if idx > 10:
            break
        idx += 1
        print(target[i])
        print(i)


# return a vector representation of the input word list
def compute_term_frequency(words):
    vec = defaultdict()
    for w in words:
        if vec.get(w, None) is None:
            vec[w] = 0
        if w in medical_terms:
            vec[w] += 20
        else:
            vec[w] += 1
    return dict(vec)


def store_csv(name, data):
    with open(name, 'w') as resultFile:
        for row in data:
            resultFile.write(row[0] + ',' + row[1] + '\n')
        resultFile.close()


def main():
    engine('Heart')


if __name__ == '__main__':
    main()
