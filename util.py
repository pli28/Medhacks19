import urllib
import re, string
import hw2
from urllib import parse, request
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')


def process_string(sent):
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    stopwords = hw2.read_stopwords('common_words')
    words = sent.split(' ')
    final = []
    for w in words:
        w = w.lower()
        if w != '' and (w not in stopwords) and (regex.search(w) is None):
            f = re.sub(r'[^\w\s]', '', w)
            final.append(stemmer.stem(f))
    print(final)
    return final



def word_list(link):
    stopwords = hw2.read_stopwords('common_words')
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    url = link
    try:
        html = request.urlopen(url).read()
    except:
        print('cannot open page')
        return []
    soup = BeautifulSoup(html)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    word = []
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    for line in lines:
        li = line.split(' ')
        for w in li:
            w = w.lower()
            if w != '' and (w not in stopwords) and (regex.search(w) is None and (w != 'cdc')):
                f = re.sub(r'[^\w\s]', '', w)
                final = stemmer.stem(f)
                word.append(final)
    return word


def extract_medical_terms():
    stopwords = hw2.read_stopwords('common_words')
    html = request.urlopen('https://en.wikipedia.org/wiki/List_of_medical_symptoms').read()
    soup = BeautifulSoup(html)
    word = []
    for t in soup.find_all('li'):
        for w in t.text.split(' '):
            if w.lower() not in stopwords and ('(' not in w.lower()) and ("/" not in w.lower())\
                    and ("(" not in w.lower()):
                f = re.sub(r'[^\w\s]', '', w.lower())
                final = stemmer.stem(f)
                word.append(final)

    with open('medical_terms', 'w', encoding='utf-8') as f:
        for item in word:
            f.write("%s\n" % item)


if __name__ == '__main__':
    extract_medical_terms()


