import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue, PriorityQueue
from urllib import parse, request
from urllib.parse import urlparse

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')

with open('city.txt', 'r') as f:
    temp = f.readlines()
    f.close()

cities = list()

for s in temp:
    cities.append(s.lower())


# return list of all links and their title in @root
def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):

        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


# return links with their weights
def parse_links_sorted(root, html):
    # TODO: implement
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):

        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            url = parse.urljoin(root, link.get('href'))
            weight = len(re.findall('/', url)) # calculate priority of page base on their distance from their root directory
            yield (weight, url, text)


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))


def not_self_reference(base, url):
    pos = base.find('#')
    if pos != -1:
        base = base[:pos]  # exclude possible postfix
    address = base.split('/')
    final_address = address[len(address) - 1]
    pos = url.find('#')
    if pos != -1:
        url = url[:pos]  # exclude possible postfix
    original = url.split('/')
    original_address = original[len(original) - 1]
    return not original_address == final_address


def not_local_links(base, url):
    return not urlparse(url).netloc == urlparse(base).netloc


def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    # TODO: implement
    res = request.urlopen(url)
    link_list = list(parse_links(url, res.read()))
    for idx, link in enumerate(link_list):
        if not_local_links(url, link[0]) and not_self_reference(url, link[0]):
            yield link


def crawl(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement

    queue = PriorityQueue()
    queue.put((0, root))
    visited = []
    extracted = []
    if within_domain:
        domain = urlparse(root).netloc

    while not queue.empty():
        url = queue.get()[1]
        print(url)
        try:
            req = request.urlopen(url)
            if req.headers['Content-Type'] not in wanted_content: # will always continue if no content-type passed in
                 continue
            html = req.read()

            visited.append(url)
            visitlog.debug(url)

            for ex in extract_information_1(url, html):
                extracted.append(ex)
                extractlog.debug(ex)

            for weight, link, title in parse_links_sorted(url, html):
                if not_self_reference(link, url) and (link not in visited):
                    if not within_domain:
                        queue.put((weight, link))
                    else:
                        if (urlparse(link).netloc in domain) or (domain in urlparse(link).netloc): # check if is in the domain; consider cs.jhu.edu and www.cs.jhu.edu as the same domain
                            queue.put((weight, link))

        except Exception as e:
            print(e, url)

    return visited, extracted


def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''
    results = []

    for match in re.findall('\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        results.append((address, 'PHONE', match))
    for match in re.findall(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", str(html)):
        results.append((address, 'EMAIL', match))
    tar = str(html).replace('.', '')
    for match in re.findall(r"\w+,\s\w+\s\d{5}", tar):
        results.append((address, 'ADDRESS', match))
    return results


def extract_information_1(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''
    results = []

    for match in re.findall('\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        results.append((address, 'PHONE', match))
    for match in re.findall(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", str(html)):
        results.append((address, 'EMAIL', match))
    tar = str(html).replace('.', '')
    for match in re.findall(r"\w+,\s\w+\s\d{5}|\w+\s\w+,\s\w+\s\d{5}|\w+\s\w+\s\w+,\s\w+\s\d{5}", tar):
        ad = match.replace(',', '')  # remove all commas to avoid noise
        ad = ad.lower()
        temp = ad.split()  # split address
        test = ''
        for i in range(len(temp) - 2):
            test = test + temp[i]
        if test in cities:
            results.append((address, 'ADDRESS', match))
    return results


def writelines(filename, data):
    with open(filename, 'w', encoding='utf-8') as fout:
        for d in data:
            print(d, file=fout)


def main():
    site = 'https://cs.jhu.edu/~winston/ir/hw4test.html'
    '''''
    links = get_links(site)
    writelines('links.txt', links)

    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)

    '''''
    visited, extracted = crawl(site, wanted_content=['text/html', 'pdf'], within_domain=True)
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)


if __name__ == '__main__':
    main()
