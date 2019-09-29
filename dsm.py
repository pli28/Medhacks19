import csv
from urllib import parse, request

def main():
    try:
        url = "https://www.thesaurus.com/browse/" + word
        req = request.urlopen(url)
        html = req.read().decode('utf-8')
        s = html.find('"synonyms":[')
        e = html.find('"antonyms":[')
        sel = html[s:e]
        proc_sel = sel.split('"term":"')[1:]
        eq_class = [word]

    except:
        print


if __name__ == '__main__':
    main()
