import csv
from urllib import parse, request


def syn_expansion(word):
    ''' given a word, returns equivalent calss of synonyms '''
    try:
        url = "https://www.thesaurus.com/browse/" + word
        req = request.urlopen(url)
        html = req.read().decode('utf-8')
        s = html.find('"synonyms":[')
        e = html.find('"antonyms":[')
        sel = html[s:e]
        proc_sel = sel.split('"term":"')[1:]
        eq_class = [word]
        
        for a_word_info in proc_sel:
            term = a_word_info.split(',')[0][0:-1]
            eq_class.append(term)
            
        return eq_class
    except:
        print(word)
        return [word]

def main():
    syn = list()
    comp_syn = list()
    with open('symptom_list') as f: 
        for line in f:
            if ' ' not in line:
                syn.append(line)
            else:
                comp_syn.append(line)

    res = []
    for a_syn in syn:
        a_syn = a_syn.rstrip()
        eq_class = syn_expansion(a_syn)
        #if len(eq_class) != 0:
        res.append(eq_class)


    with open('expanded_symptoms', 'w') as f:
        for eq in res:
            for element in eq:
                f.write("%s, "%element)
            f.write('\n')
        for cs in comp_syn:
            f.write("%s\n"%cs)

if __name__ == '__main__':
    main()
    
