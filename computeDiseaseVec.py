from defaultlist import defaultlist
import numpy as np

dlist = []
with open('disease_list2') as f:
    for l in f:
        dlist.append(l.strip('\n'))

print(dlist)

f2 = open('disease_list_final.txt','w')
for l in dlist:
    f2.write(l+'\n')


dNum = defaultlist(int)
with open('disease_vector.csv') as f:
    for l in f:
        dname = l.split('[')[0][:-1]
        indexOfDisease = dlist.index(dname)
        dNum[indexOfDisease] += 1

dveclist = defaultlist()
with open('disease_vector.csv') as f2:
    for l in f2:
        dname = l.split('[')[0][:-1]
        indexOfDisease = dlist.index(dname)
        if dveclist[indexOfDisease] == None:
            dveclist[indexOfDisease] = defaultlist(int)
        scorelist = l.split('[')[1].split(',')
        if scorelist[len(scorelist)-1] == '\n':
            scorelist = scorelist[:-1]
        for ind in range(len(scorelist)):
            if ind == len(scorelist)-1:
                if scorelist[ind].endswith(']\n'):
                    scorelist[ind]=scorelist[ind][:-2]
                if scorelist[ind].endswith(']'):
                    scorelist[ind]=scorelist[ind][:-1]
            if not int(scorelist[ind]) == 0:
                dveclist[indexOfDisease][ind] += 1/dNum[indexOfDisease]


f3 = open('disease_vector_list.txt','w')

for ind in range(len(dveclist)):
    if dveclist[ind] == None:
        dveclist[ind] = defaultlist()
    if not len(dveclist[ind]) == 127:
        for k in range(len(dveclist[ind]),127):
            dveclist[ind][k] = 0

print(dveclist[80])

for v in dveclist:
    f3.write(str(v)+'\n')
