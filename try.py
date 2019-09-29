
import matplotlib.pyplot as pl
import numpy as np

disease_vector = []
line = ''
with open('svdresult25.txt','r') as fid:
    line = fid.readline()
fid.close()
line = line.split('], [') # convert to vector of strings
line[0] = line[0][2:] # processing the beginning
line[-1] = line[-1][:-2] # processing the end
for l in line: 
    temp = []
    line_vec = l.split(', ')
    for lv in line_vec:
        temp.append(float(lv))
    disease_vector.append(temp)

eq_summary = []
for i in range(len(disease_vector[0])): # 127
    eq_vals = []
    for j in range(len(disease_vector)): # 175
        eq_vals.append(disease_vector[j][i])
    eq_summary.append(eq_vals)

median = np.median(eq_summary, axis=1)
twentyfifth = np.percentile(eq_summary, 25, axis=1)
seventyfifth = np.percentile(eq_summary, 75, axis=1)
print(str(twentyfifth[0]) + ', ' + str(median[0])+', ' + str(seventyfifth[0]))
print(str(twentyfifth[1]) + ', ' + str(median[1])+', ' + str(seventyfifth[1]))
print(str(twentyfifth[-2]) + ', ' + str(median[-2])+', ' + str(seventyfifth[-2]))
print(str(twentyfifth[-1]) + ', ' + str(median[-1])+', ' + str(seventyfifth[-1]))
print(len(median))