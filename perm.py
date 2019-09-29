from query_engine import *
import random
import matplotlib.pyplot as pl
import numpy as np

q_eq = [[2, 94, 73, 82], [3, 6, 63, 92, 78, 123], [4, 113], [52, 6, 78], [64, 97, 121, 112], [5, 18, 31, 51], [27, 41, 61], [80, 35, 86], [10, 60, 76, 28, 119], [7, 74, 79], [15, 75], [107, 108], [62, 66, 22, 20], [118, 100, 87, 50], [46, 68, 37, 67], [24, 56], [126, 83], [40, 103], [72], [39, 55, 70, 89, 85, 10], [17, 110], [106], [71, 101], [36], [32], [11, 65, 26], [29], [115, 44, 111, 120, 38], [9, 117, 91, 45]]


def improve_user_vector(x, dv):
    eq_summary = []
    for i in range(len(raw_disease_vector[0])): # 127
        eq_vals = []
        for j in range(len(raw_disease_vector)): # 175
            eq_vals.append(raw_disease_vector[j][i])
        eq_summary.append(eq_vals)

    median = np.median(eq_summary, axis=1)
    twentyfifth = np.percentile(eq_summary, 25, axis=1)
    seventyfifth = np.percentile(eq_summary, 75, axis=1)

    for idx, val in enumerate(x):
        if val == 0:
            res[idx] = 0
        elif (val > 0.3 and val < 0.4):
            res[idx] = twentyfifth[idx]
        elif (val > 0.4 and val < 0.7):
            res[idx] = median[idx]
        else:
            res[idx] = seventyfifth[idx]
    return res

svd_files = [25, 30, 35, 40, 60, 80, 100]
for k in range(4):
    dn_vec = []
    v_vec = []
    disease_vector = []
    line = ''
    with open('svdresult'+str(svd_files[k])+'.txt','r') as fid:
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

    out = []
    while(len(out) < 1000):
        temp = []
        for i in range(28 - 4):
            temp.append(random.randint(0,3)) # randomly generate integers from [0, 3]
        for i in range(4):
            temp.append(random.randint(1,2))

        # to exactly mimic the command line argument
        inp_arr = str(temp).replace(' ','')
        inp_arr = inp_arr.replace('[','')
        inp_arr = inp_arr.replace(']','')

        
        norm_inp = user_inp(inp_arr)
        res = [x-x for x in range(127)]
        for n in range(28):
            for eq in q_eq[n]: # each eq class corresponding to q n   
                res[eq] += norm_inp[n]
        norm_inp = improve_user_vector(norm_inp, disease_vector)
        
        similarity = list()

        for idx, v in enumerate(disease_vector):
            similarity.append(cosine_sim(v, res))

        val, idx = max((val, idx) for (idx, val) in enumerate(similarity))
        disease_name = disease_list[idx]
        out.append((disease_name, res, val)) # (diagosis result, vector model value, maximum similarity)
        # the 3 vectors below are for plotting purposes
        dn_vec.append(disease_name)
        v_vec.append(val)

    with open('random_results'+str(svd_files[k])+'.txt','w') as f:
        for l in out:
            f.write(str(l[0]) + '\t' + str(l[1]) + '\t' + str(l[2]) + '\n')
    f.close()
    
    freq = {}
    for d in disease_list:
        freq[d] = 0
    for dn in dn_vec:
        freq[dn] += 1

    with open('dn'+str(svd_files[k])+'.csv','w') as f:
        f.write('DiseaseName,Frequency,\n')
        for d in freq.keys():
            f.write(d + ','+ str(freq[d]) +',\n')
    f.close()


    with open('v'+str(svd_files[k])+'.csv','w') as f:
        for l in out:
            f.write(str(l[2]) + ',\n')
    f.close()

    