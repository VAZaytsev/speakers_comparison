import tensorflow as tf
import numpy as np
import sys

import EER_mod
import read_mod

# provide the file with the speeches passed trough openl3 in .npy format
fl_model = open(sys.argv[1],'r')
fl_speeches = open(sys.argv[2],'r')

model_name, thres, speeches, indx_i_f, labels = read_mod.read_model(fl_model, fl_speeches)


# load model
model = tf.keras.models.load_model(model_name)


# Concatenate speeches and create feature vectors for them
data = np.concatenate(speeches)
features = model.predict(data, verbose=0)


# Average features
# Alternatively, one can try to compare the averaged outputs from openl3
n_sp = len( speeches )

av_emb = []
for i in range(n_sp):
    av_emb.append( np.average(features[indx_i_f[i][0]:indx_i_f[i][1]+1,:],axis=0) )


# Compare speakers
print('\nThreshold = ', f'{thres: .3f}')
for i in range(n_sp):
    for j in range(i+1, n_sp):
        cos_av = EER_mod.cos_dist(av_emb[i], av_emb[j])
        
        answer = cos_av > thres
        correct_answer = labels[i].split('-')[0] == labels[j].split('-')[0]

        if answer == correct_answer:
            if answer:
                print(labels[i], labels[j], f'{cos_av: .3f}', 'same')
            else:
                print(labels[i], labels[j], f'{cos_av: .3f}', 'diff')
        else:
            if answer:
                print(labels[i], labels[j], f'{cos_av: .3f}', 'same', 
                      '\033[97;41m' + 'WRONG!!!' + '\033[0m')
            else:
                print(labels[i], labels[j], f'{cos_av: .3f}', 'diff', 
                      '\033[97;41m' + 'WRONG!!!' + '\033[0m')
