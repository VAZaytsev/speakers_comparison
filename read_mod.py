import numpy as np

# ===================================================================
def read_speeches(fl_inp):
    emb_arr = []
    labels_arr = []
    for l in fl_inp.readlines():
        l = l.strip()
        emb = np.load(l)
        emb_arr.append(emb)

        l = l.replace('speeches/','')
        lbl = int(l.split('-')[0])
        labels_arr.append(lbl)


    # number of speeches
    n_speeches = len(emb_arr)


    # concatenate data
    emb = np.concatenate(emb_arr)


    # total number of data
    n_data = emb.shape[0]
    #print('number of data = ', n_data)


    # Number of speakers/labels (output neurons)
    n_S = len(set(labels_arr))
    #print('number of speakers = ', n_S)


    # Create array with labels
    labels = np.zeros(n_data)


    # dictionary, which maps old labels to range(n_S)
    convert_lbls = dict(zip(sorted(set(labels_arr)), range(n_S)))


    # convert old labels to range(n_S)
    indx = 0
    for i,e in enumerate(emb_arr):
        add = e.shape[0]
        labels[indx:indx+add] = convert_lbls[labels_arr[i]]
        indx += add
    
    return n_S, n_speeches, emb, labels
# ===================================================================


# ===================================================================
def read_model(fl_model, fl_speeches):
    #read file with data for the model
    known_speeches = []
    known_speakers = []
    for l in fl_model.readlines():
        l = l.replace(' ','').strip()

        key, val = l.split('=')

        if key == 'model_name':
            model_name = val
        
        if key == 'thres':
            thres = float(val)
            
        if key == 'speech':
            val = val.replace('speeches/','').strip()

            known_speeches.append(val)
            known_speakers.append(val.split('-')[0])


    print('Known speakers are', sorted(list(set(known_speakers))) )
    # file with speeches
    speakers = []
    speeches = []
    indx_i_f = []

    counter = 0
    labels = []
    for l in fl_speeches.readlines():
        l = l.replace(' ','').strip()

        speach = np.load(l)
        speeches.append( speach )
        
        l = l.replace('speeches/','').strip()
        speaker = l.split('-')[0]
        speakers.append(speaker)

        indx_i_f.append( (counter,counter+speach.shape[0]-1) )

        counter += speach.shape[0]

        name = l.replace('.npy','')

        #add zeros to numbers
        x,y,z = name.split('-')
        name = x.zfill(4) + '-' + y.zfill(6) + '-' + z
        
        if l in known_speeches:
            lbl = '\033[42m' + name.split('-')[0] + '\033[0m' \
                + ''.join(['-'+'\033[42m'+x+'\033[0m' for x in name.split('-')[1:]])
        else:
            if speaker in known_speakers:
                lbl = '\033[42m' + name.split('-')[0] + '\033[0m' \
                    + ''.join(['-'+x for x in name.split('-')[1:]])
            else:
                lbl = name
        print(lbl)
        labels.append(lbl)

    return model_name, thres, speeches, indx_i_f, labels
# ===================================================================
