import numpy as np

# cosine distance
def cos_dist(a, b):
    num = a.dot(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den


# Subroutine to compute EER
def EER(features, labels):
    same_arr = []
    cos_arr = []

    n = len(labels)

    for i in range(n):
        for j in range(i,n):
            same = int( labels[i] == labels[j] )
            dist = cos_dist(features[i,:], features[j,:])

            same_arr.append(same)
            cos_arr.append(dist)


    # Sort for simple extraction of the EER
    tot = list(zip(same_arr,cos_arr))
    tot = sorted(tot, key=lambda x: x[1])


    # Threshold is always set between cos_arr[i] and cos_arr[i+1]
    EER = 1
    eps = 1


    # The smallest meaningful threshold
    TA = 0
    FR = 0
    FA = sum([1-tot[i][0] for i in range(len(tot))])

    for i in range( len(tot)-1 ):
        thres = 0.5*(tot[i][1] + tot[i+1][1])

        TA += 1

        FR += tot[i][0]
        FA -= (1-tot[i][0])

        FRR = FR / TA
        FAR = FA / (len(tot) - TA)

        if abs(FAR - FRR) < eps:
            eps = abs(FAR - FRR)
            EER = 0.5*(FAR + FRR)
            thres_bst = thres

    return EER, thres_bst
