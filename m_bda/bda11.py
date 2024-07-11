H = {'gone':[[1,8],[4,5]],
     'with':[[2,7],[3,6]],
     'the':[[3,6],[2,4]],
     'wind':[[4,5],[1,3]],
     'mind':[[5,4],[8,7]],
     'if':[[6,3],[7,8]],
     'i':[[7,2],[6,2]],
     'am':[[8,1],[5,1]]
     }

q = ['gone', 'with','the','wind']
x1 = ['gone', 'with', 'wind']
x2 = ['gone', 'with','the','mind']
x3 = ['mind', 'if', 'i', 'am', 'gone']
x4 = ['if', 'mind', 'gone', 'with', 'wind'] 

k = 2
l = 2

def jaccard_similarity(a,b):
    a = set(a)
    b = set(b)
    return len(a.intersection(b))/len(a.union(b))

def minhash(a):
    h = [[10]*k]*l
    for word in a:
        for i in range(k):
            for j in range(l):
                h[i][j] = min(h[i][j],H[word][i][j])
    return h

def similarity(a,b):
    return a[0] == b[0] or a[1] == b[1]

TP = []
FP = []
FN = []
TN = []
for x in [x1,x2,x3,x4]:
    if similarity(minhash(q),minhash(x)):
        if jaccard_similarity(q,x) > 0.6:
            TP.append(x)
        else:
            FP.append(x)
    else:
        if jaccard_similarity(q,x) > 0.6:
            FN.append(x)
        else:
            TN.append(x)

print("True Positives:",TP)
print("False Positives:",FP)
print("False Negatives:",FN)
print("True Negatives:",TN)