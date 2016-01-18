import pandas as pd
import numpy  as np
import string
from   collections import Counter

# -----------------------------------------------------------------------------
# IMPORT DATA
# -----------------------------------------------------------------------------

data   = pd.read_csv('input-data-labeling.csv')
labels = pd.read_csv('input-data-labels.csv')
swords = pd.read_csv('stop-words-list.csv')
swords  = list(swords.loc[np.arange(0,len(swords)),'stop'])

# -----------------------------------------------------------------------------
# CREATE TRAIN AND TEST DATASETS
# -----------------------------------------------------------------------------

nPoint = data.shape[0]
sRate  = 0.8

training = np.random.choice(np.arange(0,nPoint,1), size = int(np.floor(nPoint*sRate-1)), replace = 0)
train    = data.loc[training,['name','description','labels']]
nTrain   = len(training)
train    = train.set_index(np.arange(nTrain))

testing  = np.setdiff1d(np.arange(0,nPoint),training)
test     = data.loc[testing,['name','description']]
nTest    = len(testing)
test     = test.set_index(np.arange(nTest))


# -----------------------------------------------------------------------------
# FEATURE PROCESSING (TRAINING)
# -----------------------------------------------------------------------------

nTrain = 10000
nTest = 50

# TRAIN Data Set
dTrain = pd.DataFrame(np.zeros([nTrain,2]), columns=['features', 'labels'])

for h in range(0, nTrain):
    
    # lower case
    x1 = train.loc[h,'name'].lower()
    y1 = train.loc[h,'description'].lower()
    
    # remove numbers, punctuation
    for c in (string.punctuation+string.digits):
        x1=x1.replace(c,"")
        y1=y1.replace(c,"")

    # remove short words
    x1 = ' '.join(w for w in x1.split() if len(w)>2)
    y1 = ' '.join(w for w in y1.split() if len(w)>2)

    # remove non-unique words
    x1 = list(set(x1.split()))                       
    y1 = list(set(y1.split()))
    
    # remove stop-words
    x1 = ' '.join(w for w in x1 if w not in swords)
    y1 = ' '.join(w for w in y1 if w not in swords)

    # use only first 15 words
    z1 = (x1.split() + y1.split())[0:15]

    dTrain.loc[h:h:1,'features'] = ','.join(z1)
    dTrain.loc[h:h:1,'labels']   = train.loc[h,'labels']
    
    print(h,'out of',nTrain)
    
# -----------------------------------------------------------------------------

# TEST Data Set
dTest = pd.DataFrame(np.zeros([nTest,1]), columns=['features'])

for h in range(0, nTest):
    
    # lower case
    x1 = test.loc[h,'name'].lower()    
    y1 = test.loc[h,'description'].lower()
    
    # remove numbers, punctuation
    for c in (string.punctuation+string.digits):
        x1=x1.replace(c,"")
        y1=y1.replace(c,"")

    # remove short words
    x1 = ' '.join(w for w in x1.split() if len(w)>2)
    y1 = ' '.join(w for w in y1.split() if len(w)>2)

    # remove non-unique words
    x1 = list(set(x1.split()))                       
    y1 = list(set(y1.split()))
    
    # remove stop-words
    x1 = ' '.join(w for w in x1 if w not in swords)
    y1 = ' '.join(w for w in y1 if w not in swords)

    # use only first 15 words
    z1 = (x1.split() + y1.split())[0:15]

    dTest.loc[h:h:1,'features'] = ','.join(z1)
    
    print(h,'out of',nTest)
    
# -----------------------------------------------------------------------------
# LEARNING
# -----------------------------------------------------------------------------

KK = 7   # number of NN

# data frame of results
res = pd.DataFrame(np.zeros([nTest,2]), columns=['score','index'])

for nn in range(0,nTest):
    
    Xsp = dTest.loc[nn,'features'].split(',')
    X   = str(dTest.loc[nn,'features'])

    vet = np.zeros(nTrain)
    
    for i in range(0,nTrain):
        
        Ysp = dTrain.loc[i,'features'].split(',')
        Y   = str(dTrain.loc[i,'features'])
        
        for j in range(0,len(Xsp)):
            vet[i] = vet[i] + int(Y.find(Xsp[j])>0)
        for j in range(0,len(Ysp)):
            vet[i] = vet[i] + int(X.find(Ysp[j])>0)
                            
    ix = vet.argsort()[::-1]
    vet.sort()
    x = vet[::-1]
    
    res.loc[nn:nn:1,'score'] = ''.join(str(x[0:KK:1]))
    res.loc[nn:nn:1,'index'] = ''.join(str(ix[0:KK:1]))
    
    print(nn,'out of',nTest)


# res: row index --> index of test product
#      score     --> number of partial matches for first KK train products
#      index     --> index of the first KK high-score train products
  
# -----------------------------------------------------------------------------  
# EXPORT CSV FILES FOR A POSSIBLE FAST IMPLEMENTATION OF THE 'LEARNING' IN C++
# -----------------------------------------------------------------------------

CTrain = pd.DataFrame(np.zeros([nTrain,1]), columns=['features']) # no labels
CTrain.loc[0:nTrain-1:1,'features'] = dTrain.loc[0:nTrain-1:1,'features']
CTrain.to_csv("Train.csv",sep='\t',encoding='utf-8',header=False,index=False)
dTest.to_csv("Test.csv",sep='\t',encoding='utf-8',header=False,index=False)

# -----------------------------------------------------------------------------  
# POSTPROCESSING AND RESULT SUMMARY
# -----------------------------------------------------------------------------

Result  = np.zeros([2,nTest])
NullTot = 0                          # counts the NULL-NULL cases
DoppleM = np.zeros([1,nTest])        # 1 if at level 4 there are two labels
listv   = list(labels.loc[:,'name']) # list of labels

for nn in range(0,nTest):
    
    index = res.loc[0,'index']
    index = index.replace("[","")
    index = index.replace("]","")
    index = index.split()
    index = [int(i) for i in index]  # indices of the KK train products

    # (for a test product) for each of its first KK high-score train product,
    # check their labels and put them in list resList, according to their level
    resList1 = []; resList2 = []; resList3 = []; resList4 = []
    for i in range(0,KK):
        
        labList = dTrain.loc[index[i],'labels'].split(',')
        
        levList = np.zeros(len(labList))
        for k in range(0,len(labList)):
            levList[k] = labels.loc[listv.index(labList[k]),'level']
            
        for j in range(0,len(levList)):
            if(levList[j]==1):
                resList1.append(labList[j])
            if(levList[j]==2):
                resList2.append(labList[j])
            if(levList[j]==3):
                resList3.append(labList[j])
            if(levList[j]==4):
                resList4.append(labList[j])
                
    # (for a test product) check frequency of predicted labels, sort them in
    # each level and take: first label for l1, first two labels for l2, first
    # three labels for l3, first label for l4. List finLab contains the final
    # predicted labels (at each level) for the test product
    finLab1 = []; finLab2 = []; finLab3 = []; finLab4 = []
    counts = Counter(resList1); counts = counts.most_common()
    finLab1.append(counts[0][0]); # one label for level 1
    counts = Counter(resList2); counts = counts.most_common()
    finLab2.append(counts[0][0]); # two labels for level 2
    finLab2.append(counts[1][0]);
    counts = Counter(resList3); counts = counts.most_common()
    finLab3.append(counts[0][0]); # three labels for level 3
    finLab3.append(counts[1][0]);
    finLab3.append(counts[2][0]);
    if(not resList4):
        finLab4 = []
    else:
        counts = Counter(resList4); counts = counts.most_common()
        finLab4.append(counts[0][0]);
    
    # create list trueList: list containing the true labels (at each level)
    # for the Test data set    
    trueList1 = []; trueList2 = []; trueList3 = []; trueList4 = []
    trueLab   = data.loc[testing[nn],'labels'].split(',')
    
    levTrue = np.zeros(len(trueLab))
    for k in range(0,len(trueLab)):
        levTrue[k] = labels.loc[listv.index(trueLab[k]),'level']
            
    for j in range(0,len(levTrue)):
        if(levTrue[j]==1):
            trueList1.append(trueLab[j])
        if(levTrue[j]==2):
            trueList2.append(trueLab[j])
        if(levTrue[j]==3):
            trueList3.append(trueLab[j])
        if(levTrue[j]==4):
            trueList4.append(trueLab[j])
    
    # compare predicted labels with true labels at level 2
    for j in range(0,len(finLab2)):
        Result[0,nn] = Result[0,nn] + trueList2.count(finLab2[j])
     
    # compare predicted labels with true labels at level 4
    if((not finLab4)&(not trueList4)):
        # case: NULL-NULL (it's a match: count as 1)
        Result[1,nn] = 1
        NullTot      = NullTot+1
    else:
        # case: NULL-something (count as 0)
        # case: something-NULL (count as 0)
        # case: something-something (count as 0 or 1, if it's a match)
        for j in range(0,len(finLab4)):
            Result[1,nn] = Result[1,nn] + trueList4.count(finLab4[j])
            
    if(len(finLab4)>1):
        DoppleM[0,nn] = 1

# percentage that at least one level 2 label is correctly predicted
Rate_l2 = 0 
for i in range(0,len(Result[0,:])):
    if(Result[0,i]>0):
        Rate_l2 = Rate_l2 + 1
Rate_l2 = Rate_l2/len(Result[0,:])

# percentage that at least one level 4 label is correctly predicted
Rate_l4a = 0 
for i in range(0,len(Result[0,:])):
    if(Result[1,i]>0):
        Rate_l4a = Rate_l4a + 1
Rate_l4a = Rate_l4a/len(Result[0,:])
