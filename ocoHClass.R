rm(list=ls())

# ----------------------------------------------
# IMPORT DATA
# ----------------------------------------------

data   = read.csv("input-data-labeling.csv") # data
labels = read.csv("input-data-labels.csv")   # labels
swords = read.csv("stop-words-list.csv")     # stop-word list (from http://www.ranks.nl/stopwords
swords = as.character(swords[,1])            # and considering only these languages: DE,EN,FR,IT,NL)

# Note: we have eliminated from the data file two products (with ID 76578631 and ID 133908997)
# as they have neither title nor description. Otherwise they generate an error in the code.

# ----------------------------------------------
# CREATE TRAIN AND TEST DATASETS
# ----------------------------------------------

n.points            = length(data[,1])       # total number of products
sampling.rate       = 0.8
num.test.set.labels = n.points*(1-sampling.rate)

training = sample(1:n.points, sampling.rate*n.points, replace=FALSE)           # indices of training data
train    = subset(data[training,], select = c('name','description','labels'))  # training data
testing  = setdiff(1:n.points,training)                                        # indices of test data
test     = subset(data[testing,], select = c('name','description'))            # test data

# ----------------------------------------------
# FEATURE PROCESSING (TRAINING)
# ----------------------------------------------

# TRAIN Data Set
dTrain = data.frame(features=NULL,labels=NULL,stringsAsFactors=FALSE) # create Train data frame
dTr    = 5000#length(training)                                             # number of train products
hh     = 1

for(h in seq(1,dTr)){
  
  x1 = as.character(train[h,1]) # product title
  x1 = strsplit(x1," ")
  x1 = unlist(x1)
  
  y1 = as.character(train[h,2]) # product description
  y1 = strsplit(y1," ")
  y1 = unlist(y1)
  
  z1 = c(x1,y1)                 # union between title and description
  
  a = gsub("[^[:alnum:][:space:]']", "", z1)    # remove punctuation
  a = tolower(a)                                # lower case
  a = gsub('[[:digit:]]+', '', a)               # remove numbers
  a = gsub("\\b[a-zA-Z0-9]{1,2}\\b", "", a)     # remove words shorter than 3
  a = a[!a%in%""]                               # remove empty words
  a = unique(a)                                 # remove dopple words
  for(i in seq(1,length(swords))){              # remove stop words
    for(j in seq(1,length(a))){
      if(swords[i]==a[j]){
        a[j] = ""
        break
      }
    }
  }
  a = a[!a%in%""]
  
  # STEMMING of words would be possible [wordStem(.)], but first we need to detect the language
  # of the product or at least have a label for the language, which is not present in the data
  
  if(length(a)>16){   # if the description contains too many words, then the result will be
    a[seq(1,15)]      # biased (too many scores for this product). Solution: we assume that
  }                   # most of the important features will be at the beginning of the vector
  
  b = unlist(strsplit(as.character(train[h,3]),","))  # labels
  
  aDf = paste0(a,collapse = ",")
  bDf = paste0(b,collapse = ",")
  dTrain[hh,1] = aDf
  dTrain[hh,2] = bDf
  hh = hh+1
  
  print(h)
  
}

colnames(dTrain) = c("features","labels")

# ----------------------------------------------

# TEST Data Set
dTest = data.frame(features=NULL,stringsAsFactors=FALSE) # create Test data frame
dTe   = 10#length(testing)                                  # number of test products

for(h in seq(1,dTe)){ # (replace dTe --> 1000 for shorter run)
  
  x1 = as.character(test[h,1]) # product title
  x1 = strsplit(x1," ")
  x1 = unlist(x1)
  
  y1 = as.character(test[h,2]) # product description
  y1 = strsplit(y1," ")
  y1 = unlist(y1)
  
  z1 = c(x1,y1)                # union between title and description
  
  a = gsub("[^[:alnum:][:space:]']", "", z1)    # remove punctuation
  a = tolower(a)                                # lower case
  a = gsub('[[:digit:]]+', '', a)               # remove numbers
  a = gsub("\\b[a-zA-Z0-9]{1,2}\\b", "", a)     # replace words shorter than 3
  a = a[!a%in%""]                               # remove empty words
  a = unique(a)                                 # remove dopple words
  for(i in seq(1,length(swords))){              # remove stop words
    for(j in seq(1,length(a))){
      if(swords[i]==a[j]){
        a[j] = ""
        break
      }
    }
  }
  a = a[!a%in%""]
  
  aDf = paste0(a,collapse = ",")
  dTest[h,1] = aDf
  
  # Note (1): as we've removed the punctuation, we have replaced in the data file the character "-"
  # with a null character (e.g. t-shirt --> tshirt, v-pullover --> vpullover).
  # Note (2): as we've removed words with length < 3, in data file we've done these replacements:
  # ["BH " --> "BHH "], [" BH" --> " BHH"], to avoid problems with Buestenhalter.
  # Note (3): there will be problems with 3/4 Hosen and 7/8 Hosen, as we have removed numbers.
  
  print(h)
  
}

colnames(dTest) = c("features")

# ----------------------------------------------
# TEST
# ----------------------------------------------

KK = 7                                                        # number of NN
res = data.frame(score="0",index="0",stringsAsFactors=FALSE)  # data frame of results

for(nn in seq(1,length(dTest[,1]))){
  
  X = unlist(strsplit(as.character(dTest[nn,1]),","))
  
  # compare features of a test product with features of all train products: count how many times
  # a match is present [we consider two times the intersection of pmatch(X,Y) and pmatch(Y,X)]
  vet = array(0,length(dTrain[,1]))
  for(i in seq(1,length(dTrain[,1]))){
    Y = unlist(strsplit(as.character(dTrain[i,1]),","))
    vet[i] = vet[i] + sum(as.integer(!is.na(pmatch(Y,X))))+sum(as.integer(!is.na(pmatch(X,Y))))    
  }
  
  # sort scores and retrieve correspondent index in Train data set
  Z = sort(vet, decreasing=TRUE, index.return=TRUE)
  
  res[nn,1] = paste0(Z$x[seq(1,KK)],collapse=" ")
  res[nn,2] = paste0(Z$ix[seq(1,KK)],collapse=" ")
  
  print(nn)
  
}

# res: row index --> index of test product
#      score     --> number of partial matches for the first KK high-score train products
#      index     --> index of the first KK high-score train products

# ----------------------------------------------
# POSTPROCESSING and RESULT SUMMARY
# ----------------------------------------------

Result  = array(0,dim=c(2,length(dTest[,1])))
NullTot = 0                                     # counts the NULL-NULL cases
DoppleM = array(0,dim=c(1,length(dTest[,1])))   # becomes 1 if at level 4 there are two labels

for(nn in seq(1,length(dTest[,1]))){
  
  index = as.integer(unlist(strsplit((as.character(res[nn,2]))," "))) # indices of the KK train products
  
  # (for a test product) for each of its first KK high-score train product, check their labels
  # and put them in list resList, according to their level (l1,l2,l3,l4)
  resList = list(l1=NULL,l2=NULL,l3=NULL,l4=NULL)
  for(i in seq(1,KK)){
    labList  = unlist(strsplit(as.character(dTrain[index[i],2]),","))
    levList  = labels[match(labList,labels[,2]),3];
    for(j in seq(1,length(levList))){
      if(levList[j]==1){
        resList$l1[length(resList$l1)+1] = labList[j]
      }
      if(levList[j]==2){
        resList$l2[length(resList$l2)+1] = labList[j]
      }
      if(levList[j]==3){
        resList$l3[length(resList$l3)+1] = labList[j]
      }
      if(levList[j]==4){
        resList$l4[length(resList$l4)+1] = labList[j]
      }      
    }  
  }
  
  # (for a test product) check frequency of predicted labels, sort them in each level and take:
  # first label for l1, first two labels for l2, first three labels for l3, first label for l4.
  # List finLab contains the final predicted labels (at each level) for the test product
  finLab = list(l1=NULL,l2=NULL,l3=NULL,l4=NULL)
  temp <- table(as.vector(resList$l1))
  finLab$l1 = names(sort(temp,decreasing=TRUE))[seq(1,1)] # one label for level 1
  temp <- table(as.vector(resList$l2))
  finLab$l2 = names(sort(temp,decreasing=TRUE))[seq(1,2)] # two labels for level 2
  temp <- table(as.vector(resList$l3))
  finLab$l3 = names(sort(temp,decreasing=TRUE))[seq(1,3)] # three labels for level 3
  if(is.null(resList$l4)){
    finLab$l4 = NULL
  } else{
    temp <- table(as.vector(resList$l4))
    finLab$l4 = names(sort(temp,decreasing=TRUE))[seq(1,2)] # one label for level 4
  }                                                         # [for 2 labels use: seq(1,2)]
  
  # create list trueList: list containing the true labels (at each level) for the Test data set
  trueList = list(l1=NULL,l2=NULL,l3=NULL,l4=NULL)
  trueLab = unlist(strsplit(as.character(data[testing[nn],7]),","))
  levTrue  = labels[match(trueLab,labels[,2]),3];
  for(j in seq(1,length(levTrue))){
    if(levTrue[j]==1){
      trueList$l1[length(trueList$l1)+1] = trueLab[j]
    }
    if(levTrue[j]==2){
      trueList$l2[length(trueList$l2)+1] = trueLab[j]
    }
    if(levTrue[j]==3){
      trueList$l3[length(trueList$l3)+1] = trueLab[j]
    }
    if(levTrue[j]==4){
      trueList$l4[length(trueList$l4)+1] = trueLab[j]
    }
  }
  
  # compare predicted labels with true labels at level 2
  Result[1,nn] = sum(as.integer(!is.na(match(finLab$l2,trueList$l2))))
  
  # compare predicted labels with true labels at level 4
  if((is.null(finLab$l4))&(is.null(trueList$l4))){
    # case: NULL-NULL (it's a match: count as 1)
    Result[2,nn] = 1
    NullTot      = NullTot+1
  } else {
    # case: NULL-something (count as 0)
    # case: something-NULL (count as 0)
    # case: something-something (count as 0 or 1, if it's a match)
    Result[2,nn] = sum(as.integer(!is.na(match(finLab$l4,trueList$l4))))
  }
  
  if(length(finLab$l4)>1){
    DoppleM[nn]=1
  }
    
}

# percentage that at least one level 2 label is correctly predicted
Rate_l2 = sum(as.integer(Result[1,]>0))/length(Result[1,])

# percentage that at least one level 4 label is correctly predicted
Rate_l4a = sum(as.integer(Result[2,]>0))/length(Result[2,])

# number of products where only one label (out of two) is correctly predicted
single_l4 = sum(as.integer((Result[2,DoppleM>0])==1))

# no wrong labelling
noWrong_l4 = (sum(as.integer(Result[2,]>0))-single_l4)/length(Result[2,])

print(Rate_l2)
print(Rate_l4a)
print(Rate_l4b)
print(noWrong_l4)
