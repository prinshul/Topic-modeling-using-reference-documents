library(rJava)
library(tm)
library(NLP)
library(openNLP)
library(qdap)
library(slam)
library(HMM)
library(lsa)
library(cluster)
library(plyr)
library(SpecsVerification)
library(lda)

words_per_sent<-3
truth<-list()
k<-20
setwd("document_directory/") # location containing corpus
targetdest<-("target_directory/") # location having docs used to create target document
topics<-list() # list of topics
sentencespertopic<-list() # number of sentences per topic
targetfiles<-list()  # list of files used to create target document
lookupTable<-data.frame()  #emission probabilities
for(dir in list.dirs(".",full.names = TRUE)){
  topic<-(strsplit(dir,split='/', fixed=TRUE)[[1]][2])
  if(!is.na(topic))
  {
    topics<-c(topics,topic)
  }
}

for(file in list.files(targetdest,full.names = TRUE)){
    print(file)
    file<-unlist(strsplit(file,split='/', fixed=TRUE))
    file<-file[length(file)]
    targetfiles<-c(targetfiles,file)
}

topics<-unlist(topics)
targetfiles<-unlist(targetfiles)

# clean the corpus- text processing
text_prep<-function(corpus,keepPunct=FALSE){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stemDocument, language = "english")
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  if(keepPunct==TRUE)
  {
    corpus <- tm_map(corpus, content_transformer(strip), char.keep=c("?", ".", "!","|"))
  }
  else
  {
    corpus <- tm_map(corpus, content_transformer(strip))
  }
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}
process<-function(x){
  sent_list<-sent_detect(x$content)
  sentences<-lapply(sent_list, function(x) strip(x)) 
  return(unlist(sentences))
}

plottruth<-function(topic){

plot(c(1:nrow(lookupTable)),(truth==which(topics==topic))^1,xlab="sentences", ylab=c(topic," probability"))
lines(c(1:nrow(lookupTable)),(truth==which(topics==topic))^1)
lines(c(1:nrow(lookupTable)),lookupTable[,which(topics==topic)],col='blue')
lines(c(1:nrow(lookupTable)),posterior[which(topics==topic),],col='red')
legend('topright', legend=c("Truth", "cosine",'HMM'),
       col=c("black", "blue",'red'), lty=1)
}

calSimScore<-function(x,y){
  wordVC <- c(x,y)
  subcorpus <- (VectorSource(wordVC))
  subcorpus <- Corpus(subcorpus)
  tdm <- TermDocumentMatrix(subcorpus)
  cosine_score <- cosine(as.matrix(tdm)[,1],as.matrix(tdm)[,2])
  if(is.nan(cosine_score[1]))
  {
    return (1e-10)  # assign low value if no matching word present
  }
  else
  {
  return (cosine_score[1])
  }
}

cosineSim<- function(x,texts){
  scores<-lapply(texts,FUN = calSimScore,x=x)
  return(scores)
}

# remve sentences containing less than words_per_sent number of words
cut_words_sent<-function(y){
  lens<-which(sapply(gregexpr("[[:alpha:]]+", y), function(x) sum(x > 0))>=words_per_sent)
  y[lens]
  return(y[lens])
}

# remove single charatcer words
rm_single_char_sent<-function(y){
  lens<-which(lapply(y, function(x) nchar(as.character(x)))>1)
  y[lens]
  return(y[lens])
}

files=list.files("./",full.names = TRUE,recursive = TRUE)
corpus <- Corpus(URISource(files))
corpus <- text_prep(corpus,keepPunct = FALSE)
ndocs <- length(corpus)
# ignore extremely rare words i.e. terms that appear in less then 0.2% of the documents
minTermFreq <- ndocs * 0.002
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * 0.5
dtm = DocumentTermMatrix(corpus,
                         control = list(
                           wordLengths=c(3, 15), 
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))

dtm_tfxidf <- weightTfIdf(dtm)  # document term matrix
m <- as.matrix(dtm_tfxidf)
#normalize euclidean distance for k means
norm_eucl <- function(m) m/apply(m, MARGIN=1, FUN=function(x) sum(x^2)^.5)
m_norm <- norm_eucl(m)

# K means
cl <- kmeans(m_norm, k,nstart = 20)
avgdistances<-list()
for(i in 1:k){
g=m_norm[which(cl$cluster==i),]
ng=cl$size[i]
total=sum(as.matrix(dist(rbind(g,cl$centers[i,])))[ng+1,])
avg=total/ng
avgdistances<-c(avgdistances,avg)
}

# find cluster labels. Labels are known topics
clusterlabels<- list()
for(i in 1:k){
  vector<-list()
  for(topic in topics){
    val<-length(which(lapply(names(which(cl$cluster==i)),function(x) startsWith(x,topic))==TRUE))
    vector <- c(vector, val)
  }
  clusterlabels<-c(clusterlabels,topics[which.max(vector)])
  clusterlabels<-unlist(clusterlabels)
}

trans_probab<-data.frame() #transition probabilities
for(i in 1:k){
  row<-list()
  for(j in 1:k){
    if(i!=j)
      val<-(1-dist(rbind(cl$centers[i,],cl$centers[j,])))  # off-diagnol elements
    else
      val<-(1/unlist(avgdistances[i])) # diagonal elements of transition matrix
    row<-c(row,val)
  }
  row_vec<-unlist(row)
  row_vec<-(row_vec^18)/sum(row_vec^18) #increase of weight similarity in a cluster than across the cluster
  trans_probab<-rbind(trans_probab,unlist(row_vec))
}

names(trans_probab)<-clusterlabels
rownames(trans_probab)<-clusterlabels

trans_probab<-trans_probab[ , order(names(trans_probab))]
trans_probab<-trans_probab[order(rownames(trans_probab)),]

corpus <- Corpus(URISource(files))
corpus <- text_prep(corpus,keepPunct = TRUE)

#processed sentences
sentences<-sapply(corpus, FUN=process)
sentences<-lapply(sentences, FUN=cut_words_sent)
#sentences<-lapply(sentences, FUN=rm_single_char_sent)
sentences<-lapply(sentences, unique)  # remove duplicate sentences

targetdoc<- list()  #create target document using transition matrix
truth<-list()  # record trur labels of sentences
row<-sample(1:k,1)
sizes<-c(rep(1,k))
while(sizes[row]<=length(sentences[[targetfiles[row]]]))  #repeat until of the document used to create documents ends
{
vec<-rmultinom(1,size = 1,trans_probab[row,])  # select next document to copy sentence from
row<-which.max(vec)
sent<-sentences[[targetfiles[row]]][sizes[row]]
targetdoc<-c(targetdoc,sent)
sizes[row]<-sizes[row]+1
truth<-c(truth,row)
}

targetdoc<-unlist(targetdoc)
truth<-unlist(truth)

for(file in targetfiles){   # remove all the documents used to create target document from the corpus
  sentences[[file]]<-NULL
}

for(sentence in targetdoc){
  print(sentence)
  scores<-lapply(sentences,FUN = cosineSim,x=sentence)  #calculate cosine similarity score for each sentence in the target document
  lookupTable<-rbind(lookupTable,unlist(scores))
}

lookupTable<-lookupTable/rowSums(lookupTable)   #Normalize across row to sum to 1

for(topic in topics){
  len<-sum(unlist(lapply(as.list(which(startsWith(names(sentences),topic))),function(x) length(sentences[[x]]))))
  sentencespertopic<-c(sentencespertopic,len)
}
sentencespertopic<-unlist(sentencespertopic)  # count of sentences per topic

#Sum the sentence scores for a topic
temp<-data.frame()
total<-0
for(ssum in sentencespertopic){
  vector<-c(rep(0,nrow(lookupTable)))
  s<-total+1
  e<-total+ssum
  for(j in s:e){
    vector<-(vector+lookupTable[,j])
  }
  total<-total+ssum
  temp<-rbind(temp,vector)
}
lookupTable<-t(temp)  #emission matrix
rm(temp)

colnames(lookupTable)<-topics
rownames(lookupTable)<-targetdoc


################################ HMM #########################################

hmm = initHMM(topics, targetdoc, transProbs=as.matrix(trans_probab),emissionProbs=as.matrix(t(lookupTable))) # initalize HMM

posterior <- posterior(hmm,targetdoc) #Forward Backward probabilities

bw<-baumWelch(hmm,targetdoc,maxIterations = 200,pseudoCount = 1e-10) # pseudocount added for additive smoothing
posterior<-posterior(bw$hmm,targetdoc) # learn the parameters using baum welch and recalculate Forward Backward probabilities

matcorp<-lexicalize(unlist(sentences),sep = ' ')  # create document matrxi as required for sLDA
label<-list()
params <- sample(c(0:(k-1)), k, replace=TRUE)

i<-0

for(val in sentencespertopic){   # label for document annotations
  label<-c(label,c(rep(i,val)))
  i<-i+1
}

label<-unlist(label)
result <- slda.em(documents=matcorp$documents,  # sLDA EM 
                  K=k,
                  vocab=matcorp$vocab,
                  num.e.iterations=10,
                  num.m.iterations=4,
                  alpha=1.0, eta=0.1,
                  label,
                  params,
                  variance=0.25,
                  lambda=1.0,
                  logistic=FALSE,
                  method="sLDA")
tcorp<-lexicalize(targetdoc)   # target document to predict
predictions <- slda.predict(tcorp$documents,  # make predictions
                            result$topics, 
                            result$model,
                            alpha = 1.0,
                            eta=0.1)
predicted.docsums <- slda.predict.docsums(tcorp$documents,
                                          result$topics, 
                                          alpha = 1.0,
                                          eta=0.1)
predicted.proportions <- t(predicted.docsums) / colSums(predicted.docsums) # calculte proportion of each topic in target document
# AUC scores for each topic
auc_df<-data.frame()  
rown<-list()
for(i in unique(truth)){
  print(topics[i])
  tval<-(truth==i)^1
  obc<-lookupTable[,i]
  cauc<-Auc(obc,tval)
  ob<-posterior[i,]
  auc<-Auc(ob,tval)
  obLDA<-predicted.proportions[,i]
  aucLDA <- Auc(obLDA,tval)
  auc_df<-rbind(auc_df,c(auc[1],cauc[1],aucLDA[1]))
  rown<-c(rown,topics[i])
}

rownames(auc_df)<-unlist(rown)
colnames(auc_df)<-c('AUC Model','AUC cosine', 'AUC sLDA')




