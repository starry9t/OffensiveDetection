# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 02:10:36 2018

@author: zy
"""
from __future__ import unicode_literals
import pandas, numpy, re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer as wnl
from nltk import word_tokenize as wt
from nltk import FreqDist
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.feature_extraction.text import *
import matplotlib.pyplot as plt 

def loadwords(bwfile):
    words = set()
    with open(bwfile,'r') as reader:
        for line in reader.readlines():
            word = line.strip()
            words.add(word)
    return words
# Global
BW = loadwords("badword.txt")

def GetMetrics(predicted,expected):
    """
    calculate auc
    """
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def DrawAUC(predicted, expected,title="AUC scores"):
    plt.title(title)
    plt.subplot(111)
    n = len(predicted)
    fpr = [0]*n
    tpr = [0]*n
    thresholds = [0]*n
    roc_auc = [0]*n
    color = ['blue','red','yellow','purple','green','m','c']
    clfname = ["LR","SVMs","CNN","LR+SVMs","LR+CNN","SVMs+CNN","LR+SVMs+CNN"] #correspond to predictions
    for i in range(n):
        prediction = predicted[i]
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(expected, prediction)  
        roc_auc[i] = metrics.auc(fpr[i],tpr[i])
        print(roc_auc[i])
        plt.plot(fpr[i], tpr[i], color[i],label='%s AUC: %0.6f' % (clfname[i], roc_auc[i]))

    plt.legend(loc='lower right')  
    plt.plot([0,1],[0,1],'k--')  
    plt.xlim([0,1])  
    plt.ylim([0,1])  
    plt.ylabel('True Positive Rate')  
    plt.xlabel('False Positive Rate')  
    return plt

def PreprocessCSV(csvfile,outputfile):
    """
    output a csv file and return a word list.
    """
    print("Start preprocessing %s ..." % csvfile)
    voc = []
    dataframe = pandas.read_csv(csvfile,usecols=["Insult","Comment"])
    labels = dataframe.iloc[:,0].tolist()
    sents = dataframe.iloc[:,1].tolist()
    newsents = []
    for sent in sents:
        # process sentences of samples
        # in case of blank, add a useless flag at the end
        sent = sent.strip("\"").lower()
        sent = sent.replace("\t"," ")
        sent = sent.replace("\n"," ")
        sent = sent.replace("\xa0"," ")
        sent = sent.replace("\xc2"," ")
        sent = sent.replace("\xc8"," ")
        sent = sent.replace("\xec"," ")
        sent = sent.replace("\x80"," ")
        sent = sent.replace("\xa6"," ")
        sent = re.sub("[$%^&*\[\]]","",sent)
        tks = wt(sent)
        newtks = []
        
        #built first-part features
        for tk in tks:
            if tk.isalpha():
                tk = wnl().lemmatize(tk)
                newtks.append(tk)
                voc.append(tk)
            else:
                pass
        newsent = " ".join(newtks)
        newsent = newsent + " " + "auselessflag"
        newsents.append(newsent)
        
    # write the outputfile
    col_order = ["Insult","Comment"]
    dataframe2 = pandas.DataFrame({"Insult":labels,"Comment":newsents})
    dataframe2.to_csv(outputfile,index=False,columns=col_order)
    fdist = FreqDist(voc)
    keys = fdist.keys()
    wordlist = []
    for key in keys:
        wordlist.append(key)
    print("file \"%s\" is preprocessed, and there are %d keys in the return wordlist." % (csvfile,len(wordlist)))
    return wordlist

def NgramWords(csvfile,n=2,minx=2,maxx=6):
    ngramlist = []
    bgramdict = defaultdict(int) 
    dataframe = pandas.read_csv(csvfile,usecols=["Comment"])
    sents = dataframe.iloc[:,0].tolist()
    for sent in sents:
        words = wt(sent)
        if len(words) > (n+2): #because there is a useless flag at the end of text
            for i in range((len(words)-n)):
                bword = ""
                for j in range(n):
                    bword += words[i+j]
                bgramdict[bword] += 1 
    for key in bgramdict.keys():
        if bgramdict[key] > minx and bgramdict[key] < maxx:
            ngramlist.append(key)     
    print("there are %d %d-gram words in ngramlist from %s." % (len(ngramlist),n,csvfile))
    #print(ngramlist[:10])
    return ngramlist

            
def ExtractFeature1(processedfile,wordlist1=False,wordlist2=False,wordlist3=False):
    """
    processedfile : csvfile, (output of 'PreprocessCSV')
    wordlist1/2/3 : list
    """
    features = []
    dataframe = pandas.read_csv(processedfile,usecols=["Insult","Comment"])
    labels = dataframe.iloc[:,0].tolist()
    sents = dataframe.iloc[:,1].tolist()
    
    # unigram feature
    for sent in sents:
        #every sent generates a feature vector
        words = wt(sent)
        sent_fea = []
        #first part feature
        if wordlist1:
            cur_fea1 = [0]*len(wordlist1)
            for word in words:
                if word in wordlist1:
                    fea_ind = wordlist1.index(word)
                    cur_fea1[fea_ind] += 1
                else:
                    pass
            sent_fea += cur_fea1
            cur_fea2 = [0]
            for word in words:
                if word in BW:
                    cur_fea2[0] += 1
                    if word.isupper():
                        cur_fea2[0] += 1
            sent_fea += cur_fea2
            cur_fea3 = [0]
            if cur_fea2[0] == 1:
                if "you" in sent:
                    cur_fea3[0] += 1
            sent_fea += cur_fea3 
        # bigram feature
        if wordlist2:
            cur_bgram = [0]*(len(wordlist2))
            for word in words:
                if len(words) > 2:
                    for i in range((len(words)-1)):
                        bword = words[i]+words[i+1]
                        if bword in wordlist2:
                            fea_ind = wordlist2.index(bword)
                            cur_bgram[fea_ind] += 1
                        else:
                            pass
            sent_fea += cur_bgram
        # trigram feature
        if wordlist3:
            cur_trigram = [0]*(len(wordlist3))
            for word in words:
                if len(words) > 3:
                    for i in range((len(words)-2)):
                        tword = ""
                        for j in range(3):
                            tword += words[i+j]
                        if tword in wordlist3:
                            fea_ind = wordlist3.index(tword)
                            cur_trigram[fea_ind] += 1
                        else:
                            pass
            sent_fea += cur_trigram
        features.append(sent_fea)
    print("labels and features are extracted from file %s." % processedfile)
    return labels,features
    
def ExtractFeature2(processedtrainfile,processedtestfile,vectorizer="count"):
    """
    vectorizer: "count"(default) or "tfidf" or "hashing"
    """
    dataframe = pandas.read_csv(processedtrainfile,usecols=["Insult","Comment"])
    train_labels = dataframe.iloc[:,0].tolist()
    sents = dataframe.iloc[:,1].tolist()
    if vectorizer == "count":
        vct = CountVectorizer()
    elif vectorizer == "tfidf":
        vct = TfidfVectorizer()
    elif vectorizer == "hashing":
        vct = HashingVectorizer()
    else:
        print("warning: the input of vectorizer is not correct. automatically set to CountVectorizer.")
        vct = CountVectorizer()
    vct_fit = vct.fit(sents)
    vct_trans = vct_fit.transform(sents)
    fea_arr = vct_trans.toarray()
    train_features = fea_arr.tolist()
    
    dataframe2 = pandas.read_csv(processedtestfile,usecols=["Insult","Comment"])
    test_labels = dataframe2.iloc[:,0].tolist()
    sents2 = dataframe2.iloc[:,1].tolist()
    vct_trans2 = vct_fit.transform(sents2)
    fea_arr2 = vct_trans2.toarray()
    test_features = fea_arr2.tolist()
    
    print("labels and features are extracted from file %s and %s." % (processedtrainfile,processedtestfile))
    return train_labels, train_features, test_labels, test_features   #  list

def MyModel(train_set, test_set, outputfile, extract = 1):
    """
    train_set and test_set : original sets
    outputfile: file with solution written
    return: prediction(probability) and expected(label) 
    """
    # preprocess files
    wordlist1 = PreprocessCSV(train_set,"ptrain.csv")
#    wordlist2 = NgramWords("ptrain.csv",2,5,11)
#    wordlist3 = NgramWords("ptrain.csv",3,3,6)
#    L1,L2,L3 = len(wordlist1),len(wordlist2),len(wordlist3)
    L1 = len(wordlist1)
    temp = PreprocessCSV(test_set,"ptest.csv")
    print("preprocessing done.")
    
    #set default
    uniwordlist = False
    biwordlist = False
    triwordlist = False
    
    # extract features
    if extract == 2:
        train_l, train_f, test_l, test_f = ExtractFeature2("ptrain.csv","ptest.csv",vectorizer="Hashing")
        train_l = numpy.array(train_l)
        train_f = numpy.array(train_f)
        test_f = numpy.array(test_f)
        print(train_f.shape)
        print(test_f.shape)
        print("feature extraction done.") 
    elif extract == 1:
        train_l, train_f = ExtractFeature1("ptrain.csv",wordlist1,False,False)
        train_l = numpy.array(train_l).reshape(len(train_l),1)
        train_f = numpy.array(train_f)
        
        # use chi2 to select features
        mym = SelectKBest(chi2, k=5000) #-----------------
        train_f1 = mym.fit_transform(train_f[:,0:-2],train_l) #first of fisrt part , nparray
        train_f = numpy.concatenate((train_f1,train_f[:,-2:]),axis=1)
        uniwordidx = mym.get_support(indices=True).tolist() 
        uniwordlist = []
        for index in uniwordidx:
            uniwordlist.append(wordlist1[index])
        numpy.save("uniwordlist.npy",uniwordlist)
        print("uniwordlist done.")
        
#        train_l2, train_f2 = ExtractFeature1("ptrain.csv",False,wordlist2,False)
#        train_l2 = numpy.array(train_l2).reshape(len(train_l2),1)
#        train_f2 = numpy.array(train_f2)
#        # use chi2 to select features
#        mym = SelectKBest(chi2, k=900)
#        train_f2 = mym.fit_transform(train_f2,train_l2)  #second part , nparray
#        biwordidx = mym.get_support(indices=True).tolist() 
#        biwordlist = []
#        for index in biwordidx:
#            biwordlist.append(wordlist2[index])
#        numpy.save("biwordlist.npy",biwordlist)
#        print("biwordlist done.")
#
#        train_l3, train_f3 = ExtractFeature1("ptrain.csv",False,False,wordlist3)
#        train_l3 = numpy.array(train_l3)
#        train_f3 = numpy.array(train_f3)
#        # use chi2 to select features
#        mym = SelectKBest(chi2, k=100)
#        train_f3 = mym.fit_transform(train_f3,train_l3)  #third part, nparray
#        triwordidx = mym.get_support(indices=True).tolist() 
#        triwordlist = []
#        for index in triwordidx:
#            triwordlist.append(wordlist3[index])
#        numpy.save("triwordlist.npy",triwordlist)
#        print("triwordlist done.")
    
        # concatenate
#        train_f = numpy.concatenate((train_f,train_f2,train_f3),axis=1)
#        test_l, test_f = ExtractFeature1("ptest.csv",uniwordlist,biwordlist,triwordlist)
#        test_l = numpy.array(test_l)
#        test_f = numpy.array(test_f)

        test_l, test_f = ExtractFeature1("ptest.csv",uniwordlist,wordlist2=False,wordlist3=False)
        test_l = numpy.array(test_l)
        test_f = numpy.array(test_f)
        
        print("feature extraction done.")
    
    numpy.save("train_f.npy",train_f)
    numpy.save("train_l.npy",train_l)
    numpy.save("expected.npy",test_l)
    # classify with svm
    mysvm = svm.SVC(C=0.5,kernel="linear",probability=True)    
    mysvm.fit(train_f,train_l)
    predicted_svm = mysvm.predict_proba(test_f)
    
    # classify with LR
    mylr = LogisticRegression(penalty="l2",C=3)
    mylr.fit(train_f,train_l)
    predicted_lr = mylr.predict_proba(test_f)

    # try different ways to generate final decision
    # 1. choose one directly
    predicted1 = predicted_lr[:,1]   #choose
    numpy.save("predicted1.npy",predicted1)
    predicted2 = predicted_svm[:,1]   #choose
    numpy.save("predicted2.npy",predicted2)

    # 2. combine two with weight 0.7 and 0.3 (can be adjusted)
    predicted5 = (predicted_svm*0.7+predicted_lr*0.3)[:,1]

    # 3. voting 
    predicted6 = []
    for i in range(len(predicted1)):
        if predicted1[i]>0.5 and predicted2[i]>0.5:
            predicted6.append(max(predicted1[i],predicted2[i]))
        elif predicted1[i]<0.5 and predicted2[i]<0.5:
            predicted6.append(min(predicted1[i],predicted2[i]))
        else:
            predicted6.append(predicted1[i]*0.3+predicted2[i]*0.7)
            
#-----            
    #read cnn result file
    with open("result.txt",'r') as reader:
        predicted3 = []
        for line in reader.readlines():
            a = float(line.strip())
            p = 1/(1+numpy.exp(-a))
            predicted3.append(p)
    predicted3 = numpy.array(predicted3)
    l = len(predicted3)
    predicted3 = predicted3.reshape(l,1)
    numpy.save("predicted3.npy",predicted3)
    
    predicted1 = numpy.load("predicted1.npy")  #LR
    n1 = len(predicted1)
    predicted1 = predicted1.reshape(n1,1)
    predicted2 = numpy.load("predicted2.npy")  #SVMs
    n2 = len(predicted2)
    predicted2 = predicted2.reshape(n2,1)
    expected = numpy.load("expected.npy")
    n3 = len(expected)
    expected = expected.reshape(n3,1)    
    
    predicted7 = predicted1*0.5+predicted3*0.5  #LR+CNN
    predicted8 = predicted2*0.5+predicted3*0.5  #SVMs+CNN
    predicted9 = predicted1*0.25+predicted2*0.25+predicted3*0.5

#-----            
    print("prediction done.")
    auc1 = GetMetrics(predicted1,test_l)
    print("auc of LR prediction is: ", auc1 )
    auc2 = GetMetrics(predicted2,test_l)
    print("auc of SVMs prediction is: ", auc2 )
    auc3 = GetMetrics(predicted3,test_l)
    print("auc of CNN prediction is: ", auc3 )

    auc5 = GetMetrics(predicted5,test_l)
    print("auc of LR+SVMs prediction is: ", auc5 )
    auc7 = GetMetrics(predicted7,test_l)
    print("auc of LR+CNN prediction is: ", auc7 )
    auc8 = GetMetrics(predicted8,test_l)
    print("auc of SVMs+CNN prediction is: ", auc8 )
    auc9 = GetMetrics(predicted9,test_l)
    print("auc of LR+SVMs+CNN prediction is: ", auc9 )

    predicteds = [predicted1,predicted2,predicted3,predicted5,predicted7,predicted8,predicted9]
    pic = DrawAUC(predicteds,test_l,"ROC curve")
    pic.show()
    
    print("now writting solution file...")
    alist = [auc1,auc2,auc3,auc5,auc7,auc8,auc9]
    auc = max(alist)
    i = alist.index(auc)
    predicted = predicteds[i]  #choose final method of prediction result

    numpy.save("predicted.npy",predicted)
    numpy.save("expected.npy",test_l)
    numpy.savetxt("predicted.txt",predicted)
    numpy.savetxt("expected.txt",test_l)
    
    # write solution file
    lattercol = pandas.read_csv(test_set, usecols=["Insult","Comment"])  #the original test_set
    dataarray = numpy.array(lattercol)
    col_lab = dataarray[:,0]
    col_comm = dataarray[:,1]
    predicted = predicted[:,0]
    col_order = ["Probability","Insult","Comment"]
    dataframe = pandas.DataFrame({"Probability":predicted,"Insult":col_lab,"Comment":col_comm})    
    dataframe.to_csv(outputfile,index=False,columns=col_order)
    print("solution file written.")
    
    # modify classification result with badword feature
    print("------this is an extra attempt------")
    predicted_r = predicted.copy()
    num = len(test_l)
    re_list = []
    for i in range(num):
        if test_f[i][-2] > 1:
            if predicted[i] < 0.55 and predicted[i] > 0.45:
                #pass
                predicted_r[i] += 0.45 #((test_f[i][-1])/10)
                re_list.append(i)
            else:
                pass
                #predicted_r[i] = 1
                #re_list.append(i)
    # 统计有多少改写了的
    print("there are %d samples are modified." % len(re_list))
    #print(re_list[:20])
    auc7 = GetMetrics(predicted_r,test_l)
    return mysvm,uniwordlist
        

if __name__ =='__main__':
    #MyModel("train.csv","test_with_solutions.csv","test_with_my_solution.csv",1)
    myclassifier,wordlist = MyModel("bigtrain.csv","impermium_verification_labels.csv","imper_with_my_solution.csv",1)

    
    
   
    