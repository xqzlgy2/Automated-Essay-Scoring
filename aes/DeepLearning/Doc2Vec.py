import time
import datetime
import EventIssuer
import csv
import sys
import io
import logging
import nltk
from nltk import word_tokenize
import numpy as np
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.corpora import WikiCorpus
import codecs

def processWikiCorpus(logfilename):
    i = 0
    # data sourse (14881 articles): https://dumps.wikimedia.org/enwiki/latest/
    inputfile = 'corpus/enwiki-latest-pages-articles1.xml-p10p30302.bz2'
    outputfile = 'corpus/wiki_en.txt'
    out = io.open(outputfile, 'w', encoding='utf-8')
    wiki = WikiCorpus(inputfile, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        out.write(' '.join(text) + "\n")
        i += 1
        EventIssuer.issueSuccess(str(i) + ' articles saved to disk.', logfilename)
    out.close()
    EventIssuer.issueSuccess('Preprocess finished.', logfilename)

def trainDoc2VecModel(logfilename):
    # parameters for doc2vec
    vector_size = 300 # length of word vectors
    window_size = 15
    min_count = 5
    sampling_threshold = 1e-5
    negative_size = 5
    train_epoch = 100
    dm = 1 # PV-DBOW = 0, PV-DM = 1
    worker_count = 1
    
    pretrained_emb = "glove_vectors/pretrained_word_embeddings.txt"
    train_corpus = "corpus/wiki_en.txt"
    saved_path = "models/wiki_en_doc2vec.model.bin"

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docs = TaggedLineDocument(train_corpus)
    model = Doc2Vec(docs, vector_size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count,\
                   hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, epochs=train_epoch)
    model.save(saved_path)

def encodeOneHot(score, dataset):
    if (dataset == "../Dataset/Set1Complete.csv"):
        onehot_y = [0] * 13
    elif (dataset == "../Dataset/Set3Complete.csv"):
        onehot_y = [0] * 4
    onehot_y[int(score)] = 1
    return np.array(onehot_y)

def getEssayNumber(dataset):
    total = 0
    with open(dataset, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)
        for row in datareader:
            #print row[0]
            essay = row[2]
            try:
                word_tokens = word_tokenize(essay)
            except UnicodeDecodeError:
                essay = essay.decode('latin-1').encode("utf-8")
                word_tokens = word_tokenize(essay)
            total += 1
    return total

def getScore(dataset, row):
    if (dataset == "../Dataset/Set1Complete.csv"):
        return float(float(row[3])+float(row[4]))
    elif (dataset == "../Dataset/Set3Complete.csv"):
        return float(row[6])
    return 0

def inferDocVector(logfilename, dataset):
    #parameters
    model = "models/wiki_en_doc2vec.model.bin"
    test_docs = "corpus/test.txt"
    output_file = "corpus/test_vector.txt"
    start_alpha = 0.01
    infer_epoch = 1000

    #load model
    EventIssuer.issueMessage("Loading Doc2Vec model...", logfilename)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    m = Doc2Vec.load(model)
    EventIssuer.issueSuccess("Loaded Doc2Vec model.", logfilename)

    #infer doc vectors
    setname = dataset
    dataset = "../Dataset/"+setname+"Complete.csv"
    total_done = 0.
    beforeStart = time.time()
    with open(dataset, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)
        X = []
        Y = []
        total_essays = getEssayNumber(dataset)
        EventIssuer.issueMessage("Number of essays in dataset:" + str(total_essays), logfilename)
        for row in datareader:
            essay = row[2]
            score = getScore(dataset, row)
            print "essay id:" + str(row[0])
            try:
                word_tokens = word_tokenize(essay)
            except UnicodeDecodeError:
                essay = essay.decode('latin-1').encode("utf-8")
                word_tokens = word_tokenize(essay)
            essay_vector = m.infer_vector(word_tokens, alpha=start_alpha, steps=infer_epoch)
            total_done += 1.
            X.append(essay_vector)
            Y.append(encodeOneHot(score, dataset))
            EventIssuer.issueSharpAlert("Complete: " + str(round(total_done*100/total_essays, 2)) + "%", logfilename)
    afterEnd = time.time()
    EventIssuer.issueSuccess(str(int(total_done)) + " essays processed in " + str(afterEnd-beforeStart), logfilename, ifBold=True)
    X = np.array(X)
    Y = np.array(Y)
    with open('ppData/X_Doc2Vec_'+setname+'.ds', 'w') as f:
        pickle.dump(X, f)
    with open('ppData/Y_Doc2Vec_'+setname+'.ds', 'w') as f:
        pickle.dump(Y, f)

def getMaxSentencesNumber(dataset):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    with open(dataset, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)
        maxlen = 0
        for row in datareader:
            essay = row[2]
            try:
                sentences = tokenizer.tokenize(essay)
            except UnicodeDecodeError:
                essay = essay.decode('latin-1').encode("utf-8")
                sentences = tokenizer.tokenize(essay)
            if maxlen < len(sentences):
                maxlen = len(sentences)
    return maxlen

def inferSentenceVector(logfilename, dataset):
    setname = dataset
    dataset = "../Dataset/"+setname+"Complete.csv"

    #load tokenizer, get max sentences number (dataset1:52 dataset2:22)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    EventIssuer.issueMessage("Computing max Sentences number...", logfilename)
    maxlen = getMaxSentencesNumber(dataset)
    EventIssuer.issueSuccess("Max Senteces number of dataset:"+str(maxlen), logfilename)

    #parameters for Doc2Vec
    model = "models/wiki_en_doc2vec.model.bin"
    test_docs = "corpus/test.txt"
    output_file = "corpus/test_vector.txt"
    start_alpha = 0.01
    infer_epoch = 1000

    #load Doc2Vec model
    EventIssuer.issueMessage("Loading Doc2Vec model...", logfilename)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    m = Doc2Vec.load(model)
    EventIssuer.issueSuccess("Loaded Doc2Vec model.", logfilename)

    #infer vectors
    beforeStart = time.time()
    with open(dataset, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)
        X = []
        Y = []
        total_essays = getEssayNumber(dataset)
        EventIssuer.issueMessage("Number of essays in dataset:" + str(total_essays), logfilename)
        total_done = 0.
        for row in datareader:
            essay = row[2]
            score = getScore(dataset, row)
            print "essay id:" + str(row[0])
            #tokenize
            try:
                sentences = tokenizer.tokenize(essay)
            except UnicodeDecodeError:
                essay = essay.decode('latin-1').encode("utf-8")
                sentences = tokenizer.tokenize(essay)
            #infer vectors for each senteces
            essay_len = 0
            essay_vector = []
            for sentence in sentences:
                essay_len += 1
                EventIssuer.issueSharpAlert("Infering vector for sentence"+str(essay_len)+" in essay"+str(row[0]), logfilename)
                word_tokens = word_tokenize(sentence)
                essay_vector.append(m.infer_vector(word_tokens, alpha=start_alpha, steps=infer_epoch))
            #fill rest part by zero vectors
            while essay_len < maxlen:
                essay_len += 1
                essay_vector.append(np.zeros(300))
            total_done += 1.
            X.append(essay_vector)
            Y.append(encodeOneHot(score, dataset))
            EventIssuer.issueSuccess("Complete: " + str(round(total_done*100/total_essays, 2)) + "%", logfilename)

    afterEnd = time.time()
    EventIssuer.issueSuccess(str(int(total_done)) + " essays processed in " + str(afterEnd-beforeStart), logfilename, ifBold=True)
    X = np.array(X)
    Y = np.array(Y)
    with open('ppData/X_Doc2Vec_3D_'+setname+'.ds', 'w') as f:
        pickle.dump(X, f)
    with open('ppData/Y_Doc2Vec_3D_'+setname+'.ds', 'w') as f:
        pickle.dump(Y, f)
    EventIssuer.issueSuccess("Shape for X:"+str(X.shape), logfilename)

def inferEssayVector(_LOGFILENAME, essay_filename):
    #parameters
    model = "models/wiki_en_doc2vec.model.bin"
    test_docs = "corpus/test.txt"
    output_file = "corpus/test_vector.txt"
    start_alpha = 0.01
    infer_epoch = 1000

    #load model
    EventIssuer.issueMessage("Loading Doc2Vec model...", _LOGFILENAME)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    m = Doc2Vec.load(model)
    EventIssuer.issueSuccess("Loaded Doc2Vec model.", _LOGFILENAME)

    #infer doc vectors
    beforeStart = time.time()
    EventIssuer.issueMessage("Reading the Essay file provided to system " + essay_filename, _LOGFILENAME)
    with open(essay_filename, 'r') as f:
        essay = f.read()
    try:
        word_tokens = word_tokenize(essay)
    except UnicodeDecodeError:
        essay = essay.decode('latin-1').encode("utf-8")
        word_tokens = word_tokenize(essay)
    EventIssuer.issueMessage("Infering essay vector...", _LOGFILENAME)
    essay_vector = m.infer_vector(word_tokens, alpha=start_alpha, steps=infer_epoch)
    afterEnd = time.time()
    EventIssuer.issueSuccess("Preprocessed the essay in " + str(afterEnd-beforeStart), _LOGFILENAME)
    return np.array(essay_vector)

def inferEssaySentenceVector(_LOGFILENAME, essay_filename, dataset):
    #load tokenizer, get max sentences number (dataset1:52 dataset2:22)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    EventIssuer.issueMessage("Computing max Sentences number...", _LOGFILENAME)
    maxlen = getMaxSentencesNumber("../Dataset/"+dataset+"Complete.csv")
    EventIssuer.issueSuccess("Max Senteces number of dataset:"+str(maxlen), _LOGFILENAME)

    #parameters
    model = "models/wiki_en_doc2vec.model.bin"
    test_docs = "corpus/test.txt"
    output_file = "corpus/test_vector.txt"
    start_alpha = 0.01
    infer_epoch = 1000

    #load model
    EventIssuer.issueMessage("Loading Doc2Vec model...", _LOGFILENAME)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    m = Doc2Vec.load(model)
    EventIssuer.issueSuccess("Loaded Doc2Vec model.", _LOGFILENAME)

    #infer doc vectors
    beforeStart = time.time()
    EventIssuer.issueMessage("Reading the Essay file provided to system " + essay_filename, _LOGFILENAME)
    with open(essay_filename, 'r') as f:
        essay = f.read()
    try:
        sentences = tokenizer.tokenize(essay)
    except UnicodeDecodeError:
        essay = essay.decode('latin-1').encode("utf-8")
        sentences = tokenizer.tokenize(essay)
    essay_len = 0
    essay_vector = []
    EventIssuer.issueMessage("Infering vector for each sentence...", _LOGFILENAME)
    for sentence in sentences:
        essay_len += 1
        EventIssuer.issueSharpAlert("Infering vector for sentence"+str(essay_len)+" in essay", _LOGFILENAME)
        word_tokens = word_tokenize(sentence)
        essay_vector.append(m.infer_vector(word_tokens, alpha=start_alpha, steps=infer_epoch))
    while essay_len < maxlen:
        essay_len += 1
        essay_vector.append(np.zeros(300))
    afterEnd = time.time()
    EventIssuer.issueSuccess("Preprocessed the essay in " + str(afterEnd-beforeStart), _LOGFILENAME)
    return np.array(essay_vector)

#processWikiCorpus('logs/DeepScore_Log_1554726321.8')
#trainDoc2VecModel('logs/DeepScore_Log_1554726321.8')
#inferDocVector('logs/DeepScore_Log_1554726321.8', "Set3")
#inferSentenceVector('logs/DeepScore_Log_1554726321.8', "Set3")