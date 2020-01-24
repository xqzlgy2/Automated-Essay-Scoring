import csv
import sys
from nltk.corpus import stopwords
import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy import spatial
import sys, getopt

# f1 = open('Results/stage1_results.csv', 'rb')

def performLSA(essay_fn, data_fn, ifesstxt=False):
    esstxts = []
    svParams = []
    '''Get perfect essays'''
    with open('Database/'+data_fn, 'rb') as f:
        perfect_essays = f.readlines()
    esstxts.append(" ".join(perfect_essays))

    if ifesstxt:
        test_essay = essay_fn
    else:
        '''Get the essay to be graded'''
        with open(essay_fn, 'rb') as f:
            test_essay = f.read()
    esstxts.append(test_essay)
    ignorechars = ''',:'!@'''

    transformer = TfidfTransformer(smooth_idf=False)
    vectorizer = TfidfVectorizer(max_features=10000,
                                 min_df=0.5, stop_words='english',
                                 use_idf=True)
    
    X = vectorizer.fit_transform(esstxts)

    tfidf = X.toarray()
    idf = vectorizer.idf_
    #print tfidf.shape
    #print tfidf

    U, s, V = np.linalg.svd(tfidf, full_matrices=True)
    #print "~~~~~~~~~~~~~~\n"
    #print s
    #print "~~~~~~~~~~~~~~\n"
    #print V
    svd = TruncatedSVD(n_iter=7, random_state=42, n_components=100)
    svd.fit(tfidf)
    svParams.append([U, s, V])
    #print "svd.explained_variance_ratio_" + str(svd.explained_variance_ratio_)
    #print svd.explained_variance_ratio_.sum()

    csim = 1 - spatial.distance.cosine(tfidf[0], tfidf[1])
    #print "CSIM: ", csim
    #print dict(zip(vectorizer.get_feature_names(), idf))
    return csim*12

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:d:",["ifile=", "dfile="])
    except getopt.GetoptError:
        print 'SEAM.py -i <inputfile> -d <datafile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'SEAM.py -i <inputfile> -d <datafile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            EssayFileName = arg
        elif opt in ("-d", "--dfile"):
            DataFileName = arg
    print "Semantic score of input essay: " + str(performLSA(EssayFileName, DataFileName)) + " out of 12"

if __name__ == "__main__":
    main(sys.argv[1:])