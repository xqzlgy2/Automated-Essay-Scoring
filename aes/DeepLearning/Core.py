import time
import datetime
import EventIssuer
import csv
import sys, getopt
import math
import pickle
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import regularizers
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Masking, MaxPooling2D, Dropout, Conv2D, Conv1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import keras.backend as K
import Word2Vec
import DataPreprocessor
import Metrics
import Doc2Vec
import Analyzer

_LOGFILENAME = ""
reload(sys)
sys.setdefaultencoding('utf8')

def start_deepscore_core():
    global _LOGFILENAME, timestamp
    timestamp = time.time()
    strstamp = datetime.datetime.fromtimestamp(timestamp).strftime('%m-%d-%Y %H:%M:%S')
    _LOGFILENAME = "logs/DeepScore_Log_" + str(timestamp)
    np.random.seed(7)
    EventIssuer.issueWelcome(_LOGFILENAME)
    EventIssuer.genLogFile(_LOGFILENAME, timestamp, strstamp)
    return _LOGFILENAME, timestamp


def loadppData(xfname, yfname):
    X = pickle.load(open(xfname, "r"))
    Y = pickle.load(open(yfname, "r"))
    return X, Y

def saveModel(model, _LOGFILENAME, timestamp):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open('models/model_' + str(timestamp) + '.dsm', "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights('models/weights_' + str(timestamp) + '.h5')
    EventIssuer.issueSuccess("Saved model to disk", _LOGFILENAME)

def loadDeepScoreModel(_LOGFILENAME, model_fn):
    EventIssuer.issueMessage("Loading DeepScore Model : " + model_fn, _LOGFILENAME)
    # dsm = pickle.load(open('models/model_' + str(model_fn) + '.dsm', 'r'))

    # load YAML and create model
    yaml_file = open('models/model_' + str(model_fn) + '.dsm', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights('models/weights_' + str(model_fn) + '.h5')

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    EventIssuer.issueSuccess("Loaded Model Successfully.", _LOGFILENAME)
    return loaded_model

def train_model_avg():
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new model", _LOGFILENAME)
    X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
    # print X.shape, Y.shape
    # split into input (X) and output (Y) variables
    train_X = X[0:1500,:]
    train_Y = Y[0:1500,]
    test_X = X[1500:1700,:]
    test_Y = Y[1500:1700,]

    print str(train_X.shape)+str(train_Y.shape)+str(test_X.shape)+str(test_Y.shape)
    model = Sequential()
    model.add(Dense(12, input_dim=300, activation='relu'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_X, train_Y, epochs=200, batch_size=10, validation_split=0.1, shuffle=True, callbacks=[TensorBoard(log_dir='logs/tensorboard/avg_log')])
    plot_history(history)
    pyplot.savefig("pics/avg_history.jpg")

    saveModel(model, _LOGFILENAME, timestamp)
    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def getRMSE(predicted_scores, actual_scores):
    err_val = 0.
    for i in range(0, len(predicted_scores)):
        predicted_score = predicted_scores[i]
        actual_score = actual_scores[i]
        err_val += math.pow((predicted_score - actual_score), 2)
    mse = err_val / len(predicted_scores)
    rmse = math.sqrt(mse)
    return rmse

def mycrossentropy(y_true, y_pred, e=0.1):
    return -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)))


def train_NN(dataset):
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new NN", _LOGFILENAME)

    #load data
    X, Y = loadppData('ppData/X_Doc2Vec_'+dataset+'.ds', 'ppData/Y_Doc2Vec_'+dataset+'.ds')
    #split into input (X) and output (Y) variables
    seed = 7
    np.random.seed(seed)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print str(train_X.shape)+str(train_Y.shape)+str(test_X.shape)+str(test_Y.shape)

    #generate model
    model = Sequential()
    model.add(Dense(16, input_dim=300, activation='relu'))
    model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dropout(0.25))
    if (dataset == 'Set1'):
        model.add(Dense(13, activation='softmax'))
    elif (dataset == 'Set3'):
        model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #plot neural network
    plot_model(model, to_file='pics/model_NN.png', show_shapes=True)

    #train and plot history
    history = model.fit(train_X, train_Y, epochs=200, batch_size=10, validation_split=0.1, shuffle=True, callbacks=[TensorBoard(log_dir='logs/tensorboard/nn_log')])
    #history = model.fit(train_X, train_Y, epochs=200, batch_size=10)
    plot_history(history)
    pyplot.savefig("pics/nn_history.jpg")

    saveModel(model, _LOGFILENAME, timestamp)
    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def train_LSTM(dataset):
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new LSTM", _LOGFILENAME)

    #load data
    X, Y = loadppData('ppData/X_Doc2Vec_3D_'+dataset+'.ds', 'ppData/Y_Doc2Vec_3D_'+dataset+'.ds')
    #split into input (X) and output (Y) variables
    seed = 7
    np.random.seed(seed)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print str(train_X.shape)+str(train_Y.shape)+str(test_X.shape)+str(test_Y.shape)

    #get max sentence number
    sentence_num = Doc2Vec.getMaxSentencesNumber("../Dataset/"+dataset+"Complete.csv")

    #generate model
    model = Sequential()
    model.add(Masking(mask_value= 0,input_shape=(sentence_num, 300)))
    model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, input_shape=(sentence_num, 300), return_sequences=True))
    model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, input_shape=(sentence_num, 300), return_sequences=True))
    model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, input_shape=(sentence_num, 300)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.25))
    if (dataset == 'Set1'):
        model.add(Dense(13, activation='softmax'))
    elif (dataset == 'Set3'):
        model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #plot neural network
    plot_model(model, to_file='pics/model_LSTM.png', show_shapes=True)

    #train and plot history
    history = model.fit(train_X, train_Y, epochs=200, batch_size=10, validation_split=0.1, shuffle=True, callbacks=[TensorBoard(log_dir='logs/tensorboard/lstm_log')])
    #history = model.fit(train_X, train_Y, epochs=200, batch_size=10)
    plot_history(history)
    pyplot.savefig("pics/lstm_history.jpg")

    saveModel(model, _LOGFILENAME, timestamp)
    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def train_CNN(dataset):
    _LOGFILENAME, timestamp = start_deepscore_core()
    EventIssuer.issueMessage("Training a new CNN", _LOGFILENAME)

    #load data
    X, Y = loadppData('ppData/X_Doc2Vec_3D_'+dataset+'.ds', 'ppData/Y_Doc2Vec_3D_'+dataset+'.ds')
    #split into input (X) and output (Y) variables
    seed = 7
    np.random.seed(seed)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=seed)
    print str(train_X.shape)+str(train_Y.shape)+str(test_X.shape)+str(test_Y.shape)

    #get max sentence number
    sentence_num = Doc2Vec.getMaxSentencesNumber("../Dataset/"+dataset+"Complete.csv")

    #reshape dataset
    EventIssuer.issueMessage("Reshaping Dataset...", _LOGFILENAME)
    train_X = train_X.reshape((len(train_X),sentence_num,300,1))
    test_X = test_X.reshape((len(test_X),sentence_num,300,1))

    #generate model
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=(2,300), activation='relu', input_shape=(sentence_num,300,1), data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.25))
    if (dataset == 'Set1'):
        model.add(Dense(13, activation='softmax'))
    elif (dataset == 'Set3'):
        model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #plot neural network
    plot_model(model, to_file='pics/model_CNN.png', show_shapes=True)

    #train and plot history
    history = model.fit(train_X, train_Y, epochs=200, batch_size=10, validation_split=0.1, shuffle=True, callbacks=[TensorBoard(log_dir='logs/tensorboard/cnn_log')])
    #history = model.fit(train_X, train_Y, epochs=200, batch_size=10)
    plot_history(history)
    pyplot.savefig("pics/cnn_history.jpg")

    saveModel(model, _LOGFILENAME, timestamp)
    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def plot_history(history):
    pyplot.subplot(2,1,1)
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.legend()
    pyplot.subplot(2,1,2)
    pyplot.plot(history.history['acc'], label='acc')
    pyplot.legend()

def infer_score(_LOGFILENAME, EssayFileName, dataset, arg):
    sentence_num = Doc2Vec.getMaxSentencesNumber("../Dataset/"+dataset+"Complete.csv")
    if (arg == 'AVG'):
        essay_vector = DataPreprocessor.preprocessEssayText(_LOGFILENAME, EssayFileName)
        # print essay_vector.shape
        model = loadDeepScoreModel(_LOGFILENAME, "1_AVG")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector)))
        # print predicted_score
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)
    elif (arg == 'NN'):
        essay_vector = Doc2Vec.inferEssayVector(_LOGFILENAME, EssayFileName)
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_NN")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_NN")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,300))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)
    elif (arg == 'LSTM'):
        essay_vector = Doc2Vec.inferEssaySentenceVector(_LOGFILENAME, EssayFileName, dataset)
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_LSTM")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_LSTM")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,sentence_num,300))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)
    elif (arg == 'CNN'):
        essay_vector = Doc2Vec.inferEssaySentenceVector(_LOGFILENAME, EssayFileName, dataset)
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_CNN")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_CNN")
        predicted_score = np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,sentence_num,300,1))))
        EventIssuer.issueSuccess("The essay has been graded. The score should be " + str(predicted_score), _LOGFILENAME, ifBold=True)
        EventIssuer.issueExit(_LOGFILENAME, timestamp)

def evaluate_model(_LOGFILENAME, dataset, arg):
    sentence_num = Doc2Vec.getMaxSentencesNumber("../Dataset/"+dataset+"Complete.csv")
    predicted_scores = []
    actual_scores = []
    correct_num = 0.
    almost_correct = 0.

    if (arg == 'AVG'):
        model = loadDeepScoreModel(_LOGFILENAME, "1_AVG")
        X, Y = loadppData('ppData/X_1492930578.7.ds', 'ppData/Y_1492930578.7.ds')
        test_X = X[1500:1700,:]
        test_Y = Y[1500:1700,]
        for essay_vector in test_X:
            predicted_scores.append(np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,300)))))
        predicted_scores = np.array(predicted_scores)
        for onehot in test_Y:
            actual_scores.append(np.argmax(np.squeeze(onehot)))

    elif (arg == 'NN'):
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_NN")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_NN")
        X, Y = loadppData('ppData/X_Doc2Vec_'+dataset+'.ds', 'ppData/Y_Doc2Vec_'+dataset+'.ds')
        test_X = X[1500:1700,:]
        test_Y = Y[1500:1700,]
        for essay_vector in test_X:
            predicted_scores.append(np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,300)))))
        predicted_scores = np.array(predicted_scores)
        for onehot in test_Y:
            actual_scores.append(np.argmax(np.squeeze(onehot)))

    elif (arg == 'LSTM'):
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_LSTM")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_LSTM")
        X, Y = loadppData('ppData/X_Doc2Vec_3D_'+dataset+'.ds', 'ppData/Y_Doc2Vec_3D_'+dataset+'.ds')
        test_X = X[1500:1700,:]
        test_Y = Y[1500:1700,]
        for essay_vector in test_X:
            predicted_scores.append(np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,sentence_num,300)))))
        predicted_scores = np.array(predicted_scores)
        for onehot in test_Y:
            actual_scores.append(np.argmax(np.squeeze(onehot)))

    elif (arg == 'CNN'):
        if (dataset == 'Set1'):
            model = loadDeepScoreModel(_LOGFILENAME, "1_CNN")
        elif (dataset == 'Set3'):
            model = loadDeepScoreModel(_LOGFILENAME, "3_CNN")
        X, Y = loadppData('ppData/X_Doc2Vec_3D_'+dataset+'.ds', 'ppData/Y_Doc2Vec_3D_'+dataset+'.ds')
        test_X = X[1500:1700,:]
        test_Y = Y[1500:1700,]
        for essay_vector in test_X:
            predicted_scores.append(np.argmax(np.squeeze(model.predict(essay_vector.reshape(1,sentence_num,300,1)))))
        predicted_scores = np.array(predicted_scores)
        for onehot in test_Y:
            actual_scores.append(np.argmax(np.squeeze(onehot)))

    actual_scores = np.array(actual_scores)
    diffs = predicted_scores - actual_scores
    for diff in diffs:
        if math.fabs(diff) == 0:
            correct_num += 1
        if math.fabs(diff) <= 1:
            almost_correct += 1
    EventIssuer.issueSuccess("The model has been evaluated. Accuracy: " + str(correct_num/len(test_Y)*100) + '%', _LOGFILENAME, ifBold=True)
    EventIssuer.issueSuccess("Accuracy(+-1): " + str(almost_correct/len(test_Y)*100) + '%', _LOGFILENAME, ifBold=True)
    EventIssuer.issueSuccess("RMSE: " + str(getRMSE(predicted_scores, actual_scores)), _LOGFILENAME, ifBold=True)
    EventIssuer.issueExit(_LOGFILENAME, timestamp)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hf:d:t:e:i:",["file","data","train","evaluate","infer"])
    except getopt.GetoptError:
        print 'Core.py -f <file>  -d <data> -e <model> -t <model> -i <model>'
        EventIssuer.issueExit(_LOGFILENAME, timestamp)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'Core.py -f <file>  -d <data> -e <model> -t <model> -i <model>'
            sys.exit()
        elif opt in ("-f", "--file"):
            EssayFileName = arg
        elif opt in ("-d", "--data"):
            dataset = arg
        elif opt in ("-t", "--train"):
            if (arg == 'NN'):
                train_NN(dataset)
            elif (arg == 'LSTM'):
                train_LSTM(dataset)
            elif (arg == 'AVG'):
                train_model_avg()
            elif (arg == 'CNN'):
                train_CNN(dataset)
            exit(0)
        elif opt in ("-e", "--evaluate"):
            _LOGFILENAME, timestamp = start_deepscore_core()
            evaluate_model(_LOGFILENAME, dataset, arg)
        elif opt in ("-i", "--infer"):
            _LOGFILENAME, timestamp = start_deepscore_core()
            infer_score(_LOGFILENAME, EssayFileName, dataset, arg)

if __name__ == "__main__":
    main(sys.argv[1:])