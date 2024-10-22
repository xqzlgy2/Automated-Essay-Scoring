import SEAM
import SAM
import SYNAN
import DISAM
import SYNERR
import sys, getopt
import time, csv, pickle
import numpy as np
import math

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:d:",["ifile=", "dfile="])
    except getopt.GetoptError:
        print 'driver.py -i <inputfile> -d <datafile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'driver.py -i <inputfile> -d <datafile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            EssayFileName = arg
        elif opt in ("-d", "--dfile"):
            DataFileName = arg
    f = open('../Dataset/Set1Complete.csv', 'rb')
    calibrator = pickle.load(open("calibrated_model.sav", 'rb'))
    count = 0.
    beforeStart = time.time()
    numCorrect = 0
    err_val = 0.
    numAcc1 = 0.
    try:
        reader = csv.reader(f)
        for row in reader:
            if count > 0:
                ess_text = unicode(row[2], errors='ignore')
                ess_score_r1 = float(row[3])
                ess_score_r2 = float(row[4])
                seam_score = SEAM.performLSA(ess_text, DataFileName, ifesstxt=True)
                sam_score = SAM.performSA(ess_text, DataFileName, ifesstxt=True)
                synan_score = SYNAN.scoreSYN(ess_text, DataFileName, ifesstxt=True)
                disam_score = DISAM.scoreDiscourse(ess_text, DataFileName, ifesstxt=True)
                synerr_score = SYNERR.scoreSYNERR(ess_text, ifesstxt=True)
                score_vector = np.array([[seam_score, sam_score, synan_score, disam_score, synerr_score]])
                predicted_score = int(calibrator.predict(score_vector))
                actual_score = ess_score_r1 + ess_score_r2
                print "Predicted : ", predicted_score, " |  Actual : ", actual_score,
                if float(predicted_score) == float(actual_score):
                    numCorrect += 1
                    print "  |  Correct Prediction ! -- ",
                if math.fabs(float(predicted_score) - float(actual_score)) == 1:
                    print "  |  Almost Correct! -- ",
                    numAcc1 += 1
                err_val += math.pow((predicted_score-actual_score),2)
                print count * 100 / 1782, "% Complete.. | Est. Time Remaining : ", ((time.time() - beforeStart) * (
                1782 - count)) / (count * 60), "Minutes"
            count += 1
    finally:
        f.close()
        mse = err_val / count
        rmse = math.sqrt(mse)
        print "MSE : ", mse
        print "RMSE : ", rmse
        print "ACC : ", float(numCorrect)/float(count)
        print "Acc(+-1) : ", (float(numCorrect)+float(numAcc1))/float(count)
        #MSE :  1.78419282511
        #RMSE :  1.33573680982
        #ACC :  0.382847533632
        #Acc(+-1) :  0.788677130045
        
if __name__ == "__main__":
    main(sys.argv[1:])