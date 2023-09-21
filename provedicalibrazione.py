import numpy
import datetime

import evaluation
import logisticRegression
import utility
import SVM
import GMM

if __name__ == '__main__':
    priors = [0.5, 0.1, 0.9]
    D, L = utility.load_dataset('Train.txt')
    DZ = utility.Znormalization(D)
    Dfolds, Lfolds = utility.Ksplit(D, L, 5)


    # O QUESTA-------------------------------------------

    # X-axis
    effPriorLogOdds = numpy.linspace(-4, 4, 100)
    effPriors = 1 / (1 + numpy.exp(-1 * effPriorLogOdds))
    # SVM RBF, RAW, C=10, y=0.001
    actualDCFs = []
    minDCFs = []
    calibrationDT = numpy.zeros(D.shape[1])
    calibrationLT = numpy.zeros(D.shape[1])

    calibratedDCFs = []
    for i, effP in enumerate(effPriors):  # Per ogni possibile application
        date = datetime.datetime.now()
        now = date.strftime("%H:%M:%S")
        print("---Ho inziato il giro %d alle ore %s" % (i, now))
        # Applico K-fold
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            # """SVM
            svm = SVM.SVM(DT, LT, 0.5, 10, 1)
            svm.train()
            scores.append(svm.getScores(DV))
            # """
            #"""LR
            lR = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
            lR.train()
            scores.append(lR.getScores(DV))
            #"""
            #""" RBF
            svm = SVM.RBF_SVM(DT, LT, 0.5, 10, 1, 0.001)
            svm.train()
            scores.append(svm.getScores(DV))
            #"""
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        # Funzione calibrate scores
        logr = logisticRegression.LogisticRegression(utility.vrow(scores), orderedLabels, 0, 0.5)
        logr.train()
        # Questi valori potrei farmeli tornare
        Cscores = logr.xOpt[0:-1] * scores + logr.xOpt[-1]  # -numpy.log(prior/(1-prior))
        # fine
        actualDCFs.append(evaluation.compute_actual_DCF(effP, 1, 1, scores, orderedLabels))
        minDCFs.append(evaluation.minimum_DCF(scores, orderedLabels, effP, 1, 1))
        calibratedDCFs.append(evaluation.compute_actual_DCF(effP, 1, 1, Cscores, orderedLabels))
    utility.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "SVM - piT=0.5 - C=10 - Z-normalized data")
    utility.bayesErrorPlot(calibratedDCFs, minDCFs, effPriorLogOdds, "SVM - piT=0.5 - C=10 - Z-normalized data")

    # O QUESTA-------------------------------------------

    #VERIFICO LE CALIBRAZIONI
    notCalibrated_scores = []
    orderedLabels = []
    for idx in range(5):
        # Evaluation set
        DV = Dfolds[idx]
        LV = Lfolds[idx]
        DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
        #"""GMM
        bestGMM = GMM.GMM(DT, LT, "tied", 4)
        bestGMM.train()
        notCalibrated_scores.append(bestGMM.getScores(DV))
        #"""
        """RBF SVM
        bestRBSVM = SVM.RBF_SVM(DT, LT, 0.5, 5, 1, 0.1)
        bestRBSVM.train()
        notCalibrated_scores.append(bestRBSVM.getScores(DV))
        """
        """SVM Lineare
        bestSVM = SVM.SVM(DT, LT, 0.5, 10, 1)
        bestSVM.train()
        notCalibrated_scores.append(bestSVM.getScores(DV))
        """
        """LR
        # Fase di training
        bestLogR = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
        bestLogR.train()
        # Fase di validation
        notCalibrated_scores.append(bestLogR.getScores(DV))
        """
        orderedLabels.append(LV)
    notCalibrated_scores = numpy.hstack(notCalibrated_scores)
    orderedLabels = numpy.hstack(orderedLabels)
    # Funzione calibrate scores
    logr = logisticRegression.LogisticRegression(utility.vrow(notCalibrated_scores), orderedLabels, 0, 0.5)
    logr.train()
    # Questi valori potrei farmeli tornare
    calibrated_scores = logr.xOpt[0:-1] * notCalibrated_scores + logr.xOpt[-1]
    for p in priors:
        actDCF_uncalibrated = evaluation.compute_actual_DCF(p, 1, 1, notCalibrated_scores, orderedLabels)
        actDCF_calibrated = evaluation.compute_actual_DCF(p, 1, 1, calibrated_scores, orderedLabels)
        minDCF = evaluation.minimum_DCF(calibrated_scores, orderedLabels, p, 1,1)
        print("PRIOR: %.1f, actDCF_uncalibrated: %.3f ------> actDCF_calibrated: %.3f" % (p, actDCF_uncalibrated, actDCF_calibrated))
        print("PRIOR: %.1f, minDCF: %.3f\n" %(p, minDCF))





