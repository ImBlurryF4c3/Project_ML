# FILE CHE CONTIENE IL MAIN PER FARE LE VARIE PROVE DEI MODULI
import numpy

import utility
import dimensionalityreduction as dr
import generative_models as gm
import evaluation
import logisticRegression
import quadLogisticRegression
import SVM


if __name__ == '__main__':
    D, L = utility.load_dataset('Train.txt')
    DC = utility.center_data(D)
    """ PROVE DI PLOTTING
    for feature in range(DC.shape[0]):
        plotHist(feature, DC, L)
    dr.plotPCAvariance(DC)
    for f1 in range(DC.shape[0]):
        for f2 in range(DC.shape[0]):
            if f1==f2:
                continue
            scatterPlot(f1, f2, DC, L)
    """
    """ ---PROVE PER LDA---
    for m in [9, 10, 11]:
        DP = dr.PCA(D, m)
        DPL = dr.LDA2(DP, L, 1)
        plotHist(0, DPL, L)
    #DL = dr.LDA(D, L, 1)
    #plotHist(0, DL, L)
    #DL2 = dr.LDA2(D, L, 1)
    #plotHist(0, DL2, L)
    """
    priors = [0.5, 0.1, 0.9]
    PCAdim = [9, 10, 11]
    types = ['MVG', 'Naive Bayes', 'Tied', 'Tied Naive Bayes']
    """
    # ----MODELLI GAUSSIANI---
    for m in PCAdim:
        print("\n")
        print("------Applying PCA with m = %d------" %(m))
        DP = dr.PCA(DC, m)

        for type in types:
            # Uso una tecnica di crossvalidation con k=5
            Dfolds, Lfolds = utility.Ksplit(DP, L, 5)
            # IDEA per pulire il codice
            # creare delle classi per modello e fare una funzione kfold a cui passo
            # Dfolds, Lfolds, modello
            # che traina il modello, fa l'evaluation e mi torna gli score
            print("")
            print("%s" %(type))
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                # Fase di training (dipende dal modello)
                muList, covMlist = gm.mvGaussian_ML_estimates(DT, LT, type)
                # Evaluation (dipende dal modello)
                scores.append(gm.getloglikelihoodRatios(DV, muList, covMlist))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            # finirebbe qui la Kfold che mi tornerebbe questi due numpy

            for i in range(len(priors)):
                minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[i], 1, 1)
                print("PRIOR: %.1f, minDCF: %.3f" %(priors[i], minDCF))
    print("-----NO PCA------")
    for type in types:
        # Uso una tecnica di crossvalidation con k=5
        Dfolds, Lfolds = utility.Ksplit(DC, L, 5)
        # IDEA per pulire il codice
        # creare delle classi per modello e fare una funzione kfold a cui passo
        # Dfolds, Lfolds, modello
        # che traina il modello, fa l'evaluation e mi torna gli score
        print("")
        print("%s" % (type))
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            # Fase di training (dipende dal modello)
            muList, covMlist = gm.mvGaussian_ML_estimates(DT, LT, type)
            # Evaluation (dipende dal modello)
            scores.append(gm.getloglikelihoodRatios(DV, muList, covMlist))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        # finirebbe qui la Kfold che mi tornerebbe questi due numpy

        for i in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[i], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[i], minDCF))
    """

    """---LOGISTIC REGRESSION---
    for i in range(len(priors)):
        print("")
        print("-----Logistic Regression(piT = %.1f)-----" %(priors[i]))
        Dfolds, Lfolds = utility.Ksplit(DC, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            lr = logisticRegression.LogisticRegression(DT, LT, 0, priors[i])
            # Fase di training (dipende dal modello)
            lr.train()
            # Evaluation (dipende dal modello)
            scores.append(lr.getloglikelihoodRatios(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        for j in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[j], minDCF))
    """

    """ ---- TROVARE LAMBDA PER LR ----
    DP = dr.PCA(DC, 11)
    lambdas = numpy.logspace(-5, 5, num=30)
    minDCF = []

    Dfolds, Lfolds = utility.Ksplit(DC, L, 5)

    for j in range(len(priors)):
        for l in lambdas:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                lr = logisticRegression.LogisticRegression(DT, LT, l, 0.5)
                # Fase di training (dipende dal modello)
                lr.train()
                # Evaluation (dipende dal modello)
                scores.append(lr.getloglikelihoodRatios(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1))
    utility.plotDCF(lambdas, minDCF, 'lambda')
    """
    """ ---QLR---
    
    for i in range(len(priors)):
        print("")
        print("-----QuadLogistic Regression(piT = %.1f)-----" % (priors[i]))
        Dfolds, Lfolds = utility.Ksplit(D, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            qlr = quadLogisticRegression.QLogisticRegression(DT, LT, 0.001, priors[i])
            # Fase di training (dipende dal modello)
            qlr.train()
            # Evaluation (dipende dal modello)
            scores.append(qlr.getloglikelihoodRatios(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        for j in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[j], minDCF))
    """
    """QLR SEARCH OF L --- TROPPO TEMPO INFATTIBILE
    lambdas = numpy.logspace(-5, 5, num=30)
    minDCF = []

    Dfolds, Lfolds = utility.Ksplit(DC, L, 5)

    for j in range(len(priors)):
        for l in lambdas:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                qlr = quadLogisticRegression.QLogisticRegression(DT, LT, l, 0.5)
                # Fase di training (dipende dal modello)
                qlr.train()
                # Evaluation (dipende dal modello)
                scores.append(qlr.getloglikelihoodRatios(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1))
    utility.plotDCF(lambdas, minDCF, 'lambda')
    """

    """ ----SVM LINEAR----
    for i in range(len(priors)):
        print("")
        print("-----SVM(piT = %.1f)-----" % (priors[i]))
        Dfolds, Lfolds = utility.Ksplit(D, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            svm = SVM.SVM(DT, LT, priors[i], 10, 1)
            # Fase di training (dipende dal modello)
            svm.train()
            # Evaluation (dipende dal modello)
            scores.append(svm.getScores(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        for j in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[j], minDCF))
    """
    """----SVM best C value----
    c_X = numpy.logspace(-5, 5, num=30)
    minDCF = []
    Dfolds, Lfolds = utility.Ksplit(DC, L, 5)
    for j in range(len(priors)):
        for c in c_X:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                svm = SVM.SVM(DT, LT, 0.5, c, 1)
                # Fase di training (dipende dal modello)
                svm.train()
                # Evaluation (dipende dal modello)
                scores.append(svm.getScores(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1))
    utility.plotDCF(c_X, minDCF, 'C')
    """
    """---POLY SVM ---
    for i in range(len(priors)):
        print("")
        print("-----Polynomial SVM-----")
        Dfolds, Lfolds = utility.Ksplit(D, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            svm = SVM.PolynomialSVM(DT, LT, 0.5, 0.001, 1, 1, 2)
            # Fase di training (dipende dal modello)
            svm.train()
            # Evaluation (dipende dal modello)
            scores.append(svm.getScores(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        for j in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[j], minDCF))
        """
    c_X = numpy.logspace(-5, 5, num=40)
    minDCF = []
    Dfolds, Lfolds = utility.Ksplit(D, L, 5)
    for j in range(len(priors)):
        for c in c_X:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                svm = SVM.PolynomialSVM(DT, LT, 0.5, c, 1, 10, 2)
                # Fase di training (dipende dal modello)
                svm.train()
                # Evaluation (dipende dal modello)
                scores.append(svm.getScores(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, priors[j], 1, 1))
    utility.plotDCF(c_X, minDCF, 'C')