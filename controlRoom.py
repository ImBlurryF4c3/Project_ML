# FILE CHE CONTIENE IL MAIN PER FARE LE VARIE PROVE DEI MODULI
import numpy
import utility
import SVM
import evaluation
import test
import logisticRegression
import datetime

if __name__ == '__main__':
    D, L = utility.load_dataset('Train.txt')
    DE, LE = utility.load_dataset('Test.txt')
    D, L = utility.shuffle_dataset(D, L)
    DE, LE = utility.shuffle_dataset(DE, LE)
    #test.test_models(D, L, DE, LE)
    DC = utility.center_data(D)
    priors = [0.5, 0.1, 0.9]
    PCAdim = [9, 10, 11]
    DZ = utility.Znormalization(D)
    """ ---PROVE DI PLOTTING---
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
    """ ---MODELLI GAUSSIANI--- FATTOOOOOOOO
    types = ['MVG', 'Naive Bayes', 'Tied', 'Tied Naive Bayes']
    for m in PCAdim:
        print("\n")
        print("------Applying PCA with m = %d------" %(m))
        DP = dr.PCA(D, m)
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
                gaussClass = gc.GaussianClassifier(DT, LT, type)
                gaussClass.train()
                # Evaluation (dipende dal modello)
                scores.append(gaussClass.getloglikelihoodRatios(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            # finirebbe qui la Kfold che mi tornerebbe questi due numpy
            for p in priors:
                minDCF = evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1)
                print("PRIOR: %.1f, minDCF: %.3f" %(p, minDCF))
    print("")
    print("-----NO PCA------")
    for type in types:
        # Uso una tecnica di crossvalidation con k=5
        Dfolds, Lfolds = utility.Ksplit(D, L, 5)
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
            gaussClass = gc.GaussianClassifier(DT, LT, type)
            gaussClass.train()
            # Evaluation (dipende dal modello)
            scores.append(gaussClass.getloglikelihoodRatios(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        # finirebbe qui la Kfold che mi tornerebbe questi due numpy
        for p in priors:
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (p, minDCF))
    """

    """---LOGISTIC REGRESSION---
    for p in priors:
        print("")
        print("-----Logistic Regression(piT = %.1f)-----" %(p))
        Dfolds, Lfolds = utility.Ksplit(DZ, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            lr = logisticRegression.LogisticRegression(DT, LT, 0, p)
            # Fase di training (dipende dal modello)
            ltr = lr.train()
            # Evaluation (dipende dal modello)
            scores.append(lr.getloglikelihoodRatios(DV))
            orderedLabels.append(LV)
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        for i in range(len(priors)):
            minDCF = evaluation.minimum_DCF(scores, orderedLabels, priors[i], 1, 1)
            print("PRIOR: %.1f, minDCF: %.3f" % (priors[i], minDCF))
    """

    """ ---- TROVARE LAMBDA PER LR ----
    DP = dr.PCA(D, 11)
    lambdas = numpy.logspace(-5, 5, num=50)
    minDCF = []
    for p in priors:
        for l in lambdas:
            scores = []
            orderedLabels = []
            Dfolds, Lfolds = utility.Ksplit(DP, L, 5)
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
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1))
    utility.plotDCF(lambdas, minDCF, '$lambda$')
    """
    """ ---QLR---
    for p in priors:
        print("")
        print("-----QuadLogistic Regression(piT = %.1f)-----" % (p))
        Dfolds, Lfolds = utility.Ksplit(DZ, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            qlr = quadLogisticRegression.QLogisticRegression(DT, LT, 0, p)
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
    """QLR SEARCH OF L --- 
    lambdas = numpy.logspace(-5, 5, num=40)
    minDCF = []
    for p in priors:
        for l in lambdas:
            scores = []
            orderedLabels = []
            Dfolds, Lfolds = utility.Ksplit(D, L, 5)
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                qlr = quadLogisticRegression.QLogisticRegression(DT, LT, l, 0.5)
                # Fase di training (dipende dal modello)
                qlr.train()
                # Evaluation (dipende dal modello)
                scores.append(qlr.getScores(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1))
    utility.plotDCF(lambdas, minDCF, 'lambda')
    """

    """ ----SVM LINEAR----
    for p in priors:
        print("")
        print("-----SVM(piT = %.1f)-----" % (p))
        Dfolds, Lfolds = utility.Ksplit(DZ, L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            svm = SVM.SVM(DT, LT, p, 10, 1)
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
    Dfolds, Lfolds = utility.Ksplit(D, L, 5)
    for p in priors:
        print("")
        print("-----Polynomial SVM (%.1f)-----" %(p))
        scores = []
        orderedLabels = []
        for idx in range(5):
            # Evaluation set
            DV = Dfolds[idx]
            LV = Lfolds[idx]
            DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
            svm = SVM.PolynomialSVM(DT, LT, p, 0.0001, 1, 1, 2)
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
    """---POLI SVM in funzione di C, Z, c=1---
    c_X = numpy.logspace(-5, 5, num=40)
    minDCF = []
    Dfolds, Lfolds = utility.Ksplit(DZ, L, 5)
    for p in priors:
        for c in c_X:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                svm = SVM.PolynomialSVM(DT, LT, 0.5, c, 1, 1, 2)
                # Fase di training (dipende dal modello)
                svm.train()
                # Evaluation (dipende dal modello)
                scores.append(svm.getScores(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1))
    utility.plotDCF(c_X, minDCF, 'C')
    """
    """---RBF SVM in funzione di C, Z, y=0.01---
    c_X = numpy.logspace(-5, 5, num=40)
    minDCF = []
    Dfolds, Lfolds = utility.Ksplit(DZ, L, 5)
    for p in priors:
        for c in c_X:
            scores = []
            orderedLabels = []
            for idx in range(5):
                # Evaluation set
                DV = Dfolds[idx]
                LV = Lfolds[idx]
                DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                svm = SVM.RBF_SVM(DT, LT, 0.5, c, 1, 0.01)
                # Fase di training (dipende dal modello)
                svm.train()
                # Evaluation (dipende dal modello)
                scores.append(svm.getScores(DV))
                orderedLabels.append(LV)
            scores = numpy.hstack(scores)
            orderedLabels = numpy.hstack(orderedLabels)
            minDCF.append(evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1))
    utility.plotDCF(c_X, minDCF, 'C')
    """
    """ ---- VERIFICA DI LAMBDA PER EVALUATION SET ----
    lambdas = numpy.logspace(-5, 5, num=50)
    minDCFe = []
    minDCFv = []
    for p in priors:
        for l in lambdas:
            lr = logisticRegression.LogisticRegression(DE, LE, l, 0.5)
            lrv = logisticRegression.LogisticRegression(D, L, l, 0.5)
            lr.train()
            lrv.train()
            scoresE = lr.getScores(DE)
            scoresV = lrv.getScores(D)
            minDCFe.append(evaluation.minimum_DCF(scoresE, LE, p, 1, 1))
            minDCFv.append(evaluation.minimum_DCF(scoresV, L, p, 1, 1))
    utility.plotDCF_evalVSval(lambdas, minDCFe, minDCFv, 'lambda')
    """
    #""" VERIFICA DI GAMMA E C
    cs = numpy.logspace(-5, 5, num=21)
    gammas = [0.1, 0.01, 0.001]
    minDCFe = []
    minDCFv = []
    count = 1
    for y in gammas:
        for p in priors:
            for c in cs:
                date = datetime.datetime.now()
                now = date.strftime("%H:%M:%S")
                print("---Ho inziato il giro %d alle ore %s" % (count, now))
                count +=1
                svm_train = SVM.RBF_SVM(D, L, 0.5, c, 1, y)
                svm_eval = SVM.RBF_SVM(DE, LE, 0.5, c, 1, y)
                svm_train.train()
                svm_eval.train()
                scoresE = svm_eval.getScores(DE)
                scoresV = svm_train.getScores(D)
                minDCFe.append(evaluation.minimum_DCF(scoresE, LE, p, 1, 1))
                minDCFv.append(evaluation.minimum_DCF(scoresV, L, p, 1, 1))
        utility.plotDCF_evalVSval(cs, minDCFe, minDCFv, 'C')
    #"""
