# FILE CHE CONTIENE IL MAIN PER FARE LE VARIE PROVE DEI MODULI
import numpy
import utility
import SVM
import evaluation
import test_evaluation
import logisticRegression
import datetime

if __name__ == '__main__':
    D, L = utility.load_dataset('Train.txt')
    DE, LE = utility.load_dataset('Test.txt')
    D, L = utility.shuffle_dataset(D, L)
    DE, LE = utility.shuffle_dataset(DE, LE)
    #test_evaluation.test_models(D, L, DE, LE)
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
    """ ----GMM-----
    D, L = utility.load_dataset('Train.txt') # Pc Roberto
    D, L = utility.load_dataset('Project_ML\Train.txt')
    #D, L = utility.load_dataset_shuffled('Project_ML\Train.txt') # PC gabri #with shuffle
    types_gmm = ['full_cov', 'diagonal', 'tied', 'tied_diagonal']
    data_type = ['raw_data', 'z_score_data']
    components = [1, 2, 4, 8, 16, 32]
    minDCF = []
    K = 5
    for title in data_type: # raw data and Z-normalization
        if title == 'raw_data':
            DTR = D
        else:
            DTR = utility.Znormalization(D) # dati normalizzati
        print("----- %s -----" % title)
        for t in types_gmm:
            for p in priors:
                for c in components:
                    print("")
                    print("----- GMM_%s (piT = %.1f) components = %.1f -----" % (t, p, c))
                    # K FOLD
                    Dfolds, Lfolds = utility.Ksplit(DTR, L, 5)
                    #Dfolds, Lfolds = numpy.array_split(D, K, axis=1), numpy.array_split(L, K)

                    scores = []
                    orderedLabels = []
                    for idx in range(K):
                        # Evaluation set
                        DV = Dfolds[idx]
                        LV = Lfolds[idx]
                        DT, LT = utility.createTrainSet(Dfolds, Lfolds, idx)
                        # DV, LV = Dfolds[idx], Lfolds[idx]
                        # DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])
                        
                        gmm = GMM.GMM(DT, LT, t, c)
                        gmm.train()
                        scores.append(gmm.getScores(DV))

                        orderedLabels.append(LV)
                    scores = numpy.hstack(scores)
                    orderedLabels = numpy.hstack(orderedLabels)
                    
                    mdcf = evaluation.minimum_DCF(scores, orderedLabels, p, 1, 1)
                    minDCF.append(mdcf)
                    print("PRIOR: %.1f, minDCF: %.3f" % (p, mdcf))
    minDCF = [0.1130952380952381, 0.07678571428571429, 0.07063492063492063, 0.09285714285714286, 0.1478174603174603, 0.2775793650793651, 0.2970238095238095, 0.24464285714285716, 0.2011904761904762, 0.2785714285714286, 0.4922619047619048, 0.7904761904761904, 0.3501984126984128, 0.20892857142857144, 0.2037698412698413, 0.23690476190476195, 0.3119047619047619, 0.47242063492063496, 0.46329365079365076, 0.20476190476190476, 0.18333333333333335, 0.1880952380952381, 0.20595238095238094, 0.21865079365079365, 0.7708333333333334, 0.5011904761904762, 0.46130952380952384, 0.4910714285714286, 0.5113095238095238, 0.4934523809523809, 0.7769841269841271, 0.5126984126984128, 0.47420634920634924, 0.48789682539682544, 0.45694444444444443, 0.5093253968253969, 0.1130952380952381, 0.1130952380952381, 0.06984126984126984, 0.06111111111111111, 0.07837301587301587, 0.0869047619047619, 0.2970238095238095, 0.2970238095238095, 0.22142857142857145, 0.21607142857142855, 0.2702380952380952, 0.25, 0.3501984126984128, 0.3501984126984128, 0.20853174603174607, 0.19980158730158734, 0.21527777777777782, 0.25019841269841275, 0.46329365079365076, 0.2152777777777778, 0.18313492063492065, 0.19702380952380952, 0.18432539682539684, 0.19027777777777777, 0.7708333333333334, 0.494047619047619, 0.4517857142857143, 0.4654761904761905, 0.4964285714285714, 0.48392857142857143, 0.7769841269841271, 0.5420634920634921, 0.47678571428571426, 0.48333333333333334, 0.4706349206349207, 0.44642857142857145, 0.1130952380952381, 0.07678571428571429, 0.07123015873015873, 0.09047619047619047, 0.12400793650793651, 0.16468253968253968, 0.2970238095238095, 0.24464285714285716, 0.2005952380952381, 0.26428571428571423, 0.39999999999999997, 0.5053571428571428, 0.3501984126984128, 0.20892857142857144, 0.2037698412698413, 0.24682539682539686, 0.26369047619047625, 0.35833333333333334, 0.46329365079365076, 0.20476190476190476, 0.18333333333333335, 0.19285714285714287, 0.19682539682539685, 0.21646825396825398, 0.7708333333333334, 0.5011904761904762, 0.461904761904762, 0.48095238095238096, 0.5011904761904762, 0.5369047619047619, 0.7769841269841271, 0.5126984126984128, 0.47420634920634924, 0.45059523809523816, 0.47440476190476194, 0.5188492063492064, 0.1130952380952381, 0.1130952380952381, 0.06765873015873015, 0.06706349206349206, 0.07757936507936508, 0.09166666666666667, 0.2970238095238095, 0.2970238095238095, 0.23690476190476192, 0.22916666666666669, 0.23869047619047618, 0.2732142857142857, 0.3501984126984128, 0.3501984126984128, 0.22242063492063496, 0.20793650793650795, 0.22063492063492066, 0.2257936507936508, 0.46329365079365076, 0.2152777777777778, 0.18432539682539684, 0.19186507936507935, 0.18174603174603174, 0.19246031746031744, 0.7708333333333334, 0.494047619047619, 0.46011904761904765, 0.44821428571428573, 0.5095238095238095, 0.4875, 0.7769841269841271, 0.5420634920634921, 0.48333333333333334, 0.4640873015873016, 0.4728174603174603, 0.43293650793650795]
    print("Plotting Histogram")
    utility.plot_GMM_histogram_1(components, minDCF, data_type) # stampa tutti e 4 insieme
    utility.plot_GMM_histogram_2(components, minDCF, data_type, types_gmm) # stampa raw data e z_normalization insieme
    print("Plotting histogram DONE!")
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

    """ -----SCORES CALIBRATION----
    #D, L = utility.load_dataset_shuffled('Train.txt') # PC roberto
    DT, LT = utility.load_dataset_shuffled('Project_ML\Train.txt') # PC gabri
    #calibration.scores_calibration(DT, LT, 'GMM_Tied_4_Components', 'gmm_tied_4_components_raw_data') # filename and title
    #calibration.scores_calibration(DT, LT, 'Logistic_Regression', 'Logistic Regression piT=0.9 lambda=0 raw_data')
    #calibration.scores_calibration(DT, LT, 'SVM', 'SVM_raw_data')
    #"""

    """ -----VALIDATION TEST-----
    DE, LE = utility.load_dataset_shuffled('Project_ML\Test.txt') # PC gabri
    DT, LT = utility.load_dataset_shuffled('Project_ML\Train.txt') # PC gabri
    test_evaluation.test_models(DT, LT, DE, LE)
    test_evaluation.test_best_3_models(DT, LT, DE, LE)
    test_evaluation.test_gmm_models(DT, LT, DE, LE)    
    #"""

