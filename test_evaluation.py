import numpy
import evaluation
import GMM
import logisticRegression
import SVM
import calibration
import utility

def test_models(DT, LT, DE, LE):

    priors = [0.5, 0.1, 0.9]
    calibrated_scores = []

    model_gmm = GMM.GMM(DT, LT, 'tied', 4)
    model_lr = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
    DTz = utility.z_normalization(DT) # per z-normalization
    model_svm = SVM.SVM(DTz, LT, 0.5, 10, 1)
    model_rbfsvm = SVM.RBF_SVM(DT, LT, 0.5, 10, 1, 0.001) # DT, LT, prior, C, K, y

    classifiers = [model_gmm, model_lr, model_svm, model_rbfsvm]
    names = ['GMM_Tied (4 components) raw_data', 'LR (piT = 0.9) raw_data', 'SVM (piT = 0.5 C = 10) z-score', 'RBFSVM (piT = 0.5 C = 10 y = 0.001) raw_data']

    for idx, model in enumerate(classifiers):
        model.train()
        sc = model.getScores(DT)
        alpha, beta = calibration.calibrated_parameters(sc, LT) # alpha = w - beta = b
        se = model.getScores(DE)
        scores = alpha * se + beta - numpy.log(0.5 / (1 - 0.5))
        for p in priors:
            mdcf = evaluation.minimum_DCF(scores, LE, p, 1, 1)
            mact = evaluation.actual_DCF(scores, LE, p, 1, 1)
            print(f'Model: %s' % names[idx])
            print(f'minDCF = %.3f (prior = {p})' % mdcf)
            print(f'actDCF = %.3f (prior = {p})' % mact)
        calibrated_scores.append(scores)

    colors = ['blue', 'red', 'darkorange', 'green']

    utility.plot_ROC_curve(names, colors, calibrated_scores, LE, 'final_classifiers', 'Final Classifiers')
    utility.plot_DET_curve(names, colors, calibrated_scores, LE, 'final_classifiers', 'Final Classifiers')
    print('DONE')
    

def test_best_3_models(DT, LT, DE, LE):
    # GMM, LR, RBFSVM
    # GMM uncalibrated
    # LR calibrate
    # RBFSVM calibrate

    p = numpy.linspace(-4, 4, 60) # BAYES Error Plot
    minDCF_gmm = []
    actDCF_gmm = []
    minDCF_lr = []
    actDCF_lr = []
    minDCF_rbfsvm = []
    actDCF_rbfsvm = []

    for ip in p:
        print(ip)
        ip = 1.0 / (1.0 + numpy.exp(-ip))
        scores_gmm = []
        scores_lr = []
        scores_rbfsvm = []

        model_gmm_uncalibrated = GMM.GMM(DT, LT, 'tied', 4)
        model_lr_calibrated = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
        model_rbfsvm_calibrated = SVM.RBF_SVM(DT, LT, 0.5, 10, 1, 0.001) # DT, LT, prior, C, K, c, d

        model_gmm_uncalibrated.train()
        model_lr_calibrated.train()
        model_rbfsvm_calibrated.train()

        scores_gmm = model_gmm_uncalibrated.getScores(DE)
        sc_lr = model_lr_calibrated.getScores(DT)
        sc_rbfsvm = model_rbfsvm_calibrated.getScores(DT)

        alpha1, beta1 = calibration.calibrated_parameters(sc_lr, LT) # alpha = w - beta = b
        alpha2, beta2 = calibration.calibrated_parameters(sc_rbfsvm, LT) # alpha = w - beta = b

        se_lr = model_lr_calibrated.getScores(DE)
        se_rbfsvm = model_rbfsvm_calibrated.getScores(DE)

        scores_lr = alpha1 * se_lr + beta1 - numpy.log(0.5 / (1 - 0.5))
        scores_rbfsvm = alpha2 * se_rbfsvm + beta2 - numpy.log(0.5 / (1 - 0.5))

        minDCF_gmm.append(evaluation.minimum_DCF(scores_gmm, LE, ip, 1, 1))
        actDCF_gmm.append(evaluation.actual_DCF(scores_gmm, LE, ip, 1, 1))
        minDCF_lr.append(evaluation.minimum_DCF(scores_lr, LE, ip, 1, 1))
        actDCF_lr.append(evaluation.actual_DCF(scores_lr, LE, ip, 1, 1))
        minDCF_rbfsvm.append(evaluation.minimum_DCF(scores_rbfsvm, LE, ip, 1, 1))
        actDCF_rbfsvm.append(evaluation.actual_DCF(scores_rbfsvm, LE, ip, 1, 1))
    
    minDCF = [minDCF_gmm, minDCF_lr, minDCF_rbfsvm]
    actDCF = [actDCF_gmm, actDCF_lr, actDCF_rbfsvm]
    utility.bayes_error_plot_best_3_models(p, minDCF, actDCF, 'bayes_plot_best_3_models', 'Bayes plot comparing GMM - LR - RBFSVM')


def test_gmm_models(D, L, DE, LE):
    components = [1, 2, 4, 8, 16, 32]
    types_gmm = ['full_cov', 'tied']
    data_type = ['raw_data', 'z_score_data']

    for t in types_gmm:         # 2 plot, in base al tipo gmm e gmm tied
        minDCF = []
        minDCF_test = []
        for title in data_type:
            if title == 'z_score_data':
                DTR = utility.z_normalization(D) # dati normalizzati
                DER = utility.z_normalization(DE) # dati normalizzati
            else:
                DTR = D
                DER = DE
            for c in components:
                print("")
                print("----- GMM_%s components = %.1f %s-----" % (t, c, title))
                model_test = GMM.GMM(DTR, L, t, c)
                model_test.train()

                Dfolds, Lfolds = numpy.array_split(DTR, 5, axis=1), numpy.array_split(L, 5)
                scores = []
                orderedLabels = []
                for idx in range(5):
                    DV, LV = Dfolds[idx], Lfolds[idx]
                    DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

                    model = GMM.GMM(DT, LT, t, c)
                    model.train()
                    scores.append(model.getScores(DV))
                    orderedLabels.append(LV)
                
                sc1 = numpy.hstack(scores)
                orderedLabels = numpy.hstack(orderedLabels)

                sc2 = model_test.getScores(DER)
                
                minDCF.append(evaluation.minimum_DCF(sc1, orderedLabels, 0.5, 1, 1))
                minDCF_test.append(evaluation.minimum_DCF(sc2, LE, 0.5, 1, 1))
                
                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.5, evaluation.minimum_DCF(sc1, LT, 0.5, 1, 1)))
                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.1, evaluation.minimum_DCF(sc1, LT, 0.1, 1, 1)))
                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.9, evaluation.minimum_DCF(sc1, LT, 0.9, 1, 1)))
                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.5, evaluation.minimum_DCF(sc2, LE, 0.5, 1, 1)))
                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.1, evaluation.minimum_DCF(sc2, LE, 0.1, 1, 1)))
                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.9, evaluation.minimum_DCF(sc2, LE, 0.9, 1, 1)))      
        utility.plot_gmm_histogram_3(minDCF, minDCF_test, components, t)