import evaluation
import GMM
import logisticRegression
import SVM
import calibration
import utility
import numpy

def test_models(DT, LT, DE, LE):
    priors = [0.5, 0.1, 0.9]
    calibrated_scores = []

    DTz = utility.Znormalization(DT) # per z-normalization
    DEz = utility.Znormalization(DE)
    model_lr = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
    model_svm = SVM.SVM(DTz, LT, 0.5, 10, 1)
    model_rbfsvm_raw = SVM.RBF_SVM(DT, LT, 0.5, 10, 1, 0.001)
    model_gmm = GMM.GMM(DT, LT, 'tied', 4)


    classifiers = [model_lr, model_svm, model_rbfsvm_raw, model_rbfsvm_z, model_gmm]
    names = ['LR (piT=0.9, lambda=0)', 'SVM (piT=0.5 - C=10 - K=1 - y=0.001)', 'RBFSVM_RAW', 'GMM_Tied (4 components)']
    for idx, model in enumerate(classifiers):
        model.train()
        scoresVal_notCalibrated = model.getScores(DT)
        alpha, beta = calibration.calibrated_parameters(scoresVal_notCalibrated, LT) # alpha = w - beta = b
        scoresEval_notCalibrated = model.getScores(DE)
        scores = alpha * scoresEval_notCalibrated + beta - numpy.log(0.5 / (1 - 0.5))
        for p in priors:
            mdcf = evaluation.minimum_DCF(scores, LE, p, 1, 1)
            mact = evaluation.compute_actual_DCF(p, 1, 1, scores, LE)
            print(f'Model: %s' % names[idx])
            print(f'minDCF = %.3f (prior = {p})' % mdcf)
            print(f'actDCF = %.3f (prior = {p})' % mact)
            print("")
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

    p = numpy.linspace(-4, 4, 60)  # BAYES Error Plot
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
        model_rbfsvm_calibrated = SVM.RBF_SVM(DT, LT, 0.5, 10, 1, 0.001)  # DT, LT, prior, C, K, c, d

        model_gmm_uncalibrated.train()
        model_lr_calibrated.train()
        model_rbfsvm_calibrated.train()

        scores_gmm = model_gmm_uncalibrated.getScores(DE)
        sc_lr = model_lr_calibrated.getScores(DT)
        sc_rbfsvm = model_rbfsvm_calibrated.getScores(DT)

        alpha1, beta1 = calibration.calibrated_parameters(sc_lr, LT)  # alpha = w - beta = b
        alpha2, beta2 = calibration.calibrated_parameters(sc_rbfsvm, LT)  # alpha = w - beta = b

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
    utility.bayes_error_plot_best_3_models(p, minDCF, actDCF, 'bayes_plot_best_3_models',
                                           'Bayes plot comparing GMM - LR - RBFSVM')


def test_gmm_models(D, L, DE, LE):
    components = [1, 2, 4, 8, 16, 32]
    types_gmm = ['full_cov', 'tied']
    data_type = ['raw_data', 'z_score_data']

    minDCF = []
    for data in data_type:
        if data == 'raw_data':
            DTR = D
        else:
            DTR = utility.z_normalization(D)  # dati normalizzati
        for t in types_gmm:  # 2 plot, in base al tipo gmm e gmm tied
            for c in components:
                print("")
                print("----- GMM_%s components = %.1f %s-----" % (t, c, data))

                Dfolds, Lfolds = numpy.array_split(DTR, 5, axis=1), numpy.array_split(L, 5)
                scores = []
                orderedLabels = []
                for idx in range(5):
                    DV, LV = Dfolds[idx], Lfolds[idx]
                    DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx + 1:]), numpy.hstack(
                        Lfolds[:idx] + Lfolds[idx + 1:])

                    model = GMM.GMM(DT, LT, t, c)
                    model.train()
                    scores.append(model.getScores(DV))
                    orderedLabels.append(LV)

                sc1 = numpy.hstack(scores)
                orderedLabels = numpy.hstack(orderedLabels)

                minDCF.append(evaluation.minimum_DCF(sc1, orderedLabels, 0.5, 1, 1))

                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.5, evaluation.minimum_DCF(sc1, orderedLabels, 0.5, 1, 1)))
                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.1, evaluation.minimum_DCF(sc1, orderedLabels, 0.1, 1, 1)))
                print("PRIOR Val: %.1f, minDCF: %.3f" % (0.9, evaluation.minimum_DCF(sc1, orderedLabels, 0.9, 1, 1)))

    minDCF_test = []
    for data in data_type:
        if data == 'raw_data':
            DER = DE
            DTR = D
        else:
            DER = utility.z_normalization(DE)
            DTR = utility.z_normalization(D)
        for t in types_gmm:
            for c in components:
                print("")
                print("----- GMM_%s components = %.1f %s-----" % (t, c, data))
                model_test = GMM.GMM(DTR, L, t, c)
                model_test.train()

                sc2 = model_test.getScores(DER)

                minDCF_test.append(evaluation.minimum_DCF(sc2, LE, 0.5, 1, 1))

                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.5, evaluation.minimum_DCF(sc2, LE, 0.5, 1, 1)))
                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.1, evaluation.minimum_DCF(sc2, LE, 0.1, 1, 1)))
                print("PRIOR Test: %.1f, minDCF: %.3f" % (0.9, evaluation.minimum_DCF(sc2, LE, 0.9, 1, 1)))

    utility.plot_gmm_histogram_3(minDCF, minDCF_test, components, types_gmm)
