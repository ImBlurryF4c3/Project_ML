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
    model_rbfsvm = SVM.PolynomialSVM(DT, LT, 0.5, 0.001, 1, 1, 2) # DT, LT, prior, C, K, c, d

    classifiers = [model_gmm, model_lr, model_svm, model_rbfsvm]
    names = ['GMM_Tied (4 components)', 'LR (piT = 0.9)', 'SVM (piT = 0.5 - C = 10 - K = 1)', 'RBFSVM']

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
    
