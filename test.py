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
    model_rbfsvm_z = SVM.RBF_SVM(DTz, LT, 0.5, 5, 1, 0.1)
    model_gmm = GMM.GMM(DT, LT, 'tied', 4)


    classifiers = [model_lr, model_svm, model_rbfsvm_raw, model_rbfsvm_z, model_gmm]
    names = ['LR (piT=0.9, lambda=0)', 'SVM (piT=0.5 - C=10 - K=1 - y=0.001)', 'RBFSVM_RAW', 'RBFSVM_Z', 'GMM_Tied (4 components)']
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
