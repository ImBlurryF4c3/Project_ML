import numpy
import evaluation
import utility
import GMM
import logisticRegression
import generative_models

def calibrated_parameters(old_scores, L):
    scores = utility.vrow(old_scores)   # vettore colonna, features della lr
    lr = logisticRegression.LogisticRegression(scores, L, 0, 0.5) # lambda = 1e-4 e prior = 0.5 #### RICORDARSI DI CAMBIARE CONTROL ROOM
    lr.train()
    alpha = lr.xOpt[0:-1]  # w
    beta = lr.xOpt[-1]     # b
    return alpha, beta

def scores_calibration(D, L, filename, title):
    ##### SCORE CALIBRATION
    ''' BAYES ERROR PLOT '''
    p = numpy.linspace(-4, 4, 20) # BAYES Error Plot
    priors = [0.5, 0.1, 0.9]
    print('Bayes Error Plot ...') 
    minDCF = []
    actDCF = []
    for ip in p:
        ip = 1.0 / (1.0 + numpy.exp(-ip))
        Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            DV, LV = Dfolds[idx], Lfolds[idx]
            DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

            if filename == 'GMM_Tied_4_Components':
                model = GMM.GMM(DT, LT, 'tied', 4)
                model.train()
                sc = model.compute_scores(DV)
            elif filename == 'MVG_tied':
                # model = GaussianClassifier()
                # sc = model.trainClassifier(DT, LT, [0.5, 1 - 0.5], 'MVG', True).computeLLR(DV)

                muList, covMlist = generative_models.mvGaussian_ML_estimates(DT, LT, 'Tied')
                sc = generative_models.getloglikelihoodRatios(DV, muList, covMlist)
            
            scores.append(sc)
            orderedLabels.append(LV)

        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        mact = evaluation.actual_DCF(scores, L, ip, 1, 1)
        actDCF.append(mact)
        mdcf = evaluation.minimum_DCF(scores, orderedLabels, ip, 1, 1)


        minDCF.append(mdcf)

    utility.bayes_error_plot(p, minDCF, actDCF, filename, title)
    print('Bayes Error Plot DONE')