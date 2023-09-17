import numpy
import evaluation
import utility
from GMM import GMM



def scores_calibration(D, L, filename, title):
    ##### SCORE CALIBRATION
    ''' BAYES ERROR PLOT '''
    DTR = D
    p = numpy.linspace(-3, 3, 15) # BAYES Error Plot
    print('Bayes Error Plot ...')
        
    minDCF = []
    actDCF = []
    for ip in p:
        ip = 1 / (1 + numpy.exp(-ip))
        Dfolds, Lfolds = numpy.array_split(DTR, 5, axis=1), numpy.array_split(L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            DV, LV = Dfolds[idx], Lfolds[idx]
            DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])
            gmm = GMM(DT, LT, 'tied', 4)
            gmm.train()
            scores.append(gmm.compute_scores(DV))

            orderedLabels.append(LV)
        
        mdcf = evaluation.minimum_DCF(scores, orderedLabels, 0.5, 1, 1)
        mact = evaluation.actual_DCF(scores, orderedLabels, 0.5, 1, 1)

        minDCF.append(mdcf)
        actDCF.append(mact)
    utility.bayes_error_plot()

    ''' BAYES ERROR PLOT CALIBRATION'''

    ''' PRINTING PRIORS'''