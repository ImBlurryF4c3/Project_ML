import numpy
import evaluation
import utility
import GMM
import logisticRegression
import generative_models
# DA SISTEMARE GLI ALTRI MODELLI, ACTDCF

def calibrated_parameters(old_scores, L):
    scores = utility.vrow(old_scores)   # vettore colonna, features della lr
    lr = logisticRegression.LogisticRegression(scores, L, 1e-4, 0.5) # lambda = 1e-4 e prior = 0.5 #### RICORDARSI DI CAMBIARE CONTROL ROOM
    model = lr.train() # ritorna self
    alpha = model.xOpt[0:-1]  # w
    beta = model.xOpt[-1]     # b 
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

    # ''' BAYES ERROR PLOT CALIBRATION'''
    # print('Calibration in progress ...')
    # minDCF = []
    # actDCF = []
    # for ip in p:
    #     ip = 1.0 / (1.0 + numpy.exp(-ip))
    #     Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
    #     scores = []
    #     orderedLabels = []
    #     for idx in range(5):
    #         DV, LV = Dfolds[idx], Lfolds[idx]
    #         DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

    #         if filename == 'GMM_Tied_4_Components':
    #             model = GMM.GMM(DT, LT, 'tied', 4)
    #             model.train()
    #             #### start calibration
    #             old_scores = model.compute_scores(DT)       # 1920
    #             alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
    #             scores_evaluation = model.compute_scores(DV)  # 480
    #             new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
    #             #### end calibration

    #         elif filename == 'MVG_tied':
    #             muList, covMlist = generative_models.mvGaussian_ML_estimates(DT, LT, 'Tied')
    #             #### start calibration
    #             old_scores = generative_models.getloglikelihoodRatios(DT, muList, covMlist)
    #             alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
    #             scores_evaluation = generative_models.getloglikelihoodRatios(DV, muList, covMlist)
    #             new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
    #             #### end calibration

    #         scores.append(new_scores)
    #         orderedLabels.append(LV)

    #     scores = numpy.hstack(scores)
    #     orderedLabels = numpy.hstack(orderedLabels)
        
    #     mdcf = evaluation.minimum_DCF(scores, orderedLabels, ip, 1, 1)
    #     mact = evaluation.actual_DCF(scores, L, ip, 1, 1)

    #     minDCF.append(mdcf)
    #     actDCF.append(mact)
    # utility.bayes_error_plot(p, minDCF, actDCF, f'calibrated_{filename}', f'calibrated_{title}')
    # print('Calibration DONE')  
    # ''' PRINTING PRIORS'''

    # print(f"Method: %s" % filename)
    # for pi in priors:
    #     Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
    #     scores = []
    #     orderedLabels = []
    #     for idx in range(5):
    #         DV, LV = Dfolds[idx], Lfolds[idx]
    #         DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])
    #         if filename == 'GMM_Tied_4_Components':
    #             model = GMM.GMM(DT, LT, 'tied', 4)
    #             model.train()
    #             #### start calibration
    #             old_scores = model.compute_scores(DT)       # 1920
    #             alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
    #             scores_evaluation = model.compute_scores(DV)  # 480
    #             new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
    #             #### end calibration
    #         elif filename == 'MVG_tied':
    #             muList, covMlist = generative_models.mvGaussian_ML_estimates(DT, LT, 'Tied')
    #             #### start calibration
    #             old_scores = generative_models.getloglikelihoodRatios(DT, muList, covMlist)
    #             alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
    #             scores_evaluation = generative_models.getloglikelihoodRatios(DV, muList, covMlist)
    #             new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
    #             #### end calibration

    #         scores.append(new_scores)
    #         orderedLabels.append(LV)
        
    #     scores = numpy.hstack(scores)
    #     orderedLabels = numpy.hstack(orderedLabels)
        
    #     mdcf = evaluation.minimum_DCF(scores, orderedLabels, pi, 1, 1)
    #     mact = evaluation.actual_DCF(scores, L, pi, 1, 1)
    #     print(f' minDCF = %.3f with prior = {pi}' % mdcf)
    #     print(f' actDCF = %.3f with prior = {pi}' % mact)
    # print('\n')





def empirical_mean(X):
    return utility.vcol(X.mean(1))


def empirical_covariance(X):
    mu = empirical_mean(X)
    C = numpy.dot((X - mu), (X - mu).T) / X.shape[1]
    return C



def ML_GAU(D):
    mu = empirical_mean(D)
    sigma = empirical_covariance(D)
    return mu, sigma


def logpdf_GAU_ND(D, mu, sigma):
    P = numpy.linalg.inv(sigma)
    c1 = 0.5 * D.shape[0] * numpy.log(2 * numpy.pi)
    c2 = 0.5 * numpy.linalg.slogdet(P)[1]
    c3 = 0.5 * (numpy.dot(P, (D - mu)) * (D - mu)).sum(0)
    return - c1 + c2 - c3

def empirical_withinclass_cov(D, labels):
    SW = 0
    for i in set(list(labels)):
        X = D[:, labels == i]
        SW += X.shape[1] * empirical_covariance(X)
    return SW / D.shape[1]

class GaussianClassifier:

    def trainClassifier(self, D, L, priors, type='MVG', tied=False):
        self.type = type
        self.tied = tied
        self.priors = priors

        self.mu0, sigma0 = ML_GAU(D[:, L == 0])
        self.mu1, sigma1 = ML_GAU(D[:, L == 1])
        if(not tied):
            self.sigma0 = sigma0
            self.sigma1 = sigma1
            if(type == 'NBG'):
                self.sigma0 *= numpy.eye(self.sigma0.shape[0], self.sigma0.shape[1])
                self.sigma1 *= numpy.eye(self.sigma1.shape[0], self.sigma1.shape[1])
        else:
            self.sigma = empirical_withinclass_cov(D, L)
            if(type == 'NBG'):
                self.sigma *= numpy.eye(self.sigma.shape[0], self.sigma.shape[1])
        return self


    def computeLLR(self, D):
        logD0 = logpdf_GAU_ND(D, self.mu0, self.sigma0 if not self.tied else self.sigma)
        logD1 = logpdf_GAU_ND(D, self.mu1, self.sigma1 if not self.tied else self.sigma)
        return logD1 - logD0
    

