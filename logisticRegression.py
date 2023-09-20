import numpy
import utility
import scipy


class LogisticRegression:

    def __init__(self, DT, LT, lamb, prior):
        self.DT = DT
        self.LT = LT
        self.ZT = LT*2-1 # Labels = 1, 0 ==> zi = 1, -1 (serve solo a semplificare i calcoli)
        self.lamb = lamb # lamda
        self.prior = prior

    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        normTerm = 0.5 * self.lamb * (numpy.linalg.norm(w) ** 2)
        s0 = numpy.dot(w.T, self.DT[:, self.LT==0]) + b
        s1 = numpy.dot(w.T, self.DT[:, self.LT==1]) + b
        sum0 = numpy.logaddexp(0, -self.ZT[self.LT==0] * s0).sum(0)
        sum1 = numpy.logaddexp(0, -self.ZT[self.LT==1] * s1).sum(0)
        return normTerm + (self.prior/self.LT[self.LT==1].size)*sum1 + ((1-self.prior)/self.LT[self.LT==0].size)*sum0

    def train(self):
        x0 = numpy.zeros(self.DT.shape[0]+1) # è dove carico i risultati di minimizzazione per w e b. ndarray (3,)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad=True)
        self.xOpt = xOpt # in xOpt[0:-1] c'è w, in xOpt[-1] ho b
    
    def get_w_b(self):
        return self.xOpt[0:-1], self.xOpt[-1]  # w, b per la calibration

    def getScores(self, DV): # this can be tresholded with 0
        s = numpy.dot(self.xOpt[0:-1], DV) + self.xOpt[-1]
        return s

    def getloglikelihoodRatios(self, DV): # this has to be thresholded with log-prior odds
        scores = self.getScores(DV)
        # to retrieve log likelihood ratios I have to subtract the log-odds
        return scores - numpy.log(self.prior/(1-self.prior))
        #return scores - numpy.log(self.LT[self.LT == 1].shape[0] / self.LT[self.LT == 0].shape[0]) # scores - log(NT/NF) dove NT è il numero di sample della classe 1 e NF è il numero di sample della classe 0

    def getPredictions(self, DV):
        scores = self.getScores(DV)
        predictions = (scores>0).astype(int)
        return predictions
