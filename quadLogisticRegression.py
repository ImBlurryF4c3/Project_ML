import numpy
import utility
import scipy



class QLogisticRegression:

    def __init__(self, DT, LT, lamb, prior):
        self.DT = DT
        self.LT = LT
        self.ZT = LT*2-1 # Labels = 1, 0 ==> zi = 1, -1 (serve solo a semplificare i calcoli)
        self.lamb = lamb # lamda
        self.prior = prior

    def mappingFi(self, D):
        fi_x = []
        for i in range(D.shape[1]):
            x = utility.vcol(D[:, i])
            vec = utility.vcol(numpy.dot(x, x.T).flatten('F'))
            expanded = numpy.vstack((vec, x))
            fi_x.append(expanded)
        fi_x = numpy.hstack(fi_x)
        return fi_x

    def computeGradient(self, w, fi_x, s0, s1):
        den0 = numpy.exp(self.ZT[self.LT == 0] * s0) + 1
        den1 = numpy.exp(self.ZT[self.LT == 1] * s1) + 1
        sum0_grad = ((fi_x[:, self.LT == 0] * (-self.ZT[self.LT == 0])) / den0).sum(1)
        sum1_grad = ((fi_x[:, self.LT == 1] * (-self.ZT[self.LT == 1])) / den1).sum(1)
        derivative_w = self.lamb * w +(self.prior/self.LT[self.LT==1].size)*sum1_grad + ((1-self.prior)/self.LT[self.LT==0].size)*sum0_grad
        sum0_grad = (-self.ZT[self.LT == 0] / den0).sum()
        sum1_grad = (-self.ZT[self.LT == 1] / den1).sum()
        derivative_b = (self.prior/self.LT[self.LT==1].size)*sum1_grad + ((1-self.prior)/self.LT[self.LT==0].size)*sum0_grad
        return numpy.hstack((derivative_w, derivative_b))

    def logreg_obj(self, v):
        w, c = v[0:-1], v[-1]
        normTerm = 0.5 * self.lamb * (numpy.linalg.norm(w) ** 2)
        fi_x = self.fi_x

        s0 = numpy.dot(w.T, fi_x[:, self.LT==0]) + c
        s1 = numpy.dot(w.T, fi_x[:, self.LT==1]) + c
        sum0 = numpy.logaddexp(0, -self.ZT[self.LT==0] * s0).sum(0)
        sum1 = numpy.logaddexp(0, -self.ZT[self.LT==1] * s1).sum(0)

        f = normTerm + (self.prior/self.LT[self.LT==1].size)*sum1 + ((1-self.prior)/self.LT[self.LT==0].size)*sum0
        #grad = self.computeGradient(w,fi_x, s0, s1)
        return f#, grad

    def train(self):
        dim = self.DT.shape[0]**2+self.DT.shape[0]+1
        x0 = numpy.zeros(dim) # è dove carico i risultati di minimizzazione per w e c.

        # L'unica cosa che cambia è che invece di utilizzare x(cioé l'intera DT) utilizzo fi(x) = [[vec(xx.T)], [x]]
        self.fi_x = self.mappingFi(self.DT)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad=True) #, approx_grad=True
        self.xOpt = xOpt # in xOpt[0:-1] c'è w, in xOpt[-1] ho b

    def getScores(self, DV): # this can be tresholded with 0
        fi_DV = self.mappingFi(DV)
        s = numpy.dot(self.xOpt[0:-1], fi_DV) + self.xOpt[-1]
        return s

    def getloglikelihoodRatios(self, DV): # this has to be thresholded with log-prior odds
        scores = self.getScores(DV)
        # to retrieve log likelihood ratios I have to subtract the log-odds
        return scores - numpy.log(self.prior/(1-self.prior))

    def getPredictions(self, DV):
        scores = self.getScores(DV)
        predictions = (scores>0).astype(int)
        return predictions


