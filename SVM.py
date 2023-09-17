import numpy
import utility
import scipy
from itertools import repeat


def obj_svm_wrap(H):
    def LD_obj(alpha):  # Questa funzione deve tornare sia LD(alpha) che il gradiente
        grad = (numpy.dot(H, alpha) - numpy.ones(H.shape[1]))
        f = 0.5 * numpy.dot(numpy.dot(alpha.T, H), alpha) - numpy.dot(alpha.T, numpy.ones(H.shape[1]))
        return f, grad
    return LD_obj

class SVM:

    def __init__(self, DT, LT, prior, C, K):
        self.DT = DT
        self.LT = LT
        self.prior = prior
        self.C = C
        self.K = K


    def train(self):
        # Compute the D matrix for the extended training set with K
        row = numpy.zeros(self.DT.shape[1]) + self.K
        D = numpy.vstack([self.DT, row])
        # Compute the H matrix exploiting broadcasting
        Gij = numpy.dot(D.T, D)
        # To compute zizj I need to reshape LT as a column vector
        # and a row vector
        ZT = self.LT*2-1
        zizj = numpy.dot(utility.vcol(ZT), utility.vrow(ZT))
        self.Hij = zizj * Gij
        C1 = self.C * self.prior / (self.DT[:, self.LT == 1].shape[1] / self.DT.shape[1])
        C0 = self.C * (1 - self.prior) / (self.DT[:, self.LT == 0].shape[1] / self.DT.shape[1])
        b = [(0,C1) if self.LT[i]==1 else (0, C0) for i in range(self.DT.shape[1])]
        obj = obj_svm_wrap(self.Hij)
        (x, f, d) = scipy.optimize.fmin_l_bfgs_b(obj, numpy.zeros(self.DT.shape[1]), bounds=b, factr=1.0)
        self.w = numpy.sum((x*ZT)*D, axis=1)

    def getScores(self, DV):
        row = numpy.zeros(DV.shape[1]) + self.K
        DVe = numpy.vstack([DV, row])
        return numpy.dot(self.w.T, DVe)


class PolynomialSVM():
    def __init__(self, DT, LT, prior, C, K, c, d):
        # KERNEL FUNCTION: k(xixj) = (xi.T*xj + c)^d
        self.DT = DT
        self.LT = LT
        self.prior = prior
        self.C = C
        self.K = K
        self.c = c # a new hyperparameter
        self.d = d # degree of the polynomial term

    def train(self):
        ZT = self.LT * 2 - 1
        k_DT = ((numpy.dot(self.DT.T, self.DT) + self.c)**self.d) + (self.K ** 2) # the latter is the constant to have a regularized bias term
        zizj = numpy.dot(utility.vcol(ZT), utility.vrow(ZT))
        self.Hij = zizj*k_DT
        C1 = self.C * self.prior / (self.DT[:, self.LT == 1].shape[1] / self.DT.shape[1])
        C0 = self.C * (1 - self.prior) / (self.DT[:, self.LT == 0].shape[1] / self.DT.shape[1])
        b = [(0, C1) if self.LT[i] == 1 else (0, C0) for i in range(self.DT.shape[1])]
        obj = obj_svm_wrap(self.Hij)
        alpha, _, _ = scipy.optimize.fmin_l_bfgs_b(obj, numpy.zeros(self.DT.shape[1]), bounds=b, factr=1.0)
        self.alpha = alpha

    def getScores(self, DV):
        ZT = self.LT * 2 - 1
        D = ((numpy.dot(self.DT.T, DV) + self.c)**self.d) + (self.K ** 2)
        return (utility.vcol(self.alpha) * utility.vcol(ZT) * D).sum(0)

