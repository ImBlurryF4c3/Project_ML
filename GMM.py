# FILE CHE CONTIENE IL MAIN PER FARE LE VARIE PROVE DEI MODULI
import numpy
import utility
import scipy
import scipy.special
from generative_models import logpdf_GAU_ND

def logpdf_GMM(D, gmm):
    S = numpy.zeros((len(gmm), D.shape[1]))
    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = logpdf_GAU_ND(D, mu, C) + numpy.log(w)
    logD = scipy.special.logsumexp(S, axis=0)
    return S, logD

class GMM:
    def __init__(self, DT, LT, type, c):
        self.DT = DT
        self.LT = LT
        self.type = type
        self.components = c
        self.gmm0 = None
        self.gmm1 = None

    def train(self):
        D0 = self.DT[:, self.LT == 0]
        D1 = self.DT[:, self.LT == 1]

        self.gmm0 = self._gmm_LBG(D0, self.components, type)
        self.gmm1 = self._gmm_LBG(D1, self.components, type)

    def _gmm_LBG(self, D, components, type):
        alpha = 1e-1
        initial_mu = utility.vcol(D.mean(1))
        initial_sigma = utility.covMatrix(D)
        gmm = [(1, initial_mu, initial_sigma)]

        while len(gmm) <= components:
            gmm = self._gmm_em(D, gmm, type)
            if(len(gmm) == components):
                break
            new_GMM = []
            for i in range(len(gmm)):
                (w, mu, sigma) = gmm[i]
                U, s, Vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * (s[0] ** 0.5) * alpha
                new_GMM.append((w / 2, mu + d, sigma))
                new_GMM.append((w / 2, mu - d, sigma))
            gmm = new_GMM
        return gmm
    
    def _gmm_em(self, DT, gmm, type):
        diff = 1e-6
        psi = 1e-2
        D, N = DT.shape
        new_ll = None
        old_ll = None
        G = len(gmm)

        while new_ll == None or old_ll - new_ll > diff:
            new_ll = old_ll

            S, logD = logpdf_GMM(DT, gmm)
            old_ll = logD.sum() / N
            P = numpy.exp(S - logD)

            new_GMM = []
            sigmaTied = numpy.zeros((D, D))
            for i in range(G):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (utility.vrow(gamma) * DT).sum(1)
                S = numpy.dot(DT, (utility.vrow(gamma) * DT).T)
                w = Z/P.sum()
                mu = utility.vcol(F / Z)
                sigma = (S / Z) - numpy.dot(mu, mu.T)

                # types = ['full_cov', 'tied', 'diagonal', 'tied_diagonal']
                if type == 'tied':
                    sigmaTied += Z * sigma
                    new_GMM.append((w, mu, sigma))
                    continue
                elif type == 'diagonal':
                    sigma *= numpy.eye(sigma.shape[0])
                elif type == 'tied_diagonal':
                    sigma *= numpy.eye(sigma.shape[0]) # diagonalization
                    sigmaTied += Z * sigma # per il tying
                    new_GMM.append((w, mu, sigma))
                    continue

                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, utility.vcol(s) * U.T)
                new_GMM.append((w, mu, sigma))

            if type == 'tied':
                sigmaTied /= N
                U, s, _ = numpy.linalg.svd(sigmaTied)
                s[s<psi] = psi
                sigmaTied = numpy.dot(U, utility.vcol(s) * U.T)
                new_GMM2 = []
                for i in range(len(new_GMM)):
                    (w, mu, _) = new_GMM[i]
                    new_GMM2.append((w, mu, sigmaTied))
                new_GMM = new_GMM2

            if type == 'tied_diagonal':
                sigma = sigmaTied / N
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, utility.vcol(s) * U.T)
                new_GMM2 = []
                for i in range(len(new_GMM)):
                    (w, mu, _) = new_GMM[i]
                    new_GMM2.append((w, mu, sigma))
                new_GMM = new_GMM2
            gmm = new_GMM

        return gmm

    def getScores(self, D):
        S, logD0 = logpdf_GMM(D, self.gmm0)
        S, logD1 = logpdf_GMM(D, self.gmm1)
        return logD1 - logD0



