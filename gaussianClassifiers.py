import generative_models as gm

class GaussianClassifier:

    def __init__(self, DT, LT, type):
        self.DT = DT
        self.LT = LT
        self.type = type

    def train(self):
        self.muList, self.covMList = gm.mvGaussian_ML_estimates(self.DT, self.LT, self.type)

    def getScores(self, DV, prior):  # this retrieves the class posterior probs (2 righe, n colonne pari al numero di samples)
        return gm.classPosteriorMVG(DV, self.muList, self.covMList, prior)

    def getloglikelihoodRatios(self, DV):  # this has to be thresholded with log-prior odds (contiene tutte le llrs)
        return gm.getloglikelihoodRatios(DV, self.muList, self.covMList)

    def getPredictions(self, DV, prior):
        postProb = self.getScores(DV, prior)
        return gm.getPredictions(postProb)