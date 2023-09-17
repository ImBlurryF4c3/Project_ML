import numpy
import utility
from cmath import inf


def optDecision_binary(prior, Cfn, Cfp, LLRs, L):
    # In output ho la confusion matrix ottenuta dalle predizioni con l'optimal treshold
    # (prior, Cfn, Cfp): working point
    # LLRs: log likelihood ratios
    # L: labels associate (Non predictions, ma label effettive)
    # compute optimal treshold
    t = -numpy.log((prior*Cfn)/((1-prior)*Cfp))
    # if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictedLabels = (LLRs > t).astype(int)
    CM = utility.confusionMatrix(predictedLabels, L, 2)
    return CM


def compute_DCF(CM, prior, Cfn, Cfp):
    # CM: Confusion Matrix
    # (prior, Cfn, Cfp): working point
    FPR = CM[1,0]/(CM[1,0]+CM[0,0]) # FP/(FP+TN), FP = predetti come positivi (i = 1) ma in realtà negativi (j = 0)
    FNR = CM[0,1]/(CM[0,1]+CM[1,1])# FN/(FN+TP), FN = predetti come negativi (i = 0) ma in realtà positivi (j = 1)
    # As we have seen, we can compute the empirical Bayes risk (or detection cost function, DCF),
    # that represents the cost that we pay due to our decisions c∗ for the test data.
    DCFu = FNR*prior*Cfn + FPR*(1-prior)*Cfp
    return DCFu


def normalized_DCF(DCFu, prior, Cfn, Cfp):
    dummy = min(prior*Cfn, (1-prior)*Cfp)
    return DCFu/dummy


def minimum_DCF(llrs, LTE, prior, Cfn, Cfp):
    # llrs: log likelihood ratios
    # LTE: label del validation set
    # dalle postProb ricavo le predictions e i valori massimi
    # sort
    # ottimizzazione per confusion matrix

    # First of all we sort the llrs with the associated labels
    toSortProp = sorted(list(zip(llrs, LTE))) # mi appoggio a questa lista di tuple per fare un sorting congruente
    sortedLLRs = numpy.array([t[0] for t in toSortProp])
    sortedL = numpy.array([t[1] for t in toSortProp])
    # We compute the first Confusion Matrix as t = -inf, so evrything is labeled as 1
    PL = (sortedLLRs > float(-inf)).astype(int)
    CM = utility.confusionMatrix(PL, sortedL, 2)
    # lista che conterrà le varie dcf al variare di t
    DCFlist = []
    DCFlist.append(normalized_DCF(compute_DCF(CM, prior, Cfn, Cfp), prior, Cfn, Cfp))
    for i in range(sortedLLRs.size):
        # Ogni iterazione muta la label di un solo sample: da True passa a False
        # Se la Label associatagli era effettivamente False, la predizione precedente (= True) era incorretta (aka -1 nei False Positive)
        if (sortedL[i] == False):
            CM[1, 0] -= 1
            CM[0, 0] += 1
        else:  # Significa che la label era True e quindi precedentemente era stata associata correttamente (aka -1 nei True Positive)
            CM[1, 1] -= 1
            CM[0, 1] += 1
        DCFlist.append(normalized_DCF(compute_DCF(CM, prior, Cfn, Cfp), prior, Cfn, Cfp))
    return min(DCFlist)


""" 
def minimum_DCF_NotOpt(llrs, LTE, prior, Cfn, Cfp):
    toSortProp = sorted(list(zip(llrs, LTE)))  # mi appoggio a questa lista di tuple per fare un sorting congruente
    sortedLLRs = numpy.array([t[0] for t in toSortProp])
    sortedL = numpy.array([t[1] for t in toSortProp])
    DCFlist = []
    for t in llrs:
        pl = (sortedLLRs > t).astype(int)
        CM = utility.confusionMatrix(pl, sortedL, 2)
        DCFlist.append(normalized_DCF(compute_DCF(CM, prior, Cfn, Cfp), prior, Cfn, Cfp))
    return min(DCFlist)
"""