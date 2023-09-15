import numpy
import utility
import scipy


# Questa funzione corrisponde alla fase di training per un classifier gaussiano
def mvGaussian_ML_estimates(D, L, type):
    # This function computes the mean_ML and covMatrix_ML for each class
    # aka it computes the values of hyperparameters of the MVgaussian that
    # maximize the likelihood function (Maximum Likelihood estimator)
    # These values are then used to compute the log densities through logpdf_GAU_ND
    muList = []
    covMList = []
    for i in range(2): # 2 bacause it's a binary problem (here you put the number of classes)
        DClass = D[:, L == i]
        muList.append(utility.vcol(DClass.mean(1)))
        covMList.append(utility.covMatrix(DClass))
    match type:
        case 'Naive Bayes':
            # Naive assumption: covMatrix are diagonal
            for i in range(len(covMList)):
                covMList[i] *= numpy.identity(D.shape[0])
        case 'Tied':
            covM = 0
            for i in range(len(covMList)):
                covM += covMList[i]*len(L[L==i])
            covMList = [covM for x in covMList]
        case 'Tied Naive Bayes':
            covM = 0
            for i in range(len(covMList)):
                covM += covMList[i] * len(L[L == i])
            covM *= numpy.identity(D.shape[0])
            covMList = [covM for x in covMList]
    return muList, covMList





def logpdf_GAU_ND(X, mu, C):
    # ottimizzata per essere più veloce rispetto al for loop.
    # X: data matrix
    # mu: mean of the specific class
    # C: covariance matrix
    # In output ho un 1D-array con tutte le log densities (1 per ogni sample)
    M = X.shape[0]
    XC = X-mu
    const = -0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1] # Il primo valore che torna slogdet è il segno del determinante, il secondo è il valore assoluto
    L = numpy.linalg.inv(C)
    v = (XC * numpy.dot(L, XC)).sum(0) # numpy.dot(L, XC) = Y mi torna una matrice che contiene in ogni colonna il sample moltiplicato per
                                       # il corretto valore di sigma^-1. Per ottenere il vettore che contiene le log densities non posso
                                       # procedere con numpy.dot(XC.T, Y) perché conterrebbe i giusti valori solo sulla diagonale principale
                                       # essendo una matrice N*N dove N è il numero dei samples.
                                       # Quello che faccio è fare una normale moltiplicazione, non element wise, tra le due matrici, ottenendo
                                       # una matrice che su ogni colonna ha i singoli valori da sommare per ottenere la log densities, ecco
                                       # perché la sum(0) cioè rispetto alle righe
    return const - 0.5*logdet - 0.5*v


def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum(0)


# This function retrieves the log likelihood ratios for a binary task
def getloglikelihoodRatios(D, muList, covMList):
    logS = []  # lista che conterrà le log densities
    for i in range(2):  # number of classes
        fcond = logpdf_GAU_ND(D, muList[i], covMList[i])
        logS.append(utility.vrow(fcond))
    logS = numpy.vstack(logS)  # logs[i, j] => contiene la log density per il sample jesimo della classe i
    return logS[1, :] - logS[0, :]


# Utilizzo il modello trainato tramite mvGaussian_ML_estimates sul validation set
def classPosteriorMVG(D, muList, covMList, prior):
    logS = [] # lista che conterrà le log densities
    for i in range(2):  # number of classes
        fcond = logpdf_GAU_ND(D, muList[i], covMList[i])
        logS.append(utility.vrow(fcond))
    logS = numpy.vstack(logS) # logs[i, j] => contiene la log density per il sample jesimo della classe i
    prior /= (1-prior) # pi/(1-pi)
    logPrior = numpy.log(utility.vcol(numpy.ones(2)*prior)) # 2 perché ci sono diue classi
    logSJoint = logS + logPrior
    logSMarginal = utility.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal # log posterior probabilities
    SPost = numpy.exp(logSPost)
    return SPost


def getPredictions(postProb):
    predictions = numpy.argmax(postProb, axis=0)# Torna un vettore con gli indici dei massimi rispetto alle righe ->
                                                # per ogni colonna torna indice riga con valore massimo,
                                                # aka max posterior prob per il sample x
    return predictions
