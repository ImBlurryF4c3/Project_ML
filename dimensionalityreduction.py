import utility
import numpy
import matplotlib.pyplot as plt
import scipy


def prepPCA(D):
    C = utility.covMatrix(D)
    s, U = numpy.linalg.eigh(C)# Dal momento che C è simmetrica utilizzo la funzione eigh, s contiene gli autovalori in ordine CRESCENTE
    return s, U


def PCA(D, m):
    # INPUT: Data matrix and the value of the m directions I qant to keep
    _, U = prepPCA(D)
    P = U[:, ::-1][:, 0:m] # Prendo le prime m direzioni
    DC = utility.center_data(D)
    return numpy.dot(P.T, DC)


def plotPCAvariance(D):
    s, _, _ = prepPCA(D)
    s = s[::-1]
    # Devo plottare sull'asse x il numero di dimensioni
    x_axis = [i+1 for i in range(D.shape[0])]
    y_axis = []
    for m_value in x_axis:
        y_axis.append(sum(s[:m_value])/sum(s))
    plt.figure()
    plt.plot(x_axis, y_axis, marker='.')
    plt.grid(True)
    plt.xlabel('PCA Dimensions (number of dimensions)')
    plt.ylabel('Fraction of explained variance')
    plt.title('PCA - explained variance')
    plt.show()


def computeSwSb(D, L):
    # compute matrices Sw, Sb
    SW = 0  # Within class scatter matrix -> can be computed as a weighted sum of the covariance matrices of each class
    # (Covariance matrices are computed as in PCA, si calcola il mean e si effettua la moltiplicazione element-wise (x-u)(x-u).T)
    SB = 0  # Between class scatter matrix -> 1/N * sommatoria(Nc*(muClass-mu)(muClass-mu).T)
    mu = utility.vcol(D.mean(1)) # dataset mean
    num_classes = L.max()+1
    for i in range(num_classes):
        DClass = D[:, L == i] # data only of the i-th class
        muClass = utility.vcol(DClass.mean(1)) # class mean
        SW += DClass.shape[1] * utility.covMatrix(DClass)
        SB += DClass.shape[1] * numpy.dot(muClass - mu, (muClass - mu).T)
    SW /= D.shape[1]
    SB /= D.shape[1]
    return SB, SW


def LDA(D, L, m):
    S_w, S_b = computeSwSb(D, L)
    # compute eigenvectors of Sb^(-1) * Sw
    _, U = scipy.linalg.eig(S_b, S_w)
    # now I take the m highest eigenvecs associated with the m highest eigenvalues
    W = U[:, :m]
    """ QUESTO METODO UTILIZZA NUMPY.LINALG
    S = numpy.dot(numpy.linalg.inv(S_b),S_w)
    s, U = numpy.linalg.eig(S)
    U = numpy.array(U, dtype='float64')
    i = numpy.argmax(s)
    W = utility.vcol(U[:, i])
    print(W)
    """
    # return the data projected
    return numpy.dot(W.T, D)


def LDA2(D, L, m):
    S_w, S_b = computeSwSb(D, L)
    U, s, _ = numpy.linalg.svd(S_w)
    P1 = numpy.dot(U, utility.vcol(1.0/s**0.5)*U.T)
    SBTilde = numpy.dot(P1, numpy.dot(S_b, P1.T))
    U, _, _ = numpy.linalg.svd(SBTilde)
    P2 = U[:, ::-1][:, 0:m] # Non capisco perché qua debba cambiare l'ordinr (nella soluzione proposta del lab non avviene)
    W = numpy.dot(P1.T, P2)
    return numpy.dot(W.T, D)

