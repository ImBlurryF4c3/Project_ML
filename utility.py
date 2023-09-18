import numpy
import matplotlib.pyplot as plt
import seaborn
import dimensionalityreduction as dr
import generative_models as gm


def vrow(v):
    return v.reshape(1, v.shape[0])


def vcol(v):
    return v.reshape(v.shape[0], 1)


def center_data(D):
    means = vcol(D.mean(axis=1))
    centeredData = D - means
    return centeredData

def Znormalization(D):
    means = vcol(D.mean(axis=1))
    stds = vcol(D.std(axis=1))
    return (D-means)/stds

def covMatrix(D):
    #Compute the covariance matrix of the input matrix
    DC = center_data(D)
    C = numpy.dot(DC, DC.T) / DC.shape[1]
    return C


def load_dataset(fname):
    samples_list = []
    labels_list = []
    with open(fname) as f:
        for line in f:
            # Ogni riga contiene le 12 features e vome ultimo valore la label
            # 0 per male, 1 per female. Prima creo una lista di vettori colonna (12,1)
            # infine concateno in verticale (vstack)
            features = line.split(',')[:-1]
            samples_list.append(vcol(numpy.array([float(xi) for xi in features])))
            labels_list.append(int(line.split(',')[-1].strip()))
    return numpy.hstack(samples_list), numpy.array(labels_list)


def plotHist(feature, D, L):
    # This function plots the histogram of a specific feature for both classes (male, female)
    # It receives in input the feature, the data matrix (D) and labels (L)
    plt.figure()
    plt.hist(D[feature, L == 0], bins=100, ec="#0000ff", density=True, alpha=0.6, label='Male')
    plt.hist(D[feature, L == 1], bins=100, ec="#d2691e", density=True, alpha=0.6, label='Female')
    plt.xlabel("Feature %d" % (feature + 1))  # Features legend goes from 1 to 12
    plt.legend()
    plt.show()


def scatterPlot(f1, f2, D, L):
    plt.figure()
    plt.xlabel("Feature %d" % (f1+1))
    plt.ylabel("Feature %d" % (f2+1))
    plt.scatter(D[f1, L==0], D[f2, L==0], label="Male")
    plt.scatter(D[f1, L == 1], D[f2, L == 1], label="Female")
    plt.legend()
    plt.show()


def heatmap(D, L):
    # xlabels and ylabels are meant to better visualize the features in the heatmap
    # I prefered to use the abs of the correlation coefficient because i am only interested
    # in the presence of the correlation, not if it's positive or negative
    xlabels = ylabels = [i + 1 for i in range(D.shape[0])]
    plt.figure()
    # Heatmap of the entire Dataset
    seaborn.heatmap(abs(numpy.corrcoef(D)), xticklabels=xlabels, yticklabels=ylabels, annot=True, cmap="Greys", square=True, cbar=True)
    plt.figure()
    # Heatmap of the Male class
    seaborn.heatmap(abs(numpy.corrcoef(D[:, L == 0])), xticklabels=xlabels, yticklabels=ylabels, annot=True, cmap="Reds", square=True, cbar=True)
    plt.figure()
    # Heatmap of the Female class
    seaborn.heatmap(abs(numpy.corrcoef(D[:, L == 1])), xticklabels=xlabels, yticklabels=ylabels, annot=True, cmap="Blues", square=True, cbar=True)
    plt.show()


# PLOTS
def plotDCF(x, y, xlabel):
    plt.figure()
    print("prior=0.5 ---> minDCF = %.3f for l = %.6f" %(min(y[0:len(x)]), x[numpy.argmin(y[0:len(x)])]))
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    print("prior=0.1 ---> minDCF = %.3f for l = %.6f" % (min(y[len(x):2*len(x)]), x[numpy.argmin(y[len(x):2*len(x)])]))
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.1', color='r')
    print("prior=0.9 ---> minDCF = %.3f for l = %.6f" % (min(y[2*len(x):3*len(x)]), x[numpy.argmin(y[2*len(x):3*len(x)])]))
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.show()
    return


def confusionMatrix(pl, l, K):
    # pl: predicted labels
    # l: actual labels
    # K: number of classes
    # Confusion matrices are a tool to visualize the number of samples predicted as class i and belonging to
    # class j. A confusion matrix is a K × K matrix, where K is number of Class, whose elements M[i,j] represent the number of samples
    # belonging to class j that are predicted as class i.
    CM = numpy.zeros((K, K), dtype = int)
    for i in range(pl.size):
        CM[pl[i], l[i]] += 1
    return CM


def split_db_singleFold(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# Per CROSS VALIDATION
def Ksplit(D, L, K, seed=27):
    # Tale funziona splitta D ed L in K folds e li torna sottoforma di liste
    nTrain = int(D.shape[1] / K)  # Grandezza di ogni fold
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    foldsD = []
    foldsL = []
    start = 0
    end = nTrain
    for _ in range(K):
        foldsD.append(D[:, idx[start:end]])
        foldsL.append(L[idx[start:end]])
        start = end
        end += nTrain
    return foldsD, foldsL


def createTrainSet(Dfolds, Lfolds, idx):
    DT = []
    LT = []
    for i in range(len(Dfolds)):
        if i == idx:
            continue
        DT.append(Dfolds[i])
        LT.append(Lfolds[i])
    return numpy.hstack(DT), numpy.hstack(LT)




