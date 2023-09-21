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
def Ksplit(D, L, K, seed=0):
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



# #### DA AGGIUNGERE ******************************+
def z_normalization(D):
    mu = vcol(D.mean(1))                # z-normalization
    std = vcol(numpy.std(D, axis=1))
    D = (D - mu) / std
    return D

def plot_GMM_histogram_1(components, y, data, defPath = ''):
    # stampa tutti e 4 i modelli insieme: full-cov, diag, tied, tied-diag
    for i in range(2): # raw_data e z_normalization
        fig = plt.figure()
        fc, d, t, td = [], [], [], []
        for j in range(len(components)):
            fc.append(y[j + (i * 12 * len(components))])
            d.append(y[j + (3 * len(components)) + (i * 12 * len(components))])
            t.append(y[j + (6 * len(components)) + (i * 12 * len(components))])
            td.append(y[j + (9 * len(components)) + (i * 12 * len(components))])
        max_y_value = max(max(max(fc), max(d)), max(max(t), max(td)))
        bar_width = 0.1
        border_width = 1.0  # Set the border width
        x = numpy.arange(len(components))

        plt.bar(x - 1.5 * bar_width, fc, bar_width, label='minDCF(π~ = 0.5) - Full-Cov', color='yellow', edgecolor='black', linewidth=border_width)
        plt.bar(x - 0.5 * bar_width, d, bar_width, label='minDCF(π~ = 0.5) - Diagonal', color='orange', edgecolor='black', linewidth=border_width)
        plt.bar(x + 0.5 * bar_width, t, bar_width, label='minDCF(π~ = 0.5) - Tied', color='red', edgecolor='black', linewidth=border_width)
        plt.bar(x + 1.5 * bar_width, td, bar_width, label='minDCF(π~ = 0.5) - Tied-Diagonal', color='darkred', edgecolor='black', linewidth=border_width)
        
        plt.ylim(0, max_y_value + 0.1)
        plt.xlabel('GMM Components')
        plt.ylabel('minDCF')
        plt.xticks(x, components)
        plt.title("%s with prior = 0.5" % data[i])

        plt.legend()


        plt.savefig(defPath + 'Project_ML/images/gmm/%s.jpg' % data[i], dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_GMM_histogram_2(components, y, data, types, defPath = ''): # x sono i components, y le mindcf
    # stampa insieme per ogni componente il valore dei raw data e
    bar_width = 0.1
    x = numpy.arange(len(components))
    border_width = 1.0  # Set the border width
    for i in range(4): # per ogni modello full-cov, diag, tied, tied-cov
        fig = plt.figure()
        plt.title('GMM %s with prior = 0.5' % types[i])
        raw_data = []
        z_normalization_data = []
        for j in range(len(components)):
            raw_data.append(y[j + (3*i*len(components))])
            z_normalization_data.append(y[j +  (12*len(components)) + (3*i*len(components))])
        max_y_value = max(max(raw_data), max(z_normalization_data))
        plt.bar(x - 0.5 * bar_width, raw_data, bar_width, label='raw_data', color='orange', edgecolor='black', linewidth=border_width)
        plt.bar(x + 0.5 * bar_width, z_normalization_data, bar_width, label='z_score', color='red', edgecolor='black', linewidth=border_width)
        
        # Set labels and title
        plt.xlabel('GMM Components')
        plt.ylabel('minDCF')
        plt.xticks(x, components)
        
        plt.legend()
        plt.ylim(0, max_y_value + 0.1)

        plt.savefig(defPath + 'Project_ML/images/gmm/%s.jpg' % types[i], dpi=300, bbox_inches='tight')
        plt.close(fig)

def load_dataset_shuffled(fname):
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
    D = numpy.hstack(samples_list)
    L = numpy.array(labels_list)
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    return D[:, idx], L[idx]

def bayes_error_plot(p, minDCF, actDCF, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(p, numpy.array(actDCF), label = 'actDCF', color='salmon')
    plt.plot(p, numpy.array(minDCF), label = 'minDCF', color='dodgerblue', linestyle='--')
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.xlabel('prior')
    plt.ylabel('DCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'Project_ML/images/calibration/%s_bayes_error_plot.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_ROC_curve(models, colors, calibrated_scores, LE, filename, title):
    figure = plt.figure()
    for i, scores in enumerate(calibrated_scores):
        threshold = numpy.array(scores)
        threshold.sort()
        threshold = numpy.concatenate([numpy.array([-numpy.inf]), threshold, numpy.array([numpy.inf])])
        FPR = [] # false positive ratio
        FNR = [] # false negative ratio
        for t in threshold:
            Predictions = (scores > t).astype(int)
            CM = confusionMatrix(Predictions, LE, 2)
            FPR.append(CM[1, 0]/(CM[1, 0] + CM[0, 0]))
            FNR.append(CM[0, 1]/(CM[0, 1] + CM[1, 1]))
        FPR = numpy.array(FPR)      # false positive ratio
        TPR = 1 - numpy.array(FNR)  # true positvie ratio
        plt.plot(FPR, TPR, label = models[i], color = colors[i])
    plt.xlabel('False Positive Ratio')
    plt.ylabel('True Positive Ratio')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('Project_ML/images/evaluation/roc_curve_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(figure)
    

def plot_DET_curve(models, colors, calibrated_scores, LE, filename, title):
    figure = plt.figure()
    for i, scores in enumerate(calibrated_scores):
        threshold = numpy.array(scores)
        threshold.sort()
        threshold = numpy.concatenate([numpy.array([-numpy.inf]), threshold, numpy.array([numpy.inf])])
        FPR = [] # false positive ratio
        FNR = [] # false negative ratio
        for t in threshold:
            Predictions = (scores > t).astype(int)
            CM = confusionMatrix(Predictions, LE, 2)
            FPR.append(CM[1, 0]/(CM[1, 0] + CM[0, 0]))
            FNR.append(CM[0, 1]/(CM[0, 1] + CM[1, 1]))
        FPR = numpy.array(FPR)      # false positive ratio
        FNR = numpy.array(FNR)      # false negative ratio
        plt.plot(FPR, FNR, label = models[i], color = colors[i])
    plt.xlabel('False Positive Ratio')
    plt.ylabel('False Negative Ratio')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('Project_ML/images/evaluation/det_curve_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(figure)


def bayes_error_plot_best_3_models(p, minDCF, actDCF, filename, title):
    names_min = ['GMM (4 components)', 'LR', 'RBFSVM']
    names_act = ['GMM (4 components) uncalibrated', 'LR calibrated', 'RBFSVM calibrated']
    fig = plt.figure()
    colors = ['red', 'blue', 'orange']
    for i in range(3):
        plt.plot(p, numpy.array(minDCF[i]), label = 'minDCF %s' % names_min[i], color=colors[i], linestyle='--')
        plt.plot(p, numpy.array(actDCF[i]), label = 'actDCF %s' % names_act[i], color=colors[i])
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.xlabel('prior')
    plt.ylabel('DCF')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig('Project_ML/images/evaluation/%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gmm_histogram_3(minDCF, minDCF_test, components, model_type):
    # plot raw_datam z_score for gmm and gmm tied for training and test data
    figure = plt.figure()
    bar_width = 0.1
    l = len(components)
    x = numpy.arange(l)
    border_width = 1.0
    
    val_raw = minDCF[:l]
    val_z = minDCF[l:]
    test_raw = minDCF_test[:l]
    test_z = minDCF_test[l:]
    
    max_y_value = max(max(max(val_raw), max(val_z)), max(max(test_raw), max(test_z)))

    plt.bar(x - 1.5 * bar_width, val_raw, bar_width, label='Validation raw_data', color='mediumpurple', edgecolor='black', hatch='/', linewidth=border_width, linestyle='--')
    plt.bar(x - 0.5 * bar_width, test_raw, bar_width, label='Test raw_data', color='mediumpurple', edgecolor='black', linewidth=border_width)
    plt.bar(x + 0.5 * bar_width, val_z, bar_width, label='Validation z-score', color='gold', edgecolor='black', hatch='/', linewidth=border_width, linestyle='--')
    plt.bar(x + 1.5 * bar_width, test_z, bar_width, label='Test z-score', color='gold', edgecolor='black', linewidth=border_width)

    plt.title('GMM %s piT = 0.5' % model_type)
    plt.xlabel('GMM Components')
    plt.ylabel('minDCF')
    plt.xticks(x, components)
    plt.legend(loc='best')
    plt.ylim(0, max_y_value + 0.1)
    plt.savefig('Project_ML/images/evaluation/evaluation_histogram_gmm_%s.jpg' % model_type, dpi=300, bbox_inches='tight')

    plt.close(figure)