import numpy
import evaluation
import utility
import GMM
import logisticRegression
import generative_models
import gaussianClassifiers
import SVM

def calibrated_parameters(old_scores, L):
    scores = utility.vrow(old_scores)   # vettore colonna, features della lr
    lr = logisticRegression.LogisticRegression(scores, L, 0, 0.5) # lambda = 1e-4 e prior = 0.5 #### RICORDARSI DI CAMBIARE CONTROL ROOM
    lr.train() # ritorna self
    alpha, beta = lr.get_w_b()
    return alpha, beta

def scores_calibration(D, L, filename, title):    ##### SCORE CALIBRATION
    ''' BAYES ERROR PLOT '''

    if filename == 'SVM':
        Dz = utility.z_normalization(D) # per z-normalization
        D = Dz
    p = numpy.linspace(-4, 4, 60) # BAYES Error Plot
    priors = [0.5, 0.1, 0.9]
    print('Bayes Error Plot ...') 
    minDCF = []
    actDCF = []
    for ip in p:
        print(ip)
        ip = 1.0 / (1.0 + numpy.exp(-ip))
        Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            DV, LV = Dfolds[idx], Lfolds[idx]
            DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

            if filename == 'GMM_Tied_4_Components':
                model = GMM.GMM(DT, LT, 'tied', 4)
                model.train()
                sc = model.getScores(DV)
            elif filename == 'Logistic_Regression':
                model = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
                model.train()
                sc = model.getScores(DV)
            elif filename == 'SVM': # piT = 0.5, in z_score con C = 10
                model = SVM.SVM(DT, LT, 0.5, 10, 1)
                model.train()
                sc = model.getScores(DV)

            scores.append(sc)
            orderedLabels.append(LV)

        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)

        mdcf = evaluation.minimum_DCF(scores, orderedLabels, ip, 1, 1)
        mact = evaluation.actual_DCF(scores, orderedLabels, ip, 1, 1)

        minDCF.append(mdcf)
        actDCF.append(mact)
    utility.bayes_error_plot(p, minDCF, actDCF, filename, title)
    print('Bayes Error Plot DONE')

    ''' BAYES ERROR PLOT CALIBRATION'''
    print('Calibration in progress ...')
    minDCF = []
    actDCF = []
    for ip in p:
        print(ip)
        ip = 1.0 / (1.0 + numpy.exp(-ip))
        Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            DV, LV = Dfolds[idx], Lfolds[idx]
            DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

            if filename == 'GMM_Tied_4_Components':
                model = GMM.GMM(DT, LT, 'tied', 4)
                model.train()
                #### start calibration
                old_scores = model.getScores(DT)       # 1920
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)  # 480
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
                #### end calibration
            elif filename == 'Logistic_Regression':
                model = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
                model.train()
                old_scores = model.getScores(DT)
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
            elif filename == 'SVM': # piT = 0.5, in z_score con C = 10
                model = SVM.SVM(DT, LT, 0.5, 10, 1)
                model.train()
                old_scores = model.getScores(DT) # 480
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))

            scores.append(new_scores)
            orderedLabels.append(LV)

        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        
        mdcf = evaluation.minimum_DCF(scores, orderedLabels, ip, 1, 1)
        mact = evaluation.actual_DCF(scores, orderedLabels, ip, 1, 1)

        minDCF.append(mdcf)
        actDCF.append(mact)
    utility.bayes_error_plot(p, minDCF, actDCF, f'calibrated_{filename}', f'calibrated_{title}')
    print('Calibration DONE')  
    ''' PRINTING PRIORS'''

    print(f"Method: %s" % filename)
    for pi in priors:
        Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
        scores = []
        orderedLabels = []
        for idx in range(5):
            DV, LV = Dfolds[idx], Lfolds[idx]
            DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])
            if filename == 'GMM_Tied_4_Components':
                model = GMM.GMM(DT, LT, 'tied', 4)
                model.train()
                #### start calibration
                old_scores = model.getScores(DT)       # 1920
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)  # 480
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
                #### end calibration
            elif filename == 'Logistic_Regression':
                model = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
                lr = model.train()
                old_scores = model.getScores(DT)
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
            elif filename == 'SVM': # piT = 0.5, in z_score con C = 10
                model = SVM.SVM(DT, LT, 0.5, 10, 1)
                model.train()
                old_scores = model.getScores(DT)
                alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
                scores_evaluation = model.getScores(DV)
                new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))

            scores.append(new_scores)
            orderedLabels.append(LV)
        
        scores = numpy.hstack(scores)
        orderedLabels = numpy.hstack(orderedLabels)
        
        mdcf = evaluation.minimum_DCF(scores, orderedLabels, pi, 1, 1)
        mact = evaluation.actual_DCF(scores, orderedLabels, pi, 1, 1)
        print(f' minDCF = %.3f with prior = {pi}' % mdcf)
        print(f' actDCF = %.3f with prior = {pi}' % mact)
    print('\n')


# def scores_calibration(D, L, filename, title):    ##### SCORE CALIBRATION

#     if filename == 'SVM':
#         Dz = utility.z_normalization(D) # per z-normalization
#         D = Dz
#     p = numpy.linspace(-4, 4, 5) # BAYES Error Plot
#     priors = [0.5, 0.1, 0.9]

#     ''' BAYES ERROR PLOT ERROR PLOT AND CALIBRATION'''
#     print('Calibration in progress ...')
#     minDCF_calibrated = []
#     actDCF_calibrated = []
#     minDCF_uncalibrated = []
#     actDCF_uncalibrated = []
#     for ip in p:
#         ip = 1.0 / (1.0 + numpy.exp(-ip))
#         Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
#         scores = []
#         calibrated_scores = []
#         orderedLabels = []
#         for idx in range(5):
#             DV, LV = Dfolds[idx], Lfolds[idx]
#             DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])

#             if filename == 'GMM_Tied_4_Components':
#                 model = GMM.GMM(DT, LT, 'tied', 4)
#                 model.train()
#                 #### start calibration
#                 old_scores = model.getScores(DT)       # 1920
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)  # 480
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
#                 #### end calibration
#             elif filename == 'Logistic_Regression':
#                 model = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
#                 model.train()
#                 old_scores = model.getScores(DT)
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
#             elif filename == 'SVM': # piT = 0.5, in z_score con C = 10
#                 model = SVM.SVM(DT, LT, 0.5, 10, 1)
#                 model.train()
#                 old_scores = model.getScores(DT) # 480
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))

#             calibrated_scores.append(old_scores)
#             scores.append(new_scores)
#             orderedLabels.append(LV)

#         scores = numpy.hstack(scores)
#         calibrated_scores = numpy.hstack(calibrated_scores)
#         orderedLabels = numpy.hstack(orderedLabels)
        
#         mdcf = evaluation.minimum_DCF(scores, orderedLabels, ip, 1, 1)
#         mact = evaluation.actual_DCF(scores, orderedLabels, ip, 1, 1)

#         mdcf_c = evaluation.minimum_DCF(calibrated_scores, orderedLabels, ip, 1, 1)
#         mact_c = evaluation.actual_DCF(calibrated_scores, orderedLabels, ip, 1, 1)

#         minDCF_uncalibrated.append(mdcf)
#         actDCF_uncalibrated.append(mact)
#         minDCF_calibrated.append(mdcf_c)
#         actDCF_calibrated.append(mact_c)
#     utility.bayes_error_plot(p, minDCF_uncalibrated, actDCF_uncalibrated, filename, title)
#     utility.bayes_error_plot(p, minDCF_calibrated, actDCF_calibrated, f'calibrated_{filename}', f'calibrated_{title}')

#     print('Calibration DONE')  
#     ''' PRINTING PRIORS'''

#     print(f"Method: %s" % filename)
#     for pi in priors:
#         print(ip)
#         Dfolds, Lfolds = numpy.array_split(D, 5, axis=1), numpy.array_split(L, 5)
#         scores = []
#         orderedLabels = []
#         for idx in range(5):
#             DV, LV = Dfolds[idx], Lfolds[idx]
#             DT, LT = numpy.hstack(Dfolds[:idx] + Dfolds[idx+1:]), numpy.hstack(Lfolds[:idx] + Lfolds[idx+1:])
#             if filename == 'GMM_Tied_4_Components':
#                 model = GMM.GMM(DT, LT, 'tied', 4)
#                 model.train()
#                 #### start calibration
#                 old_scores = model.getScores(DT)       # 1920
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)  # 480
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
#                 #### end calibration
#             elif filename == 'Logistic_Regression':
#                 model = logisticRegression.LogisticRegression(DT, LT, 0, 0.9)
#                 lr = model.train()
#                 old_scores = model.getScores(DT)
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))
#             elif filename == 'SVM': # piT = 0.5, in z_score con C = 10
#                 model = SVM.SVM(DT, LT, 0.5, 10, 1)
#                 model.train()
#                 old_scores = model.getScores(DV)
#                 alpha, beta = calibrated_parameters(old_scores, LT) # alpha = w - beta = b
#                 scores_evaluation = model.getScores(DV)
#                 new_scores = alpha * scores_evaluation + beta - numpy.log(0.5 / (1 - 0.5))

#             scores.append(new_scores)
#             orderedLabels.append(LV)
        
#         scores = numpy.hstack(scores)
#         orderedLabels = numpy.hstack(orderedLabels)
        
#         mdcf = evaluation.minimum_DCF(scores, orderedLabels, pi, 1, 1)
#         mact = evaluation.actual_DCF(scores, orderedLabels, pi, 1, 1)
#         print(f' minDCF = %.3f with prior = {pi}' % mdcf)
#         print(f' actDCF = %.3f with prior = {pi}' % mact)
#     print('\n')



