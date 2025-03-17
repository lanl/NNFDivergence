import numpy as np 
import os 
import pickle
from scipy.stats import ttest_rel
     
"""
Calculate overall RMSE 
"""
def CalculateOverallRMSEGivenPredandName(Pred, SampleName, TrueLabel_Dic, Oxide_Names, OxideID=-1):
    # Get label number 
    if OxideID == -1: # Single oxide-weight prediction
        Target_Weights = np.zeros((0, len(Oxide_Names)))
        Prediction_Weights = np.zeros((0, len(Oxide_Names))) 
    else: # Full multi-oxide weight predictions 
        Target_Weights = np.zeros((0, 1))
        Prediction_Weights = np.zeros((0, 1)) 
    for j, name in enumerate(SampleName):
        if OxideID == -1:
            Target_Weights = np.vstack((Target_Weights, TrueLabel_Dic[name]))
            Prediction_Weights = np.vstack((Prediction_Weights, Pred[name] * 100))
        else:
            Target_Weights = np.vstack((Target_Weights, np.ones((1,1)) * TrueLabel_Dic[name][OxideID]))
            Prediction_Weights = np.vstack((Prediction_Weights, np.ones((1,1)) * Pred[name] * 100))          

    # RMSE
    MSE = np.mean((Target_Weights - Prediction_Weights) ** 2)
    RMSE = np.sqrt(MSE)
    return RMSE

"""
Calculate RMSE for every oxide weight prediction averaging many runs 
"""
def CalculateEveryRMSEforManyRuns(PredManyRuns, SampleNameManyRuns, TrueLabel_Dic, Oxide_Names, OxideID = []):
    # Get label number 
    
    MSE_Err_EveryOxides = np.zeros((0, len(Oxide_Names)))
    for i in range(len(PredManyRuns)):
        Pred = PredManyRuns[i]; SampleName = SampleNameManyRuns[i]
        for j, name in enumerate(SampleName):
            # Target_Weights = np.vstack((Target_Weights, TrueLabel_Dic[name]))
            # # debug
            # # print(Prediction_Weights.shape, Pred[name].shape)
            # Prediction_Weights = np.vstack((Prediction_Weights, Pred[name] * 100)) 
            if len(OxideID) == 0:
                MSE_Err_EveryOxides = np.vstack((MSE_Err_EveryOxides, (TrueLabel_Dic[name] - Pred[name] * 100) ** 2))  
            else:
                Target_Weight = np.zeros(len(OxideID))
                for i, ID in enumerate(OxideID):
                    # debug
                    # print(TrueLabel_Dic[name])
                    Target_Weight[i] = TrueLabel_Dic[name][ID] 
                MSE_Err_EveryOxides = np.vstack((MSE_Err_EveryOxides, (Target_Weight - Pred[name] * 100) ** 2))

    # Calculate Overall RMSE
    RMSE_Err_EveryRun = np.sqrt(np.mean(MSE_Err_EveryOxides, axis = 1))
    RMSE_Mean = np.mean(RMSE_Err_EveryRun)
    RMSE_Std = np.std(RMSE_Err_EveryRun)
    # print('%.2f$\\pm$ %.2f&'%(RMSE_Mean, RMSE_Std), end='')
    print('%.2f&'%(RMSE_Mean), end='')

    # Calculate RMSE for each oxide weight
    for i, name in enumerate(Oxide_Names):
        MSE_Err_A_Oxides = MSE_Err_EveryOxides[:, i]
        RMSE_Err_A_Oxides = np.sqrt(MSE_Err_A_Oxides)
        RMSE_Mean = np.mean(RMSE_Err_A_Oxides)
        RMSE_Std = np.std(RMSE_Err_A_Oxides)       
        if i == len(Oxide_Names) - 1:
            # print('%.2f$\\pm$ %.2f'%(RMSE_Mean, RMSE_Std))
            print('%.2f'%(RMSE_Mean))
        else:
            # print('%.2f$\\pm$ %.2f&'%(RMSE_Mean, RMSE_Std), end='')
            print('%.2f&'%(RMSE_Mean), end='')
    # Return RMSE
    return np.sqrt(MSE_Err_EveryOxides)
"""
Calculate RMSE for every oxide weight prediction with one run 
"""
def CalculateEveryRMSEforOneRun(PredOneRun, SampleNameOneRun, TrueLabel_Dic, Oxide_Names):

    # Measurement variables 
    R = np.zeros((0, len(Oxide_Names) + 1))

    # Acquire all measurements and save them in R 
    for j, name in enumerate(SampleNameOneRun):
        Target_Weights = np.zeros((0, len(Oxide_Names)))
        Prediction_Weights = np.zeros((0, len(Oxide_Names)))
        Target_Weights = np.vstack((Target_Weights, TrueLabel_Dic[name]))
        # debug
        Prediction_Weights = np.vstack((Prediction_Weights, PredOneRun[name]))  
        MultiOxideRMSE = (Target_Weights - Prediction_Weights * 100) ** 2
        AllMultiOxideRMSE = np.zeros((1, len(Oxide_Names) + 1))
        AllMultiOxideRMSE[0, :-1] = np.sqrt(MultiOxideRMSE)
        AllMultiOxideRMSE[0,-1] = np.sqrt(np.mean(MultiOxideRMSE))
        R = np.vstack((R, AllMultiOxideRMSE.reshape((1,-1))))

    return R


"""
Measure the significance by the matched-pair test
"""
def MeasurePairedTest(args, ResultDir):
    # set directory parameters
    cwd = os.getcwd() 

    if args.DataType == 'PDS_ChemCam':
        DataDir = cwd + '/PDS_ChemCam/Data/'
        Oxide_Names =  ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    elif args.DataType == 'PDS_SuperCam':
        pass 

    # Load true labels 
    TrueLabel_Dic = {}
    if args.DataType == 'PDS_ChemCam':
        Labels = pickle.load(open(DataDir + 'ChemCam_Mars_labels.pickle', "rb" ))
        SampleName = Labels.keys()
        for name in SampleName:
            TrueLabel_Dic[name] = Labels[name]
    elif args.DataType == 'PDS_SuperCam':
        pass     

    if args.LossType3 == 'NoReg_fdiv':
        # Measurements from without regurization and training with f-divergence
        Control = np.zeros((0,len(Oxide_Names) + 1)); Treatment = np.zeros((0,len(Oxide_Names) + 1));
        # set up regularization parameters 
        Control_RegStrengths = [0];     
        Treatment_RegStrengths = [(0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014), 
                         (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015), 
                         (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)]  
        print('Conducting matched-pair t-test between no regularization and f-divergence regularization')
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Control_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # For treatment 
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths)); 
            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)
            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))
            
        # Matched-Pair TTest
        print('Left and right side t-test')

        left_statistic, left_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'less')
        right_statistic, right_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'greater')
        
        # Print oxide names
        print('All oxides', end = ' ')
        for i in range(len(Oxide_Names)):
            if i != len(Oxide_Names) - 1:
                print(Oxide_Names[i], end = ' ')
            else:
                print(Oxide_Names[i])
        
        if left_pvalue <= 0.1:
            print('Better&', end='')
        elif right_pvalue <= 0.1:
            print('Worse&', end='')
        else:
            print('-&', end='')

        # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'greater')
            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
            if i == len(Oxide_Names) - 1:
                if left_pvalue <= 0.1:
                    print('Better')
                elif right_pvalue <= 0.1:
                    print('Worse')
                else:
                    print('-')
            else:
                if left_pvalue <= 0.1:
                    print('Better&', end='')
                elif right_pvalue <= 0.1:
                    print('Worse&', end='')
                else:
                    print('-&', end='')

    elif args.LossType3 == 'l2_fdiv':
        # Measurements from without regurization and training with f-divergence
        Control = np.zeros((0,len(Oxide_Names) + 1)); Treatment = np.zeros((0,len(Oxide_Names) + 1));
        # set up regularization parameters 
        Control_RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001];    
        Treatment_RegStrengths = [(0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014), 
                         (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015), 
                         (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)]  
        print('Conducting matched-pair t-test between l2 and f-divergence regularizations')
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Control_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # For treatment 
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths)); 
            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)
            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))
            
        # Matched-Pair TTest
        print('Left and right side t-test')
        left_statistic, left_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'less')
        right_statistic, right_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'greater')

        # Print oxide names
        print('All oxides', end = ' ')
        for i in range(len(Oxide_Names)):
            if i != len(Oxide_Names) - 1:
                print(Oxide_Names[i], end = ' ')
            else:
                print(Oxide_Names[i])

        # Print out paired-testing results 
        if left_pvalue <= 0.1:
            print('Better&', end='')
        elif right_pvalue <= 0.1:
            print('Worse&', end='')
        else:
            print('-&', end='')

        # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'greater')
            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
            if i == len(Oxide_Names) - 1:
                if left_pvalue <= 0.1:
                    print('Better')
                elif right_pvalue <= 0.1:
                    print('Worse')
                else:
                    print('-')
            else:
                if left_pvalue <= 0.1:
                    print('Better&', end='')
                elif right_pvalue <= 0.1:
                    print('Worse&', end='')
                else:
                    print('-&', end='')
    elif args.LossType3 == 'Dropout_fdiv':
        # Measurements from without regurization and training with f-divergence
        Control = np.zeros((0,len(Oxide_Names) + 1)); Treatment = np.zeros((0,len(Oxide_Names) + 1));
        # set up regularization parameters 
        Control_RegStrengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1];  
        Treatment_RegStrengths = [(0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014), 
                         (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015), 
                         (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)]  
        print('Conducting matched-pair t-test between dropout and f-divergence regularizations')
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Control_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # For treatment 
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths)); 
            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)
            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))
            
        # Matched-Pair TTest
        print('Left and right side t-test')
        left_statistic, left_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'less')
        right_statistic, right_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'greater')

        # Print oxide names
        print('All oxides', end = ' ')
        for i in range(len(Oxide_Names)):
            if i != len(Oxide_Names) - 1:
                print(Oxide_Names[i], end = ' ')
            else:
                print(Oxide_Names[i])

        # Print out paired-testing results 
        if left_pvalue <= 0.1:
            print('Better&', end='')
        elif right_pvalue <= 0.1:
            print('Worse&', end='')
        else:
            print('-&', end='')

        # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'greater')
            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
            if i == len(Oxide_Names) - 1:
                if left_pvalue <= 0.1:
                    print('Better')
                elif right_pvalue <= 0.1:
                    print('Worse')
                else:
                    print('-')
            else:
                if left_pvalue <= 0.1:
                    print('Better&', end='')
                elif right_pvalue <= 0.1:
                    print('Worse&', end='')
                else:
                    print('-&', end='')

    elif args.LossType3 == 'l1_fdiv':
        # Measurements from without regurization and training with f-divergence
        Control = np.zeros((0,len(Oxide_Names) + 1)); Treatment = np.zeros((0,len(Oxide_Names) + 1));
        # set up regularization parameters 
        Control_RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001]; 
        Treatment_RegStrengths = [(0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014), 
                         (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015), 
                         (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)]  
        print('Conducting matched-pair t-test between l1 and f-divergence regularizations')
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Control_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # For treatment 
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths)); 
            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)
            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter, with one run 
            Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))
            
        # Matched-Pair TTest
        print('Left and right side t-test')
        left_statistic, left_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'less')
        right_statistic, right_pvalue= ttest_rel(Treatment[:, -1], Control[:, -1], alternative = 'greater')

        # Print oxide names
        print('All oxides', end = ' ')
        for i in range(len(Oxide_Names)):
            if i != len(Oxide_Names) - 1:
                print(Oxide_Names[i], end = ' ')
            else:
                print(Oxide_Names[i])

        # Print out paired-testing results 
        if left_pvalue <= 0.1:
            print('Better&', end='')
        elif right_pvalue <= 0.1:
            print('Worse&', end='')
        else:
            print('-&', end='')

        # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(Treatment[:, i], Control[:, i], alternative = 'greater')
            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
            if i == len(Oxide_Names) - 1:
                if left_pvalue <= 0.1:
                    print('Better')
                elif right_pvalue <= 0.1:
                    print('Worse')
                else:
                    print('-')
            else:
                if left_pvalue <= 0.1:
                    print('Better&', end='')
                elif right_pvalue <= 0.1:
                    print('Worse&', end='')
                else:
                    print('-&', end='')

    elif args.LossType3 == 'l2_fdivMixture':
        # Measurements from without regurization and training with f-divergence
        ControlonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        TreatmentonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        # set up regularization parameters 
        Control_RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007];    

        if args.DataType == 'PDS_SuperCam':
            pass 
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)];   
            RegStrengths4 = [(0.000070, 0.001200, 0.0007), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)]; 
        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between l2 and f-divergence mixture regularizations')
        
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))
   

            # For treatment 
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                                           
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter, with one run 
                Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))

        # Print oxide names
        print('All oxides', end = ' ')
        for i in range(len(Oxide_Names)):
            if i != len(Oxide_Names) - 1:
                print(Oxide_Names[i], end = ' ')
            else:
                print(Oxide_Names[i])

        # Matched-Pair TTest
        print('Left and right side t-test')
        for j in range(len(Control_RegStrengths)):
            print('group %d'%j)

            left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'greater')
            
            # Print out paired-testing results 
            if left_pvalue <= 0.1:
                print('Better&', end='')
            elif right_pvalue <= 0.1:
                print('Worse&', end='')
            else:
                print('-&', end='')

            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

            for i in range(len(Oxide_Names)):
                left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, i], TreatmentonLevels[j][:, i], alternative = 'less')
                right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, i], TreatmentonLevels[j][:, i], alternative = 'greater')
                # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
                if i == len(Oxide_Names) - 1:
                    if left_pvalue <= 0.1:
                        print('Better')
                    elif right_pvalue <= 0.1:
                        print('Worse')
                    else:
                        print('-')
                else:
                    if left_pvalue <= 0.1:
                        print('Better&', end='')
                    elif right_pvalue <= 0.1:
                        print('Worse&', end='')
                    else:
                        print('-&', end='')

    elif args.LossType3 == 'l1_fdivMixture':
        # Measurements from without regurization and training with f-divergence
        ControlonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        TreatmentonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        # set up regularization parameters 
        Control_RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007];    

        if args.DataType == 'PDS_SuperCam':
            pass 
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)];   
            RegStrengths4 = [(0.000070, 0.001200, 0.0007), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)]; 
        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between l1 and f-divergence mixture regularizations')
        
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                
                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))
   

            # For treatment 
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                                           
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter, with one run 
                Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))
                
        # Matched-Pair TTest
        print('Left and right side t-test')
        for j in range(len(Control_RegStrengths)):
            print('group %d'%j)
            # debug
            # print(np.mean(ControlonLevels[j][:, :-1], axis = 0))
            left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'greater')

            # Print oxide names
            print('All oxides', end = ' ')
            for i in range(len(Oxide_Names)):
                if i != len(Oxide_Names) - 1:
                    print(Oxide_Names[i], end = ' ')
                else:
                    print(Oxide_Names[i])

            # Print out paired-testing results 
            if left_pvalue <= 0.1:
                print('Better&', end='')
            elif right_pvalue <= 0.1:
                print('Worse&', end='')
            else:
                print('-&', end='')

            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

            for i in range(len(Oxide_Names)):
                left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, i], ControlonLevels[j][:, i], alternative = 'less')
                right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, i], ControlonLevels[j][:, i], alternative = 'greater')
                # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
                if i == len(Oxide_Names) - 1:
                    if left_pvalue <= 0.1:
                        print('Better')
                    elif right_pvalue <= 0.1:
                        print('Worse')
                    else:
                        print('-')
                else:
                    if left_pvalue <= 0.1:
                        print('Better&', end='')
                    elif right_pvalue <= 0.1:
                        print('Worse&', end='')
                    else:
                        print('-&', end='')

    elif args.LossType3 == 'Dropout_fdivMixture':
        # Measurements from without regurization and training with f-divergence
        ControlonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        TreatmentonLevels = [np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1)), np.zeros((0,len(Oxide_Names) + 1))]; 
        # set up regularization parameters 
        Control_RegStrengths = [0.04, 0.06, 0.08, 0.1];    

        if args.DataType == 'PDS_SuperCam':
            pass
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.04), (0.000300, 0.015000, 0.04), (0.000140, 0.010000, 0.04), (0.005000, 0.030000, 0.04), (0.000160, 0.015000, 0.04)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.06), (0.000300, 0.015000, 0.06), (0.000140, 0.010000, 0.06), (0.005000, 0.030000, 0.06), (0.000160, 0.015000, 0.06)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.08), (0.000300, 0.015000, 0.08), (0.000140, 0.010000, 0.08), (0.005000, 0.030000, 0.08), (0.000160, 0.015000, 0.08)];
            RegStrengths4 = [(0.000070, 0.001200, 0.1), (0.000300, 0.015000, 0.1), (0.000140, 0.010000, 0.1), (0.005000, 0.030000, 0.1), (0.000160, 0.015000, 0.1)];     

        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between dropout and f-divergence mixture regularizations')
        
        for j in range(args.TrialNum):
            # For Control  
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths)); 
            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j+1), "rb" ))    

                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))
   

            # For treatment 
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j+1), "rb" ))
                                           
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter, with one run 
                Treatment_OneRun = CalculateEveryRMSEforOneRun(TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))
                
        # Matched-Pair TTest
        print('Left and right side t-test')
        for j in range(len(Control_RegStrengths)):
            print('group %d'%j)

            left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'less')
            right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, -1], ControlonLevels[j][:, -1], alternative = 'greater')

            # Print oxide names
            print('All oxides', end = ' ')
            for i in range(len(Oxide_Names)):
                if i != len(Oxide_Names) - 1:
                    print(Oxide_Names[i], end = ' ')
                else:
                    print(Oxide_Names[i])

            # Print out paired-testing results 
            if left_pvalue <= 0.1:
                print('Better&', end='')
            elif right_pvalue <= 0.1:
                print('Worse&', end='')
            else:
                print('-&', end='')

            # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))

            for i in range(len(Oxide_Names)):
                left_statistic, left_pvalue= ttest_rel(TreatmentonLevels[j][:, i], ControlonLevels[j][:, i], alternative = 'less')
                right_statistic, right_pvalue= ttest_rel(TreatmentonLevels[j][:, i], ControlonLevels[j][:, i], alternative = 'greater')
                # print('Oxide:%s, Left T-statistic: %.5f,  Left p-value: %.5f, reject: %d, Right T-statistic: %.5f,  Right p-value: %.5f, reject: %d'%(Oxide_Names[i], left_statistic, left_pvalue, left_pvalue<0.1, right_statistic, right_pvalue, right_pvalue<0.1))
                if i == len(Oxide_Names) - 1:
                    if left_pvalue <= 0.1:
                        print('Better')
                    elif right_pvalue <= 0.1:
                        print('Worse')
                    else:
                        print('-')
                else:
                    if left_pvalue <= 0.1:
                        print('Better&', end='')
                    elif right_pvalue <= 0.1:
                        print('Worse&', end='')
                    else:
                        print('-&', end='')


"""
Print test error for each method that include the hyper-parameter selection
"""         
def MeasureTE_EachMethod(args, ResultDir):
    # set directory parameters
    cwd = os.getcwd() 

    if args.DataType == 'PDS_ChemCam':
        DataDir = cwd + '/%s/Data/'%(args.DataType)
        Oxide_Names =  ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    elif args.DataType == 'PDS_SuperCam':
        pass

    # Load true labels 
    TrueLabel_Dic = {}
    if args.DataType == 'PDS_ChemCam':
        Labels = pickle.load(open(DataDir + 'ChemCam_Mars_labels.pickle', "rb" ))
        SampleName = Labels.keys()
        for name in SampleName:
            TrueLabel_Dic[name] = Labels[name]
    elif args.DataType == 'PDS_SuperCam':
        pass        

    # Define prediction dictionary
    if args.LossType3 == 'NoReg':
        RegStrengths = [0]; 
        BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
        print('RMSE for the CNN without regularization')
        for j in range(args.TrialNum):
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(RegStrengths)); 
            for i in range(len(RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter
            BestTePredManyRuns.append(TePred_A_Run[BestIdx])
            BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])

        CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)       
    elif args.LossType3 == 'ParamSelect_L2_fdiv': 
        if args.DataType == 'PDS_SuperCam':
            pass 
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)];  
            RegStrengths4 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)];   
        RegStrengthsList = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        for RegStrengths in RegStrengthsList:
            print('RMSE for the CNN with l2 strength %.6f + f-divergence regularization'%RegStrengths[0][2])
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for i in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    
                    ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
            
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                # debug
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred_A_Run[BestIdx])
                BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])
            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)
    elif args.LossType3 == 'L1':
        RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007]   
        for l2strength in RegStrengths:
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            print('Printing Test error for the CNN with l1 regularization%.6f'%l2strength)
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
        
                ValPred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred)
                BestTeSampleNameManyRuns.append(TeSampleName)

            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names) 
    elif args.LossType3 == 'ParamSelect_L1_fdiv':
        # F-divergence hyper-parameters acquired by looking at the validation errors 
        if args.DataType == 'PDS_SuperCam':
            pass 
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)];
            RegStrengths4 = [(0.000070, 0.001200, 0.0007), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)];    
        RegStrengthsList = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        for RegStrengths in RegStrengthsList:
            print('RMSE for the CNN with l1 strength %.6f + f-divergence regularization'%RegStrengths[0][2])
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for i in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    
                    ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
            
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                # debug
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred_A_Run[BestIdx])
                BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])
            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)
    elif args.LossType3 == 'ParamSelect_Dropout_fdiv': 
        if args.DataType == 'PDS_SuperCam':
            pass 
        elif args.DataType == 'PDS_ChemCam':    
            RegStrengths1 = [(0.000070, 0.001200, 0.04), (0.000300, 0.015000, 0.04), (0.000140, 0.010000, 0.04), (0.005000, 0.030000, 0.04), (0.000160, 0.015000, 0.04)];       
            RegStrengths2 = [(0.000070, 0.001200, 0.06), (0.000300, 0.015000, 0.06), (0.000140, 0.010000, 0.06), (0.005000, 0.030000, 0.06), (0.000160, 0.015000, 0.06)];    
            RegStrengths3 = [(0.000070, 0.001200, 0.08), (0.000300, 0.015000, 0.08), (0.000140, 0.010000, 0.08), (0.005000, 0.030000, 0.08), (0.000160, 0.015000, 0.08)];     
            RegStrengths4 = [(0.000070, 0.001200, 0.1),  (0.000300, 0.015000, 0.1), (0.000140, 0.010000, 0.1), (0.005000, 0.030000, 0.1), (0.000160, 0.015000, 0.1)]; 
        RegStrengthsList = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        for RegStrengths in RegStrengthsList:
            print('RMSE for the CNN with Dropout strength %.6f + f-divergence regularization'%RegStrengths[0][2])
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
                for i in range(len(RegStrengths)):
                    ValPred = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    ValSampleName = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))

                    TePred = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    TeSampleName = pickle.load(open(ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], RegStrengths[i][2], j+1), "rb" ))
                    
                    ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
            
                    # Save test results 
                    TePred_A_Run.append(TePred);
                    TeSampleName_A_Run.append(TeSampleName)
                # debug
                BestIdx = np.argmin(ValidationError_A_Run)
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred_A_Run[BestIdx])
                BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])

            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)
    elif args.LossType3 == 'L2':
        RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007]   
        for l2strength in RegStrengths:
            print('RMSE for the CNN with the l2 regularization strength%.6f'%l2strength)
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths));         

                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, l2strength, j+1), "rb" ))
    
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred)
                BestTeSampleNameManyRuns.append(TeSampleName)

            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)  
    elif args.LossType3 == 'Dropout':
        RegStrengths = [0.04, 0.06, 0.08, 0.1]; 
        for DropoutRate in RegStrengths:
            print('RMSE for the CNN with the dropout rate%.6f'%DropoutRate)
            BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
            for j in range(args.TrialNum):
                TePred_A_Run = []; TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths)); 
             
                ValPred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, DropoutRate, j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, DropoutRate, j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, DropoutRate, j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, DropoutRate, j+1), "rb" ))
                
                # Test results for the method with the best hyper-parameter
                BestTePredManyRuns.append(TePred)
                BestTeSampleNameManyRuns.append(TeSampleName)

            CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)              
    elif args.LossType3 == 'ParamSelect_L2':
        RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001];
        BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
        print('RMSE for the CNN with l2 regularization')
        for j in range(args.TrialNum):
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(RegStrengths)); 
            for i in range(len(RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter
            BestTePredManyRuns.append(TePred_A_Run[BestIdx])
            BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])

        CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)      
    elif args.LossType3 == 'ParamSelect_L1':
        RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001];
        BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
        print('RMSE for the CNN with l1 regularization')
        for j in range(args.TrialNum):
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(RegStrengths)); 
            for i in range(len(RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter
            BestTePredManyRuns.append(TePred_A_Run[BestIdx])
            BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])

        CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)  
    elif args.LossType3 == 'ParamSelect_Dropout':
        RegStrengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]; 
        BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
        print('RMSE for the CNN with dropout rate')
        for j in range(args.TrialNum):
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(RegStrengths)); 
            for i in range(len(RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.lr, RegStrengths[i], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter
            BestTePredManyRuns.append(TePred_A_Run[BestIdx])
            BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])
        CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names)   
    elif args.LossType3 == 'ParamSelect_fdiv':
        RegStrengths = [(0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014), 
                        (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015), 
                        (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)]        

        BestTePredManyRuns = []; BestTeSampleNameManyRuns = []
        print('RMSE for the CNN with f-divergence regularization')
        for j in range(args.TrialNum):
            TePred_A_Run = []; TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(RegStrengths)); 
            for i in range(len(RegStrengths)):
                ValPred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], j+1), "rb" ))
                ValSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], j+1), "rb" ))

                TePred = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], j+1), "rb" ))
                TeSampleName = pickle.load(open(ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.epochs, args.batch_size, args.Lambda, args.lr, RegStrengths[i][0], RegStrengths[i][1], j+1), "rb" ))
                
                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)
        
                # Save test results 
                TePred_A_Run.append(TePred);
                TeSampleName_A_Run.append(TeSampleName)
            BestIdx = np.argmin(ValidationError_A_Run)
            # Test results for the method with the best hyper-parameter
            BestTePredManyRuns.append(TePred_A_Run[BestIdx])
            BestTeSampleNameManyRuns.append(TeSampleName_A_Run[BestIdx])

        CalculateEveryRMSEforManyRuns(BestTePredManyRuns, BestTeSampleNameManyRuns, TrueLabel_Dic, Oxide_Names) 


   

  