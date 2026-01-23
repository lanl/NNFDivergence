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
            Target_Weights = np.vstack((Target_Weights, np.array(TrueLabel_Dic[name])))
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
def MeasurePairedTest(args, Parent_Dir, alpha=0.05):
    """
    Compare RMSEs (paired) between two methods across oxides and print:
      Better/Worse/- plus p-value (min of one-sided p-values) in a LaTeX-table-friendly row.

    Conventions:
      - "Better" means Treatment RMSE < Control RMSE (one-sided 'less')
      - "Worse"  means Treatment RMSE > Control RMSE (one-sided 'greater')
      - p-value reported is min(left_pvalue, right_pvalue)
    """

    # ---------- helpers ----------
    def _print_cmp(left_pvalue, right_pvalue, alpha=0.05, is_last=False):
        pvalue = min(left_pvalue, right_pvalue)
        if left_pvalue <= alpha:
            label = "Better"
        elif right_pvalue <= alpha:
            label = "Worse"
        else:
            label = "-"
        if is_last:
            print(f"{label} {pvalue:.4f}")
        else:
            print(f"{label} {pvalue:.4f}&", end="")

    def _print_oxide_header(oxide_names):
        print("All oxides", end=" ")
        for i, ox in enumerate(oxide_names):
            if i != len(oxide_names) - 1:
                print(ox, end=" ")
            else:
                print(ox)

    # ---------- set directory parameters ----------
    ResultDir = Parent_Dir + '/%s/Results/' % args.DataType
    DataDir   = Parent_Dir + '/%s/Data/' % args.DataType

    if args.DataType == 'PDS_ChemCam':
        Oxide_Names = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    elif args.DataType == 'PDS_SuperCam':
        Oxide_Names = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
    else:
        raise ValueError(f"Unknown args.DataType: {args.DataType}")

    # ---------- Load true labels ----------
    TrueLabel_Dic = {}
    Labels = pickle.load(open(DataDir + f'{args.DataType[4:]}_Mars_labels.pickle', "rb"))
    SampleName = Labels.keys()
    for name in SampleName:
        TrueLabel_Dic[name] = Labels[name]

    # NOTE: These functions are assumed to exist in your codebase:
    #   - CalculateOverallRMSEGivenPredandName(...)
    #   - CalculateEveryRMSEforOneRun(...)

    # ======================================================================
    # 1) NoReg_fdiv
    # ======================================================================
    if args.LossType3 == 'NoReg_fdiv':
        Control   = np.zeros((0, len(Oxide_Names) + 1))
        Treatment = np.zeros((0, len(Oxide_Names) + 1))

        Control_RegStrengths = [0]
        Treatment_RegStrengths = [
            (0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014),
            (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015),
            (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)
        ]

        print('Conducting matched-pair t-test between no regularization and f-divergence regularization')
        for j in range(args.TrialNum):
            # ---- Control (L2 delta=0, your naming uses L2 folder) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths))

            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Control_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # ---- Treatment (fdiv) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths))

            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))

        print('Left and right side t-test')
        _print_oxide_header(Oxide_Names)

        # All-oxides column (stored in last col)
        left_statistic, left_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='less')
        right_statistic, right_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='greater')
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='less')
            right_statistic, right_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    # ======================================================================
    # 2) l2_fdiv
    # ======================================================================
    elif args.LossType3 == 'l2_fdiv':
        Control   = np.zeros((0, len(Oxide_Names) + 1))
        Treatment = np.zeros((0, len(Oxide_Names) + 1))

        Control_RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001]
        Treatment_RegStrengths = [
            (0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014),
            (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015),
            (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)
        ]

        print('Conducting matched-pair t-test between l2 and f-divergence regularizations')
        for j in range(args.TrialNum):
            # ---- Control (L2) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths))

            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Control_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # ---- Treatment (fdiv) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths))

            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))

        print('Left and right side t-test')
        _print_oxide_header(Oxide_Names)

        left_statistic, left_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='less')
        right_statistic, right_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='greater')
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='less')
            right_statistic, right_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    # ======================================================================
    # 3) Dropout_fdiv
    # ======================================================================
    elif args.LossType3 == 'Dropout_fdiv':
        Control   = np.zeros((0, len(Oxide_Names) + 1))
        Treatment = np.zeros((0, len(Oxide_Names) + 1))

        Control_RegStrengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
        Treatment_RegStrengths = [
            (0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014),
            (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015),
            (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)
        ]

        print('Conducting matched-pair t-test between dropout and f-divergence regularizations')
        for j in range(args.TrialNum):
            # ---- Control (Dropout) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths))

            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Control_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # ---- Treatment (fdiv) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths))

            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))

        print('Left and right side t-test')
        _print_oxide_header(Oxide_Names)

        left_statistic, left_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='less')
        right_statistic, right_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='greater')
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='less')
            right_statistic, right_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    # ======================================================================
    # 4) l1_fdiv
    # ======================================================================
    elif args.LossType3 == 'l1_fdiv':
        Control   = np.zeros((0, len(Oxide_Names) + 1))
        Treatment = np.zeros((0, len(Oxide_Names) + 1))

        Control_RegStrengths = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 0.001]
        Treatment_RegStrengths = [
            (0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014),
            (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015),
            (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)
        ]

        print('Conducting matched-pair t-test between l1 and f-divergence regularizations')
        for j in range(args.TrialNum):
            # ---- Control (L1) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_RegStrengths))

            for i in range(len(Control_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Control_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # ---- Treatment (fdiv) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths))

            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))

        print('Left and right side t-test')
        _print_oxide_header(Oxide_Names)

        left_statistic, left_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='less')
        right_statistic, right_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='greater')
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='less')
            right_statistic, right_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))
    # ======================================================================
    # 4) pls_fdiv
    # ======================================================================
    elif args.LossType3 == 'pls_fdiv':
        Control   = np.zeros((0, len(Oxide_Names) + 1))
        Treatment = np.zeros((0, len(Oxide_Names) + 1))

        Control_Dimension = [8, 16, 32, 64, 128]
        Treatment_RegStrengths = [
            (0.00005, 0.001), (0.00007, 0.0012), (0.00009, 0.0014),
            (0.0001, 0.02), (0.00012, 0.005), (0.00014, 0.01), (0.00016, 0.015),
            (0.0003, 0.015), (0.0005, 0.02), (0.0007, 0.025), (0.005, 0.03), (0.01, 0.02)
        ]

        print('Conducting matched-pair t-test between pls and f-divergence regularizations')
        for j in range(args.TrialNum):
            # ---- Control (pls) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Control_Dimension))

            for i in range(len(Control_Dimension)):
                ValPred = pickle.load(open(
                    ResultDir + 'PLS/EmbeddingDim%d/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (Control_Dimension[i], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'PLS/EmbeddingDim%d/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (Control_Dimension[i], j + 1), "rb"))
                TePred = pickle.load(open(
                    ResultDir + 'PLS/EmbeddingDim%d/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (Control_Dimension[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'PLS/EmbeddingDim%d/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (Control_Dimension[i], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Control_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Control = np.vstack((Control, Control_OneRun))

            # ---- Treatment (fdiv) ----
            TePred_A_Run = []
            TeSampleName_A_Run = []
            ValidationError_A_Run = np.zeros(len(Treatment_RegStrengths))

            for i in range(len(Treatment_RegStrengths)):
                ValPred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                ValSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                TePred = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.Lambda, args.lr,
                       Treatment_RegStrengths[i][0], Treatment_RegStrengths[i][1], j + 1), "rb"))

                ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                    ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                TePred_A_Run.append(TePred)
                TeSampleName_A_Run.append(TeSampleName)

            BestIdx = np.argmin(ValidationError_A_Run)
            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
            Treatment = np.vstack((Treatment, Treatment_OneRun))

        print('Left and right side t-test')
        _print_oxide_header(Oxide_Names)

        left_statistic, left_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='less')
        right_statistic, right_pvalue = ttest_rel(Treatment[:, -1], Control[:, -1], alternative='greater')
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='less')
            right_statistic, right_pvalue = ttest_rel(Treatment[:, i], Control[:, i], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))
    # ======================================================================
    # 6) l2_fdivMixture
    # ======================================================================
    elif args.LossType3 == 'l2_fdivMixture':
        ControlonLevels   = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]
        TreatmentonLevels = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]

        Control_RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007]

        if args.DataType == 'PDS_SuperCam':
            RegStrengths1 = [(0.00016, 0.015, 0.0001), (0.005, 0.03, 0.0001), (0.0005, 0.02, 0.0001), (0.00007, 0.0012, 0.0001), (0.0007, 0.025, 0.0001)]
            RegStrengths2 = [(0.00016, 0.015, 0.0003), (0.005, 0.03, 0.0003), (0.0005, 0.02, 0.0003), (0.00007, 0.0012, 0.0003), (0.0007, 0.025, 0.0003)]
            RegStrengths3 = [(0.00016, 0.015, 0.0005), (0.005, 0.03, 0.0005), (0.0005, 0.02, 0.0005), (0.00007, 0.0012, 0.0005), (0.0007, 0.025, 0.0005)]
            RegStrengths4 = [(0.00016, 0.015, 0.0007), (0.005, 0.03, 0.0007), (0.0005, 0.02, 0.0007), (0.00007, 0.0012, 0.0007), (0.0007, 0.025, 0.0007)]
        elif args.DataType == 'PDS_ChemCam':
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)]
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)]
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)]
            RegStrengths4 = [(0.000070, 0.001200, 0.0007), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)]
        else:
            raise ValueError(f"Unknown args.DataType: {args.DataType}")

        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between l2 and f-divergence mixture regularizations')

        for j in range(args.TrialNum):
            # ---- Control levels ----
            for i in range(len(Control_RegStrengths)):
                TePred = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))

            # ---- Treatment levels ----
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []
                TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths))

                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(
                        ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    ValSampleName = pickle.load(open(
                        ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred = pickle.load(open(
                        ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    TeSampleName = pickle.load(open(
                        ResultDir + 'L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred_A_Run.append(TePred)
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(
                        ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                BestIdx = np.argmin(ValidationError_A_Run)
                Treatment_OneRun = CalculateEveryRMSEforOneRun(
                    TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))

        # Print header once
        _print_oxide_header(Oxide_Names)
        print('Left and right side t-test')

        for lvl in range(len(Control_RegStrengths)):
            print(f'group {lvl}')

            # All-oxides summary
            left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='less')
            right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

            # Per-oxide
            for i in range(len(Oxide_Names)):
                # IMPORTANT BUG FIX: compare Treatment vs Control (not Treatment vs itself)
                left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='less')
                right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='greater')
                _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    # ======================================================================
    # 7) l1_fdivMixture
    # ======================================================================
    elif args.LossType3 == 'l1_fdivMixture':
        ControlonLevels   = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]
        TreatmentonLevels = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]

        Control_RegStrengths = [0.0001, 0.0003, 0.0005, 0.0007]

        if args.DataType == 'PDS_SuperCam':
            RegStrengths1 = [(0.00016, 0.015, 0.0001), (0.005, 0.03, 0.0001), (0.0005, 0.02, 0.0001), (0.00007, 0.0012, 0.0001), (0.0007, 0.025, 0.0001)]
            RegStrengths2 = [(0.00016, 0.015, 0.0003), (0.005, 0.03, 0.0003), (0.0005, 0.02, 0.0003), (0.00007, 0.0012, 0.0003), (0.0007, 0.025, 0.0003)]
            RegStrengths3 = [(0.00016, 0.015, 0.0005), (0.005, 0.03, 0.0005), (0.0005, 0.02, 0.0005), (0.00007, 0.0012, 0.0005), (0.0007, 0.025, 0.0005)]
            RegStrengths4 = [(0.00016, 0.015, 0.0007), (0.005, 0.03, 0.0007), (0.0005, 0.02, 0.0007), (0.00007, 0.0012, 0.0007), (0.0007, 0.025, 0.0007)]
        elif args.DataType == 'PDS_ChemCam':
            RegStrengths1 = [(0.000070, 0.001200, 0.0001), (0.000300, 0.015000, 0.0001), (0.000140, 0.010000, 0.0001), (0.005000, 0.030000, 0.0001), (0.000160, 0.015000, 0.0001)]
            RegStrengths2 = [(0.000070, 0.001200, 0.0003), (0.000300, 0.015000, 0.0003), (0.000140, 0.010000, 0.0003), (0.005000, 0.030000, 0.0003), (0.000160, 0.015000, 0.0003)]
            RegStrengths3 = [(0.000070, 0.001200, 0.0005), (0.000300, 0.015000, 0.0005), (0.000140, 0.010000, 0.0005), (0.005000, 0.030000, 0.0005), (0.000160, 0.015000, 0.0005)]
            RegStrengths4 = [(0.000070, 0.001200, 0.0007), (0.000300, 0.015000, 0.0007), (0.000140, 0.010000, 0.0007), (0.005000, 0.030000, 0.0007), (0.000160, 0.015000, 0.0007)]
        else:
            raise ValueError(f"Unknown args.DataType: {args.DataType}")

        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between l1 and f-divergence mixture regularizations')

        for j in range(args.TrialNum):
            # ---- Control levels ----
            for i in range(len(Control_RegStrengths)):
                TePred = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))

            # ---- Treatment levels ----
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []
                TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths))

                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(
                        ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    ValSampleName = pickle.load(open(
                        ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred = pickle.load(open(
                        ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    TeSampleName = pickle.load(open(
                        ResultDir + 'L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred_A_Run.append(TePred)
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(
                        ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                BestIdx = np.argmin(ValidationError_A_Run)
                Treatment_OneRun = CalculateEveryRMSEforOneRun(
                    TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))

        _print_oxide_header(Oxide_Names)
        print('Left and right side t-test')

        for lvl in range(len(Control_RegStrengths)):
            print(f'group {lvl}')

            left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='less')
            right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

            for i in range(len(Oxide_Names)):
                left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='less')
                right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='greater')
                _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    # ======================================================================
    # 8) Dropout_fdivMixture
    # ======================================================================
    elif args.LossType3 == 'Dropout_fdivMixture':
        ControlonLevels   = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]
        TreatmentonLevels = [np.zeros((0, len(Oxide_Names) + 1)) for _ in range(4)]

        Control_RegStrengths = [0.04, 0.06, 0.08, 0.1]

        if args.DataType == 'PDS_SuperCam':
            RegStrengths1 = [(0.00016, 0.015, 0.04), (0.005, 0.03, 0.04), (0.0005, 0.02, 0.04), (0.00007, 0.0012, 0.04), (0.0007, 0.025, 0.04)]
            RegStrengths2 = [(0.00016, 0.015, 0.06), (0.005, 0.03, 0.06), (0.0005, 0.02, 0.06), (0.00007, 0.0012, 0.06), (0.0007, 0.025, 0.06)]
            RegStrengths3 = [(0.00016, 0.015, 0.08), (0.005, 0.03, 0.08), (0.0005, 0.02, 0.08), (0.00007, 0.0012, 0.08), (0.0007, 0.025, 0.08)]
            RegStrengths4 = [(0.00016, 0.015, 0.1),  (0.005, 0.03, 0.1),  (0.0005, 0.02, 0.1),  (0.00007, 0.0012, 0.1),  (0.0007, 0.025, 0.1)]
        elif args.DataType == 'PDS_ChemCam':
            RegStrengths1 = [(0.000070, 0.001200, 0.04), (0.000300, 0.015000, 0.04), (0.000140, 0.010000, 0.04), (0.005000, 0.030000, 0.04), (0.000160, 0.015000, 0.04)]
            RegStrengths2 = [(0.000070, 0.001200, 0.06), (0.000300, 0.015000, 0.06), (0.000140, 0.010000, 0.06), (0.005000, 0.030000, 0.06), (0.000160, 0.015000, 0.06)]
            RegStrengths3 = [(0.000070, 0.001200, 0.08), (0.000300, 0.015000, 0.08), (0.000140, 0.010000, 0.08), (0.005000, 0.030000, 0.08), (0.000160, 0.015000, 0.08)]
            RegStrengths4 = [(0.000070, 0.001200, 0.1),  (0.000300, 0.015000, 0.1),  (0.000140, 0.010000, 0.1),  (0.005000, 0.030000, 0.1),  (0.000160, 0.015000, 0.1)]
        else:
            raise ValueError(f"Unknown args.DataType: {args.DataType}")

        Treatment_RegStrengths = [RegStrengths1, RegStrengths2, RegStrengths3, RegStrengths4]
        print('Conducting matched-pair t-test between dropout and f-divergence mixture regularizations')

        for j in range(args.TrialNum):
            # ---- Control levels (Dropout) ----
            for i in range(len(Control_RegStrengths)):
                TePred = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))
                TeSampleName = pickle.load(open(
                    ResultDir + 'Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                    % (args.epochs, args.batch_size, args.lr, Control_RegStrengths[i], j + 1), "rb"))

                Control_OneRun = CalculateEveryRMSEforOneRun(TePred, TeSampleName, TrueLabel_Dic, Oxide_Names)
                ControlonLevels[i] = np.vstack((ControlonLevels[i], Control_OneRun))

            # ---- Treatment levels ----
            for i in range(len(Treatment_RegStrengths)):
                RegStrengths = Treatment_RegStrengths[i]
                TePred_A_Run = []
                TeSampleName_A_Run = []
                ValidationError_A_Run = np.zeros(len(RegStrengths))

                for u in range(len(RegStrengths)):
                    ValPred = pickle.load(open(
                        ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    ValSampleName = pickle.load(open(
                        ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_ValSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred = pickle.load(open(
                        ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))
                    TeSampleName = pickle.load(open(
                        ResultDir + 'Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/Stats/Seed%d_TeSampleIdx_StandardSplit.pickle'
                        % (args.epochs, args.batch_size, args.Lambda, args.lr,
                           RegStrengths[u][0], RegStrengths[u][1], RegStrengths[u][2], j + 1), "rb"))

                    TePred_A_Run.append(TePred)
                    TeSampleName_A_Run.append(TeSampleName)
                    ValidationError_A_Run[u] = CalculateOverallRMSEGivenPredandName(
                        ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names)

                BestIdx = np.argmin(ValidationError_A_Run)
                Treatment_OneRun = CalculateEveryRMSEforOneRun(
                    TePred_A_Run[BestIdx], TeSampleName_A_Run[BestIdx], TrueLabel_Dic, Oxide_Names)
                TreatmentonLevels[i] = np.vstack((TreatmentonLevels[i], Treatment_OneRun))

        _print_oxide_header(Oxide_Names)
        print('Left and right side t-test')

        for lvl in range(len(Control_RegStrengths)):
            print(f'group {lvl}')

            left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='less')
            right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='greater')
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

            for i in range(len(Oxide_Names)):
                left_statistic, left_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='less')
                right_statistic, right_pvalue = ttest_rel(TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='greater')
                _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))

    elif args.LossType3 == 'Single_Oxide_Dropout_fdivMixture':
        print('Conducting matched-pair t-test between dropout and f-divergence mixture regularizations')
        print('(single-oxide saved predictions; ChemCam; dropout_prob = 0.06)')

        if args.DataType != "PDS_ChemCam":
            raise ValueError("Single-oxide t-test setup is only available for ChemCam.")

        DropoutRate = 0.06  # only available setting for single-oxide artifacts

        # Storage: rows per run, columns = per-oxide RMSE + overall RMSE
        ControlonLevels   = [np.zeros((0, len(Oxide_Names) + 1))]   # only 1 level (0.06)
        TreatmentonLevels = [np.zeros((0, len(Oxide_Names) + 1))]   # only 1 level (0.06)

        # Treatment candidate list (Weight, Delta, Dropout)
        RegStrengths = [
            (0.000070, 0.001200, 0.06),
            (0.000300, 0.015000, 0.06),
            (0.000140, 0.010000, 0.06),
            (0.005000, 0.030000, 0.06),
            (0.000160, 0.015000, 0.06),
        ]

        for j in range(args.TrialNum):

            # ------------------------------------------------------------
            # Control (Dropout only): assemble multi-oxide vectors per sample
            # from single-oxide saved predictions (code2 pattern)
            # ------------------------------------------------------------
            Control_TePred_A_Run = {}
            Control_TeSampleName_A_Run = None

            for idx, oxide_id in enumerate(Oxide_Names):

                TePred = pickle.load(open(
                    ResultDir
                    + "Dropout_single_oxide/%s/"
                    "Epochs%dBatch%dLR%.3fDropout_prob%.6f/"
                    "Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle"
                    % (oxide_id, args.epochs, args.batch_size, args.lr, DropoutRate, j + 1),
                    "rb"
                ))

                TeSampleName = pickle.load(open(
                    ResultDir
                    + "Dropout_single_oxide/%s/"
                    "Epochs%dBatch%dLR%.3fDropout_prob%.6f/"
                    "Stats/Seed%d_TeSampleIdx_StandardSplit.pickle"
                    % (oxide_id, args.epochs, args.batch_size, args.lr, DropoutRate, j + 1),
                    "rb"
                ))

                if Control_TeSampleName_A_Run is None:
                    Control_TeSampleName_A_Run = TeSampleName
                    for s in Control_TeSampleName_A_Run:
                        Control_TePred_A_Run[s] = np.zeros(len(Oxide_Names), dtype=float)
                else:
                    assert TeSampleName == Control_TeSampleName_A_Run, \
                        f"[Control] Sample mismatch for oxide {oxide_id}, seed {j+1}"

                for s in TePred:
                    Control_TePred_A_Run[s][idx] = TePred[s]

            Control_OneRun = CalculateEveryRMSEforOneRun(
                Control_TePred_A_Run, Control_TeSampleName_A_Run, TrueLabel_Dic, Oxide_Names
            )
            ControlonLevels[0] = np.vstack((ControlonLevels[0], Control_OneRun))


            # ------------------------------------------------------------
            # Treatment (Dropout + fdiv): for each oxide, pick best (Weight,Delta)
            # using validation RMSE for that oxide, then assemble vectors (code1 pattern)
            # ------------------------------------------------------------
            Treatment_TePred_A_Run = {}
            Treatment_TeSampleName_A_Run = None

            for idx, oxide_id in enumerate(Oxide_Names):

                ValidationError_A_Run = np.zeros(len(RegStrengths))
                TePred_candidates = []
                TeSampleName_candidates = []

                for i, (w, d, dr) in enumerate(RegStrengths):

                    ValPred = pickle.load(open(
                        ResultDir
                        + "Dropout_fdiv_single_oxide/%s/"
                        "Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/"
                        "Stats/Seed%d_ValPredictLabelDic_StandardSplit.pickle"
                        % (oxide_id, args.epochs, args.batch_size, args.Lambda, args.lr, w, d, dr, j + 1),
                        "rb"
                    ))
                    ValSampleName = pickle.load(open(
                        ResultDir
                        + "Dropout_fdiv_single_oxide/%s/"
                        "Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/"
                        "Stats/Seed%d_ValSampleIdx_StandardSplit.pickle"
                        % (oxide_id, args.epochs, args.batch_size, args.Lambda, args.lr, w, d, dr, j + 1),
                        "rb"
                    ))

                    TePred = pickle.load(open(
                        ResultDir
                        + "Dropout_fdiv_single_oxide/%s/"
                        "Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/"
                        "Stats/Seed%d_TePredictLabelDic_StandardSplit.pickle"
                        % (oxide_id, args.epochs, args.batch_size, args.Lambda, args.lr, w, d, dr, j + 1),
                        "rb"
                    ))
                    TeSampleName = pickle.load(open(
                        ResultDir
                        + "Dropout_fdiv_single_oxide/%s/"
                        "Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/"
                        "Stats/Seed%d_TeSampleIdx_StandardSplit.pickle"
                        % (oxide_id, args.epochs, args.batch_size, args.Lambda, args.lr, w, d, dr, j + 1),
                        "rb"
                    ))

                    # validation RMSE for THIS oxide only
                    ValidationError_A_Run[i] = CalculateOverallRMSEGivenPredandName(
                        ValPred, ValSampleName, TrueLabel_Dic, Oxide_Names, idx
                    )

                    TePred_candidates.append(TePred)
                    TeSampleName_candidates.append(TeSampleName)

                    if Treatment_TeSampleName_A_Run is None:
                        Treatment_TeSampleName_A_Run = TeSampleName

                BestIdx = int(np.argmin(ValidationError_A_Run))
                TePred_best = TePred_candidates[BestIdx]
                TeSampleName_best = TeSampleName_candidates[BestIdx]

                if len(Treatment_TePred_A_Run) == 0:
                    # initialize vectors
                    for s in Treatment_TeSampleName_A_Run:
                        Treatment_TePred_A_Run[s] = np.zeros(len(Oxide_Names), dtype=float)

                assert TeSampleName_best == Treatment_TeSampleName_A_Run, \
                    f"[Treatment] Sample mismatch for oxide {oxide_id}, seed {j+1}"

                for s in TePred_best:
                    Treatment_TePred_A_Run[s][idx] = TePred_best[s]

            Treatment_OneRun = CalculateEveryRMSEforOneRun(
                Treatment_TePred_A_Run, Treatment_TeSampleName_A_Run, TrueLabel_Dic, Oxide_Names
            )
            TreatmentonLevels[0] = np.vstack((TreatmentonLevels[0], Treatment_OneRun))


        # ------------------------------------------------------------
        # Print paired t-tests (same style as your old code, but only 1 level)
        # ------------------------------------------------------------
        _print_oxide_header(Oxide_Names)
        print('Left and right side t-test')

        lvl = 0
        print(f'group {lvl} (dropout_prob={DropoutRate:.2f})')

        left_statistic, left_pvalue = ttest_rel(
            TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='less'
        )
        right_statistic, right_pvalue = ttest_rel(
            TreatmentonLevels[lvl][:, -1], ControlonLevels[lvl][:, -1], alternative='greater'
        )
        _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=False)

        for i in range(len(Oxide_Names)):
            left_statistic, left_pvalue = ttest_rel(
                TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='less'
            )
            right_statistic, right_pvalue = ttest_rel(
                TreatmentonLevels[lvl][:, i], ControlonLevels[lvl][:, i], alternative='greater'
            )
            _print_cmp(left_pvalue, right_pvalue, alpha=alpha, is_last=(i == len(Oxide_Names) - 1))


    else:
        raise ValueError(f"Unsupported args.LossType3: {args.LossType3}")



   

  