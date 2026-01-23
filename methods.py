from __future__ import print_function
from sklearn import preprocessing
from BaselineModels import Baseline
from DataUtils import ExtractOneHoldStandardSplit
import argparse
from Models import CNN, MyAIDataset, test, train, TwoSampletrain

import os
import pickle 
import random
import numpy as np 
import torch
import torch.optim as optim




def PLS(args: argparse.Namespace, saving_dir: str) -> None:
    # Set up the environment
    ResultDir = saving_dir + '/%s/Results/PLS/EmbeddingDim%d/'%(args.DataType, args.EmbeddingDim)
    SampleSplitDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)

    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)

    # Construct variables and get training, validation and testing splits of the original data
    # TrSampleName, ValSampleName, TeSampleName = LoadLIBSDataforNetworkTraining_StandardSplit(args) 

    # Load sample splits. 
    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file)     
    # Construct variables and set varibles
    ValPredictLabelDic = {}; TePredictLabelDic = {}


    # set random seed 
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, ValShotCount, TeShotCount = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName, saving_dir)


    # Predictor normorlization
    TrFeat = preprocessing.normalize(TrFeat)
    ValFeat = preprocessing.normalize(ValFeat)
    TeFeat = preprocessing.normalize(TeFeat)
    
    # label scalling
    TrLabel = TrLabel/100; TeLabel = TeLabel/100; ValLabel = ValLabel/100
    
    # oxide weight normalization 
    scaler = preprocessing.StandardScaler().fit(TrLabel)
    TrLabel = scaler.transform(TrLabel)

    # PLS
    Model = Baseline(args, TrFeat, TrLabel, ValFeat, ValLabel, scaler)
    Model.fit()

    # Validation results 
    ValPred, ValErr1,  ValErr2 = Model.predict(); 

    # Test results 
    Model.ValFeat = TeFeat; Model.ValLabel = TeLabel 
    TePred, TeErr1,  TeErr2 = Model.predict(); 

    # Coefficients 
    Coefficients =  Model.Model.coef_
    print('DataType:%s, Fold: %d , Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.seed, ValErr1, TeErr1))


    # validation results saving
    for j, sn in enumerate(ValSampleName):
        ValPredictLabelDic[sn] = ValPred[j]
    # Test results saving
    for j, sn in enumerate(TeSampleName):
        TePredictLabelDic[sn] = TePred[j]


    # Save result
    np.save(ResultDir + 'Seed%d_Coefficients_StandardSplit'%args.seed,Coefficients)

    with open(ResultDir + 'Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TePredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ResultDir + 'Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValPredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(ResultDir + 'Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ResultDir + 'Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL) 

def standard_regularization(args: argparse.Namespace, saving_dir: str) -> None:    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs1 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        cuda_kwargs2 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs1)
        test_kwargs.update(cuda_kwargs2)
    

    if args.LossType =='Dropout':
        ResultDir = saving_dir + '/%s/Results/Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/'%(args.DataType, args.epochs, args.batch_size, args.lr, args.Dropout_prob)
    elif args.LossType == 'L1':
        ResultDir = saving_dir + '/%s/Results/L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/'%(args.DataType, args.epochs, args.batch_size, args.lr, args.l1delta)
    elif args.LossType == 'L2':
        ResultDir = saving_dir + '/%s/Results/L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/'%(args.DataType,  args.epochs, args.batch_size, args.lr, args.l2delta)           

    # Set the saving directories
    ModelSaveDir = ResultDir + 'Models/'
    StatsSaveDir = ResultDir + 'Stats/'
    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)	     
    if not os.path.exists(ModelSaveDir):
        os.makedirs(ModelSaveDir)	
    if not os.path.exists(StatsSaveDir):
        os.makedirs(StatsSaveDir)	

    # Construct variables and acquire the training, validation and testing splits of the original data
    SampleSplitDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)
    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file)  
    
    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, ValShotCount, TeShotCount = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName, saving_dir)

    # Preprocess the data 
    TrFeat = preprocessing.normalize(TrFeat) # remove outlier intensities
    ValFeat = preprocessing.normalize(ValFeat)
    TeFeat = preprocessing.normalize(TeFeat)

    # Label scalling
    TrLabel = TrLabel/100; ValLabel = ValLabel/100; TeLabel = TeLabel/100

    # Construct dictionaries that save the predictions 
    ValPredictLabelDic = {}; TePredictLabelDic = {}

    # normalize training targets
    scaler = preprocessing.StandardScaler().fit(TrLabel)
    TrLabel = scaler.transform(TrLabel) 

    # Set variables
    ValErrEpoches1 = np.ones((args.epochs)); 
    ValErrEpoches2 = np.ones((args.epochs)); 

    # Set pytorch datasets
    TrDataSet = MyAIDataset(TrFeat, TrLabel)
    ValDataSet = MyAIDataset(ValFeat, ValLabel)
    TeDataSet = MyAIDataset(TeFeat, TeLabel)

    # Convert data to pytorch datasets 
    train_loader = torch.utils.data.DataLoader(TrDataSet, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(ValDataSet, **test_kwargs)
    te_loader = torch.utils.data.DataLoader(TeDataSet, **test_kwargs)

    # Build models

    if args.Dropout_prob > 0:
        model = CNN(TrLabel.shape[1], args.Dropout_prob, data_type=args.DataType).to(device)
    else:
        model = CNN(TrLabel.shape[1], data_type=args.DataType).to(device) # one-D CNN              

    # Select an optimizer
    if args.LossType == 'L2':
        optimizer = optim.Adadelta(model.parameters(), weight_decay = args.l2delta, lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(),  lr=args.lr)

    # Set variable to save training details
    MinErr1 = np.inf; 
    for epoch in range(1, args.epochs + 1):
        _ = train(args, model, device, train_loader, optimizer)
        ValPred, ValErr1, Valfdiv = test(args, model, device, val_loader, scaler)
        ValErrEpoches1[int(epoch - 1)] = ValErr1
        ValErrEpoches2[int(epoch - 1)] = Valfdiv
        
        if ValErr1 < MinErr1:
            MinErr1 = ValErr1 
            ValPred = ValPred
            if args.save_model:
                torch.save(model.state_dict(), ModelSaveDir+  "MSEnetwork_StandardSplit%d.pt"%args.seed)

    # Evaluate the model on the test set 
    model.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed, weights_only=True))
    TePred, TeErr1, Tefdiv = test(args, model, device, te_loader, scaler)
    
    # print results
    if args.LossType =='Dropout':
        print('DataType:%s, Dropout prob:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.Dropout_prob, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L1':
        print('DataType:%s, L1 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.l1delta, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L2':
        print('DataType:%s, L2 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.l2delta, args.LossType, args.seed, MinErr1, TeErr1))

    # save results to dictionaries
    for j, sn in enumerate(ValSampleName):
        ValPredictLabelDic[sn] = ValPred[j]
    for j, sn in enumerate(TeSampleName):
        TePredictLabelDic[sn] = TePred[j]

    # Save predictions
    with open(StatsSaveDir + 'Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TePredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValPredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(StatsSaveDir + 'Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL) 


def fdivergence_regularization(args: argparse.Namespace, saving_dir: str) -> None:

    # Training settings
    # args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs1 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        cuda_kwargs2 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs1)
        test_kwargs.update(cuda_kwargs2)


    if args.LossType2 == 'fdiv':
        ResultDir = saving_dir + '/%s/Results/fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma)
    elif args.LossType2 == 'L2_fdiv':
        ResultDir = saving_dir + '/%s/Results/L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_l2strength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.l2delta)
    elif args.LossType2 == 'L1_fdiv':
        ResultDir = saving_dir + '/%s/Results/L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_l1strength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.l1delta)    
    elif args.LossType2 == 'Dropout_fdiv':
        ResultDir = saving_dir + '/%s/Results/Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_dropoutstrength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.Dropout_prob)

    # Set the saving directories 
    ModelSaveDir = ResultDir + 'Models/'
    StatsSaveDir = ResultDir + 'Stats/'

    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)	
        
    if not os.path.exists(ModelSaveDir):
        os.makedirs(ModelSaveDir)	

    if not os.path.exists(StatsSaveDir):
        os.makedirs(StatsSaveDir)	

    # Construct variables and acquire the training, validation and testing splits of the original data 
    SampleSplitDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)

    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file) 


    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, ValShotCount, TeShotCount = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName, saving_dir)
    
    # Preprocess the data 
    TrFeat = preprocessing.normalize(TrFeat) # remove outlier intensities
    ValFeat = preprocessing.normalize(ValFeat)
    TeFeat = preprocessing.normalize(TeFeat)

    # Label scalling
    TrLabel = TrLabel/100; ValLabel = ValLabel/100; TeLabel = TeLabel/100

    # Construct dictionaries that save the predictions 
    ValPredictLabelDic = {}; TePredictLabelDic = {}

    # normalize training targets
    scaler = preprocessing.StandardScaler().fit(TrLabel)
    TrLabel = scaler.transform(TrLabel)

    # Set variables
    ValErrEpoches1 = np.ones((args.epochs)); 
    ValErrEpoches2 = np.ones((args.epochs))

    # Set pytorch datasets
    TrDataSet = MyAIDataset(TrFeat, TrLabel)
    ValDataSet = MyAIDataset(ValFeat, ValLabel)
    TeDataSet = MyAIDataset(TeFeat, TeLabel)

    # Convert data to pytorch datasets 
    train_loader = torch.utils.data.DataLoader(TrDataSet, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(ValDataSet, **test_kwargs)
    te_loader = torch.utils.data.DataLoader(TeDataSet, **test_kwargs)

    # Build models

    if args.LossType2 == 'Dropout_fdiv':
        predmodel = CNN(TrLabel.shape[1], args.Dropout_prob, data_type=args.DataType).to(device) # one-D CNN
    else:
        predmodel = CNN(TrLabel.shape[1], data_type=args.DataType).to(device) # one-D CNN
      

    if args.LossType2 == 'L2_fdiv':
        pred_optimizer = optim.Adadelta(predmodel.parameters(), lr=args.lr, weight_decay=args.l2delta)
    else:
        pred_optimizer = optim.Adadelta(predmodel.parameters(), lr=args.lr)

    # Set variable to save training details
    MinErr1 = np.inf; MinErr2 = np.inf; 
    Fdivtrain2 = TwoSampletrain(args, predmodel, device, train_loader, val_loader, pred_optimizer)
    for epoch in range(1, args.epochs + 1):

        Fdivtrain2.train(epoch)
        Pred, Err1, Fdiv = test(args, predmodel, device, val_loader, scaler) # output prediction for the validation set, validation mse and f-divergence

        ValErrEpoches1[int(epoch - 1)] = Err1
        ValErrEpoches2[int(epoch - 1)] = Fdiv

        if Err1 < MinErr1:
            ValPred = Pred; MinErr1 = Err1
            if args.save_model:
                torch.save(predmodel.state_dict(), ModelSaveDir+  "MSEnetwork_StandardSplit%d.pt"%args.seed)
        if Fdiv < MinErr2: 
            MinErr2 = Fdiv
            if args.save_model:
                torch.save(predmodel.state_dict(), ModelSaveDir+  "fdivnetwork_StandardSplit%d.pt"%args.seed)
    

    # Evaluate model on the test set 
    predmodel.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed, weights_only=True))
    TePred, TeErr1, TeFdiv = test(args, predmodel, device, te_loader, scaler)

    # print results
    if args.LossType2 == 'fdiv':
        print('DataType:%s, Loss:%s, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L2_fdiv':
        print('DataType:%s, Loss:%s, l2 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l2delta, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L1_fdiv':  
        print('DataType:%s, Loss:%s, l1 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l1delta, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'Dropout_fdiv':
        print('DataType:%s, Loss:%s, dropout strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.Dropout_prob, args.weight, args.gamma, args.seed, MinErr1, TeErr1))

    # save results to dictionaries
    for j, sn in enumerate(ValSampleName):
        ValPredictLabelDic[sn] = ValPred[j]
    for j, sn in enumerate(TeSampleName):
        TePredictLabelDic[sn] = TePred[j]

    # Save predictions 
    with open(StatsSaveDir + 'Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TePredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValPredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(StatsSaveDir + 'Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)


def standard_regularization_single_oxide(args: argparse.Namespace, saving_dir: str) -> None:

    # Training settings
    # args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs1 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        cuda_kwargs2 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs1)
        test_kwargs.update(cuda_kwargs2)

    # oxide names for selection
    if args.DataType == 'PDS_ChemCam':
        OXIDE_NAMES = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    elif args.DataType == 'PDS_SuperCam':
        OXIDE_NAMES = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']    

    if args.LossType =='Dropout':
        ResultDir = saving_dir + '/%s/Results/Dropout_single_oxide/%s/Epochs%dBatch%dLR%.3fDropout_prob%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.lr, args.Dropout_prob)
    elif args.LossType == 'L1':
        ResultDir = saving_dir + '/%s/Results/L1_single_oxide/%s/Epochs%dBatch%dLR%.3fL1Delta%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.lr, args.l1delta)
    elif args.LossType == 'L2':
        ResultDir = saving_dir + '/%s/Results/L2_single_oxide/%s/Epochs%dBatch%dLR%.3fL2Delta%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.lr, args.l2delta)           

    # Set the saving directories
    ModelSaveDir = ResultDir + 'Models/'
    StatsSaveDir = ResultDir + 'Stats/'
    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)	     
    if not os.path.exists(ModelSaveDir):
        os.makedirs(ModelSaveDir)	
    if not os.path.exists(StatsSaveDir):
        os.makedirs(StatsSaveDir)	

    # Construct variables and acquire the training, validation and testing splits of the original data
    SampleSplitDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)
    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file)  
    
    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, ValShotCount, TeShotCount = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName, saving_dir)

    # Preprocess the data 
    TrFeat = preprocessing.normalize(TrFeat) # remove outlier intensities
    ValFeat = preprocessing.normalize(ValFeat)
    TeFeat = preprocessing.normalize(TeFeat)

    # Label scalling
    TrLabel = TrLabel/100; ValLabel = ValLabel/100; TeLabel = TeLabel/100
    TrLabel = TrLabel[:, args.oxide_id].reshape((-1,1)); 
    ValLabel = ValLabel[:, args.oxide_id].reshape((-1,1)); 
    TeLabel = TeLabel[:, args.oxide_id].reshape((-1,1)); 


    # Construct dictionaries that save the predictions 
    ValPredictLabelDic = {}; TePredictLabelDic = {}

    # normalize training targets
    scaler = preprocessing.StandardScaler().fit(TrLabel)
    TrLabel = scaler.transform(TrLabel) 

    # Set variables
    ValErrEpoches1 = np.ones((args.epochs)); 
    ValErrEpoches2 = np.ones((args.epochs)); 

    # Set pytorch datasets
    TrDataSet = MyAIDataset(TrFeat, TrLabel)
    ValDataSet = MyAIDataset(ValFeat, ValLabel)
    TeDataSet = MyAIDataset(TeFeat, TeLabel)

    # Convert data to pytorch datasets 
    train_loader = torch.utils.data.DataLoader(TrDataSet, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(ValDataSet, **test_kwargs)
    te_loader = torch.utils.data.DataLoader(TeDataSet, **test_kwargs)

    # Build models

    if args.Dropout_prob > 0:
        model = CNN(TrLabel.shape[1], args.Dropout_prob, data_type=args.DataType).to(device)
    else:
        model = CNN(TrLabel.shape[1], data_type=args.DataType).to(device) # one-D CNN              

    # Select an optimizer
    if args.LossType == 'L2':
        optimizer = optim.Adadelta(model.parameters(), weight_decay = args.l2delta, lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(),  lr=args.lr)

    # Set variable to save training details
    MinErr1 = np.inf; 
    for epoch in range(1, args.epochs + 1):
        _ = train(args, model, device, train_loader, optimizer)
        ValPred, ValErr1, Valfdiv = test(args, model, device, val_loader, scaler)
        ValErrEpoches1[int(epoch - 1)] = ValErr1
        ValErrEpoches2[int(epoch - 1)] = Valfdiv
        
        if ValErr1 < MinErr1:
            MinErr1 = ValErr1 
            ValPred = ValPred
            if args.save_model:
                torch.save(model.state_dict(), ModelSaveDir+  "MSEnetwork_StandardSplit%d.pt"%args.seed)

    # Evaluate the model on the test set 
    model.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed, weights_only=True))
    TePred, TeErr1, Tefdiv = test(args, model, device, te_loader, scaler)
    
    # print results
    if args.LossType =='Dropout':
        print('DataType:%s, Dropout prob:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.Dropout_prob, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L1':
        print('DataType:%s, L1 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.l1delta, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L2':
        print('DataType:%s, L2 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.l2delta, args.LossType, args.seed, MinErr1, TeErr1))

    # save results to dictionaries
    for j, sn in enumerate(ValSampleName):
        ValPredictLabelDic[sn] = ValPred[j]
    for j, sn in enumerate(TeSampleName):
        TePredictLabelDic[sn] = TePred[j]

    # Save predictions
    with open(StatsSaveDir + 'Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TePredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValPredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(StatsSaveDir + 'Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL) 


def fdivergence_regularization_single_oxide(args: argparse.Namespace, saving_dir: str) -> None:

    # Training settings
    # args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # set random seed 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs1 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        cuda_kwargs2 = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs1)
        test_kwargs.update(cuda_kwargs2)

    if args.DataType == 'PDS_ChemCam':
        OXIDE_NAMES = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
    elif args.DataType == 'PDS_SuperCam':
        OXIDE_NAMES = ['SiO2', 'TiO2', 'AL2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']

    if args.LossType2 == 'fdiv':
        ResultDir = saving_dir + '/%s/Results/fdiv_single_oxide/%s/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma)
    elif args.LossType2 == 'L2_fdiv':
        ResultDir = saving_dir + '/%s/Results/L2_fdiv_single_oxide/%s/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_l2strength%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.l2delta)
    elif args.LossType2 == 'L1_fdiv':
        ResultDir = saving_dir + '/%s/Results/L1_fdiv_single_oxide/%s/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_l1strength%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.l1delta)    
    elif args.LossType2 == 'Dropout_fdiv':
        ResultDir = saving_dir + '/%s/Results/Dropout_fdiv_single_oxide/%s/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fgamma%.6f_dropoutstrength%.6f/'%(args.DataType, OXIDE_NAMES[args.oxide_id], args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.gamma, args.Dropout_prob)

    # Set the saving directories 
    ModelSaveDir = ResultDir + 'Models/'
    StatsSaveDir = ResultDir + 'Stats/'

    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)	
        
    if not os.path.exists(ModelSaveDir):
        os.makedirs(ModelSaveDir)	

    if not os.path.exists(StatsSaveDir):
        os.makedirs(StatsSaveDir)	

    # Construct variables and acquire the training, validation and testing splits of the original data 
    SampleSplitDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)

    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file) 


    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, _, _ = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName, saving_dir)
    
    # Preprocess the data 
    TrFeat = preprocessing.normalize(TrFeat) # remove outlier intensities
    ValFeat = preprocessing.normalize(ValFeat)
    TeFeat = preprocessing.normalize(TeFeat)

    # Label scalling
    TrLabel = TrLabel/100; ValLabel = ValLabel/100; TeLabel = TeLabel/100
    TrLabel = TrLabel[:, args.oxide_id].reshape((-1,1)); 
    ValLabel = ValLabel[:, args.oxide_id].reshape((-1,1)); 
    TeLabel = TeLabel[:, args.oxide_id].reshape((-1,1)); 

    # Construct dictionaries that save the predictions 
    ValPredictLabelDic = {}; TePredictLabelDic = {}

    # normalize training targets
    scaler = preprocessing.StandardScaler().fit(TrLabel)
    TrLabel = scaler.transform(TrLabel)

    # Set variables
    ValErrEpoches1 = np.ones((args.epochs)); 
    ValErrEpoches2 = np.ones((args.epochs));

    # Set pytorch datasets
    TrDataSet = MyAIDataset(TrFeat, TrLabel)
    ValDataSet = MyAIDataset(ValFeat, ValLabel)
    TeDataSet = MyAIDataset(TeFeat, TeLabel)

    # Convert data to pytorch datasets 
    train_loader = torch.utils.data.DataLoader(TrDataSet, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(ValDataSet, **test_kwargs)
    te_loader = torch.utils.data.DataLoader(TeDataSet, **test_kwargs)

    # Build models

    if args.LossType2 == 'Dropout_fdiv':
        predmodel = CNN(TrLabel.shape[1], args.Dropout_prob).to(device) # one-D CNN
    else:
        predmodel = CNN(TrLabel.shape[1],).to(device) # one-D CNN
      

    if args.LossType2 == 'L2_fdiv':
        pred_optimizer = optim.Adadelta(predmodel.parameters(), lr=args.lr, weight_decay=args.l2delta)
    else:
        pred_optimizer = optim.Adadelta(predmodel.parameters(), lr=args.lr)

    # Set variable to save training details
    MinErr1 = np.inf; MinErr2 = np.inf; 
    Fdivtrain2 = TwoSampletrain(args, predmodel, device, train_loader, val_loader, pred_optimizer)
    for epoch in range(1, args.epochs + 1):

        Fdivtrain2.train(epoch)
        Pred, Err1, Fdiv = test(args, predmodel, device, val_loader, scaler) # output prediction for the validation set, validation mse and f-divergence

        ValErrEpoches1[int(epoch - 1)] = Err1
        ValErrEpoches2[int(epoch - 1)] = Fdiv

        if Err1 < MinErr1:
            ValPred = Pred; MinErr1 = Err1
            if args.save_model:
                torch.save(predmodel.state_dict(), ModelSaveDir+  "MSEnetwork_StandardSplit%d.pt"%args.seed)
        if Fdiv < MinErr2: 
            MinErr2 = Fdiv
            if args.save_model:
                torch.save(predmodel.state_dict(), ModelSaveDir+  "fdivnetwork_StandardSplit%d.pt"%args.seed)
    

    # Evaluate model on the test set 
    predmodel.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed, weights_only=True))
    TePred, TeErr1, _ = test(args, predmodel, device, te_loader, scaler)

    # print results
    if args.LossType2 == 'fdiv':
        print('DataType:%s, Loss:%s, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L2_fdiv':
        print('DataType:%s, Loss:%s, l2 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l2delta, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L1_fdiv':  
        print('DataType:%s, Loss:%s, l1 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l1delta, args.weight, args.gamma, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'Dropout_fdiv':
        print('DataType:%s, Loss:%s, dropout strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.Dropout_prob, args.weight, args.gamma, args.seed, MinErr1, TeErr1))

    # save results to dictionaries
    for j, sn in enumerate(ValSampleName):
        ValPredictLabelDic[sn] = ValPred[j]
    for j, sn in enumerate(TeSampleName):
        TePredictLabelDic[sn] = TePred[j]

    # Save predictions 
    with open(StatsSaveDir + 'Seed%d_TePredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TePredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValPredictLabelDic_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValPredictLabelDic, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(StatsSaveDir + 'Seed%d_TeSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(StatsSaveDir + 'Seed%d_ValSampleIdx_StandardSplit.pickle'%(args.seed), 'wb') as file:
        pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)


