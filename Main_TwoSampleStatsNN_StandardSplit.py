from __future__ import print_function
import os
import pickle 
import numpy as np
import torch
import torch.optim as optim
import random
from DataUtils import ExtractOneHoldStandardSplit
from Models import ChemCam_CNN, MyAIDataset, TwoSampletrain, test
from Options import *
from sklearn import preprocessing

def main():

    # Training settings
    args = parser.parse_args()
    
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

    # Set up directory 
    cwd = os.getcwd()    

    if args.LossType2 == 'fdiv':
        ResultDir = cwd + '/%s/Results/fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.delta)
    elif args.LossType2 == 'L2_fdiv':
        ResultDir = cwd + '/%s/Results/L2_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l2strength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.delta, args.l2delta)
    elif args.LossType2 == 'L1_fdiv':
        ResultDir = cwd + '/%s/Results/L1_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_l1strength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.delta, args.l1delta)    
    elif args.LossType2 == 'Dropout_fdiv':
        ResultDir = cwd + '/%s/Results/Dropout_fdiv/Epochs%dBatch%dLambda%.3fLR%.3fWeight%.6fDelta%.6f_dropoutstrength%.6f/'%(args.DataType, args.epochs, args.batch_size, args.Lambda, args.lr, args.weight, args.delta, args.Dropout_prob)

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
    SampleSplitDir = cwd + '/%s/SampleSplits/'%(args.DataType)

    with open(SampleSplitDir + 'Seed%d_TrSampleName.pickle'%args.seed, 'rb')  as file:
        TrSampleName = pickle.load(file)      
    with open(SampleSplitDir + 'Seed%d_ValSampleName.pickle'%args.seed, 'rb')  as file:
        ValSampleName = pickle.load(file)    
    with open(SampleSplitDir + 'Seed%d_TeSampleName.pickle'%args.seed, 'rb')  as file:
        TeSampleName = pickle.load(file) 


    # Extract train, validation and test sets
    TrFeat, ValFeat, TeFeat, \
    TrLabel, ValLabel, TeLabel, \
    ValSampleName, TeSampleName, ValShotCount, TeShotCount = ExtractOneHoldStandardSplit(args, TrSampleName, ValSampleName, TeSampleName)
    
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
    if args.DataType == 'PDS_ChemCam':
        if args.LossType2 == 'Dropout_fdiv':
            predmodel = ChemCam_CNN(args.Dropout_prob).to(device) # one-D CNN
        else:
            predmodel = ChemCam_CNN().to(device) # one-D CNN
    elif args.DataType == 'PDS_SuperCam':
        pass       

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
    predmodel.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed))
    TePred, TeErr1, TeFdiv = test(args, predmodel, device, te_loader, scaler)

    # print results
    if args.LossType2 == 'fdiv':
        print('DataType:%s, Loss:%s, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.weight, args.delta, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L2_fdiv':
        print('DataType:%s, Loss:%s, l2 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l2delta, args.weight, args.delta, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'L1_fdiv':  
        print('DataType:%s, Loss:%s, l1 strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.l1delta, args.weight, args.delta, args.seed, MinErr1, TeErr1))
    elif args.LossType2 == 'Dropout_fdiv':
        print('DataType:%s, Loss:%s, dropout strength:%.5f, w:%.5f, gamma:%.5f, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.LossType2, args.Dropout_prob, args.weight, args.delta, args.seed, MinErr1, TeErr1))

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

if __name__ == '__main__':
    main()