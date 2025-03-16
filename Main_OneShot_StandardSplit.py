from __future__ import print_function
import os
import pickle 
import numpy as np
import torch
import torch.optim as optim
import random
from sklearn import preprocessing
from DataUtils import ExtractOneHoldStandardSplit
from Models import ChemCam_CNN, MyAIDataset, test, train
from Options import *
      
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
    
    # set up directories
    cwd = os.getcwd() 
   
    if args.LossType =='Dropout':
        ResultDir = cwd + '/%s/Results/Dropout/Epochs%dBatch%dLR%.3fDropout_prob%.6f/'%(args.DataType, args.epochs, args.batch_size, args.lr, args.Dropout_prob)
    elif args.LossType == 'L1':
        ResultDir = cwd + '/%s/Results/L1/Epochs%dBatch%dLR%.3fL1Delta%.6f/'%(args.DataType, args.epochs, args.batch_size, args.lr, args.delta)
    elif args.LossType == 'L2':
        ResultDir = cwd + '/%s/Results/L2/Epochs%dBatch%dLR%.3fL2Delta%.6f/'%(args.DataType,  args.epochs, args.batch_size, args.lr, args.delta)           

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
        if args.Dropout_prob > 0:
            model = ChemCam_CNN(args.Dropout_prob).to(device)
        else:
            model = ChemCam_CNN().to(device) # one-D CNN              
    elif args.DataType == 'PDS_SuperCam':
        pass       

    # Select an optimizer
    if args.LossType == 'L2':
        optimizer = optim.Adadelta(model.parameters(), weight_decay = args.delta, lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(),  lr=args.lr)

    # Set variable to save training details
    MinErr1 = np.inf; MinErr2= np.inf
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
    model.load_state_dict(torch.load(ModelSaveDir + 'MSEnetwork_StandardSplit%d.pt'%args.seed))
    TePred, TeErr1, Tefdiv = test(args, model, device, te_loader, scaler)
    
    # print results
    if args.LossType =='Dropout':
        print('DataType:%s, Dropout prob:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.Dropout_prob, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L1':
        print('DataType:%s, L1 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.delta, args.LossType, args.seed, MinErr1, TeErr1))
    elif args.LossType =='L2':
        print('DataType:%s, L2 strength:%.5f, Loss:%s, Fold:%d, Val MSE: %.5f, Test MSE: %.5f'%(args.DataType, args.delta, args.LossType, args.seed, MinErr1, TeErr1))

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