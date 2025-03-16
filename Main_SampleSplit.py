import os
import pickle 
import random 
import numpy as np 
from DataUtils import LoadLIBSDataforNetworkTraining_StandardSplit, DownLoadDatasets
from Options import *


def main():
    # Set up the environment
    args = parser.parse_args()
    cwd = os.getcwd() 
    ResultDir = cwd + '/%s/SampleSplits/'%(args.DataType)
    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)

    # Download SuperCam or ChemCam datasets if it does not exist 
    Download = DownLoadDatasets(args)
    Download.Parallel_AcquireData()
    # set random seed 
    for i in range(args.TrialNum):
        np.random.seed(i)
        random.seed(i)

        # Construct variables and get training, validation and testing splits of the original data
        TrSampleName, ValSampleName, TeSampleName = LoadLIBSDataforNetworkTraining_StandardSplit(args) 
    
        # Save sample splits
        with open(ResultDir + 'Seed%d_TrSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(TrSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ResultDir + 'Seed%d_TeSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ResultDir + 'Seed%d_ValSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == '__main__':
    main()