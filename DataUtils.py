import os
import numpy as np 
import pandas as pd
import pickle 
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
  
# Read ChemCam or SuperCam LIBS data for stanard split
def LoadLIBSDataforNetworkTraining_StandardSplit(args):
    cwd = os.getcwd() 
    # set up directory
    if args.DataType == 'PDS_ChemCam':
        DataDir = cwd + '/%s/Data/'%(args.DataType)
        # Split data to training and validation
        ChemCam_Mars_labels = pickle.load(open(DataDir + 'ChemCam_Mars_labels.pickle', "rb" ))
        SampleName = list(ChemCam_Mars_labels.keys())
        SampleName2 = []
        for sn in SampleName:
            if np.sum(np.isnan(ChemCam_Mars_labels[sn])) == 0: # remove the sample which has missing values in the label 
                SampleName2.append(sn)
    elif args.DataType == 'PDS_SuperCam':
        DataDir = cwd + '/../../Data/PDS_SuperCam/'
        with open(DataDir + 'Calibrated_LIBS_EightOxide_Labels', 'rb') as handle:
            LIBS_Labels = pickle.load(handle)
        SampleName2 = list(LIBS_Labels.keys())

    # K-fold split 
    TrSize = int(len(SampleName2) * 0.8); ValSize = int(len(SampleName2) * 0.1); 
    TrSampleName = np.random.choice(SampleName2, size = TrSize, replace = False)
    ValSampleName = np.random.choice(list(set(SampleName2) - set(TrSampleName)), size=ValSize, replace = False)
    TeSampleName = list(set(SampleName2) - set(TrSampleName) - set(ValSampleName))

    return TrSampleName, ValSampleName, TeSampleName


def ExtractOneHoldStandardSplit(args, TrSampleName, TeSampleName,ValSampleName ):
    cwd = os.getcwd() 
    # set up directory
    if args.DataType == 'PDS_ChemCam':
        DataDir = cwd + '/%s/Data/'%(args.DataType)
        FeatLen = 6144 - 50; LabLen = 9; 
        
        # Split data to training and validation
        ChemCam_Mars_predictors = pickle.load(open(DataDir + 'ChemCam_Mars_predictors.pickle', "rb" ))
        ChemCam_Mars_labels = pickle.load(open(DataDir + 'ChemCam_Mars_labels.pickle', "rb" ))
   
        # Get training, validation and test sets
        TrFeat = np.zeros((0, FeatLen)); TrLabel = np.zeros((0, LabLen))
        ValFeat = np.zeros((0, FeatLen)); ValLabel = np.zeros((0, LabLen))
        TeFeat = np.zeros((0, FeatLen)); TeLabel = np.zeros((0, LabLen))

        TeShotCount = [];  ValShotCount = []; # number of spectra for a sample 

        # Construct training data for ML training
        for sn in TrSampleName:
            # check if there is missing data
            if -1 in ChemCam_Mars_labels[sn]:
                continue
            TrFeat = np.vstack((TrFeat, ChemCam_Mars_predictors[sn].reshape((1, -1))[:, 50:]))
            TrLabel = np.vstack((TrLabel, ChemCam_Mars_labels[sn]))

        # Construct validation data for ML training
        ValSampleName2 = []
        for sn in ValSampleName:
            # check if there is missing data
            if -1 in ChemCam_Mars_labels[sn]:
                continue
            ValSampleName2.append(sn)
            ValFeat = np.vstack((ValFeat, ChemCam_Mars_predictors[sn].reshape((1, -1))[:, 50:]))
            ValLabel = np.vstack((ValLabel, ChemCam_Mars_labels[sn]))
            ValShotCount.append(len(ChemCam_Mars_predictors[sn]))
        ValSampleName = ValSampleName2

        # Construct test data for ML training 
        TeSampleName2 = []
        for sn in TeSampleName:
            if -1 in ChemCam_Mars_labels[sn]:
                continue
            TeSampleName2.append(sn)

            TeFeat = np.vstack((TeFeat, ChemCam_Mars_predictors[sn].reshape((1, -1))[:, 50:]))
            TeLabel = np.vstack((TeLabel, ChemCam_Mars_labels[sn]))                    
            TeShotCount.append(len(ChemCam_Mars_predictors[sn]))
        TeSampleName = TeSampleName2

    elif args.DataType == 'PDS_SuperCam':
        pass
    return TrFeat, ValFeat, TeFeat, TrLabel, ValLabel, TeLabel, ValSampleName, TeSampleName, ValShotCount, TeShotCount

def Clean_Name(name):
    lowercased = name.lower()
    clean_name = lowercased.replace('-', '').replace(' ', '')
    return clean_name

def list_csv_files(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return [node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.csv')]

def read_csv_files(urls):
    Spectrum = np.zeros(6144); Count=0
    for url in urls:
        df = pd.read_csv('https://pds-geosciences.wustl.edu/' + url, header = None)
        Spectrum+=np.mean(df.iloc[15:6159, 11: 51].astype(float), axis = 1)
        Count+=1
    SpectralMean = Spectrum / Count 
    return SpectralMean

class DownLoadDatasets:
    def __init__(self, args):
        cwd = os.getcwd() 
        self.DataDir = cwd + '/%s/Data/'%(args.DataType)
        if not os.path.exists(self.DataDir + 'ChemCam_Mars_labels.pickle'):
            # os.makedirs(DataDir)
            if args.DataType == 'PDS_ChemCam':
                label_url = "https://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx/calib/ccam_calibration_compositions.csv"
                Feat_url1 = 'https://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx/calib/calib_2021/1600mm/'
                Feat_url2 = 'https://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx/calib/calib_2015/1600mm/'
            # Load the label CSV file directly into a pandas DataFrame
            self.df_Labels = pd.read_csv(label_url)
            
            # Label_Dictionary 
            self.Label_Row_Dic = {}

            NonNaN_Row_Target = self.df_Labels['Target'].notna()
            NonNaN_Row_SpectrumName = self.df_Labels['Spectrum Name'].notna()
            NonNaN_Row_SampleName = self.df_Labels['Sample Name'].notna()
        
            for i in range(len(self.df_Labels)):
                if NonNaN_Row_Target[i]:
                    Target_name = Clean_Name(self.df_Labels.loc[i, 'Target'])
                    self.Label_Row_Dic[Target_name] = i
                if NonNaN_Row_SpectrumName[i]:
                    Spectrum_name = Clean_Name(self.df_Labels.loc[i, 'Spectrum Name'])
                    self.Label_Row_Dic[Spectrum_name] = i
                if NonNaN_Row_SampleName[i]:
                    Sample_name = Clean_Name(self.df_Labels.loc[i, 'Sample Name'])
                    self.Label_Row_Dic[Sample_name] = i
            Label_SampleName = self.df_Labels.loc[NonNaN_Row_Target, 'Target'].to_list()
            Label_SampleName+=self.df_Labels.loc[NonNaN_Row_SpectrumName, 'Spectrum Name'].to_list()
            Label_SampleName+=self.df_Labels.loc[NonNaN_Row_SampleName, 'Sample Name'].to_list()
            self.Clean_Label_SampleName = []
            # clean sample names 
            for s in Label_SampleName:
                self.Clean_Label_SampleName.append(Clean_Name(s))
            # load the name of files relevant to feature
            response1 = requests.get(Feat_url1); response2 = requests.get(Feat_url2);
            # Parse the HTML content
            soup1 = BeautifulSoup(response1.text, 'html.parser')
            soup2 = BeautifulSoup(response2.text, 'html.parser')
            # Find all 'a' tags, as directories usually list files as links
            file_names1 = [link.get('href') for link in soup1.find_all('a')]
            file_names2 = [link.get('href') for link in soup2.find_all('a')]
            file_names = file_names1 + file_names2
            self.Clean_Feat_SampleName = {}
            for s in file_names:
                self.Clean_Feat_SampleName[Clean_Name(s.split('/')[-2])] = s

            self.Feat_Dic = {}; self.Label_Dic = {}; 
            self.count = 1 # indicate the number of samples finished 
    def process_sample(self, s):
        if s in self.Clean_Label_SampleName:
            # get url for the feature directory
            url = self.Clean_Feat_SampleName[s]
            # List CSV files
            csv_urls = list_csv_files('https://pds-geosciences.wustl.edu' + url)
            # Read spectral mean from CSV files
            SpectralMean = read_csv_files(csv_urls)
            # Assign data to dictionaries
            self.Feat_Dic[s] = SpectralMean
            self.Label_Dic[s] = self.df_Labels.iloc[self.Label_Row_Dic[s], 3:12].astype(float)
            # This part might need to be handled differently depending on whether you need to keep track of the count globally or not
            return f'{s} finished'
        return None
    # Using ThreadPoolExecutor to parallelize the process
    def parallel_process_samples(self):
        # Create a ThreadPoolExecutor 
        Count = 1
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks to the executor
            futures = {executor.submit(self.process_sample, s): s for s in self.Clean_Feat_SampleName.keys()}
            # Process as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    print(Count, result);Count+=1
    def Parallel_AcquireData(self):    
        if not os.path.exists(self.DataDir + 'ChemCam_Mars_predictors.pickle'):
            self.parallel_process_samples()
            # Save feature and label data 
            if not os.path.exists(self.DataDir):
                    os.makedirs(self.DataDir)
                    with open(self.DataDir + 'ChemCam_Mars_predictors.pickle', 'wb') as file:
                        pickle.dump(self.Feat_Dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(self.DataDir + 'ChemCam_Mars_labels.pickle', 'wb') as file:
                        pickle.dump(self.Label_Dic, file, protocol=pickle.HIGHEST_PROTOCOL)
            print('Data download is complete.')
        else:
            print('Data is already downloaded.')

    

