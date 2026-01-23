import argparse
import os
import random
import numpy as np 
import pandas as pd
import pickle 
import requests

from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
  
# Read ChemCam or SuperCam LIBS data for stanard split
def LoadLIBSDataforNetworkTraining_StandardSplit(dataType, saving_path):
    # set up directory
   
    DataDir = saving_path + '/%s/Data/'%(dataType)
    # Split data to training and validation
    oxide_labels = pickle.load(open(DataDir + f'{dataType[4:]}_Mars_labels.pickle', "rb" ))
    SampleName = list(oxide_labels.keys())
    SampleName2 = []
    for sn in SampleName:
        if np.sum(np.isnan(oxide_labels[sn])) == 0: # remove the sample which has missing values in the label 
            SampleName2.append(sn)
 
    # K-fold split 
    TrSize = int(len(SampleName2) * 0.8); ValSize = int(len(SampleName2) * 0.1); 
    TrSampleName = np.random.choice(SampleName2, size = TrSize, replace = False)
    ValSampleName = np.random.choice(list(set(SampleName2) - set(TrSampleName)), size=ValSize, replace = False)
    TeSampleName = list(set(SampleName2) - set(TrSampleName) - set(ValSampleName))

    return TrSampleName, ValSampleName, TeSampleName


def ExtractOneHoldStandardSplit(args, TrSampleName, TeSampleName,ValSampleName, saving_path):

    DataDir = saving_path + '/%s/Data/'%(args.DataType)
    if args.DataType == 'PDS_ChemCam':
        FeatLen = 6144 - 50; LabLen = 9; 
    elif args.DataType == 'PDS_SuperCam':
        FeatLen = 7933; LabLen = 8; 
    
    # Split data to training and validation
    predictors = pickle.load(open(DataDir + f'{args.DataType[4:]}_Mars_predictors.pickle', "rb" ))
    labels = pickle.load(open(DataDir + f'{args.DataType[4:]}_Mars_labels.pickle', "rb" ))

    # Get training, validation and test sets
    TrFeat = np.zeros((0, FeatLen)); TrLabel = np.zeros((0, LabLen))
    ValFeat = np.zeros((0, FeatLen)); ValLabel = np.zeros((0, LabLen))
    TeFeat = np.zeros((0, FeatLen)); TeLabel = np.zeros((0, LabLen))

    TeShotCount = [];  ValShotCount = []; # number of spectra for a sample 

    # Construct training data for ML training
    for sn in TrSampleName:
        # check if there is missing data
        if np.sum(np.isnan(labels[sn])):
            continue
        if args.DataType == 'PDS_ChemCam':
            TrFeat = np.vstack((TrFeat, np.array(predictors[sn]).reshape((1, -1))[:, 50:])) # exclude the noisy spectroscopy at the beginning
        else:
            TrFeat = np.vstack((TrFeat, np.array(predictors[sn]).reshape((1, -1))))
        TrLabel = np.vstack((TrLabel, labels[sn]))

    # Construct validation data for ML training
    ValSampleName2 = []
    for sn in ValSampleName:
        # check if there is missing data
        if np.sum(np.isnan(labels[sn])):
            continue
        ValSampleName2.append(sn)
        if args.DataType == 'PDS_ChemCam':
            ValFeat = np.vstack((ValFeat, np.array(predictors[sn]).reshape((1, -1))[:, 50:]))
        else:
            ValFeat = np.vstack((ValFeat, np.array(predictors[sn]).reshape((1, -1))))
        ValLabel = np.vstack((ValLabel, labels[sn]))
        ValShotCount.append(len(predictors[sn]))
    ValSampleName = ValSampleName2

    # Construct test data for ML training 
    TeSampleName2 = []
    for sn in TeSampleName:
        if np.sum(np.isnan(labels[sn])):
            continue
        TeSampleName2.append(sn)
        if args.DataType == 'PDS_ChemCam':
            TeFeat = np.vstack((TeFeat, np.array(predictors[sn]).reshape((1, -1))[:, 50:]))
        else:
            TeFeat = np.vstack((TeFeat, np.array(predictors[sn]).reshape((1, -1))))
        TeLabel = np.vstack((TeLabel, labels[sn]))                    
        TeShotCount.append(len(np.array(predictors[sn])))
    TeSampleName = TeSampleName2


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

class DownLoadChemCam:
    def __init__(self, args, saving_path= None):
        if saving_path == None:
            cwd = os.getcwd() 
            self.DataDir = cwd + '/PDS_ChemCam/Data/'
        else:
            self.DataDir = saving_path + '/PDS_ChemCam/Data/'
    
        if not os.path.exists(self.DataDir + 'ChemCam_Mars_labels.pickle'):
            # os.makedirs(DataDir)
           
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

    

def download_file(url: str, out_path: Path, chunk_size: int = 1 << 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        n = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    n += len(chunk)
                    if total:
                        pct = 100.0 * n / total
                        print(f"\rDownloading: {pct:6.2f}% ({n}/{total} bytes)", end="")
        print("\nDone:", out_path)


def load_twoheader_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV with 2 header rows into MultiIndex columns.
    Handles non-utf8 gracefully by trying common encodings.
    """
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, header=[0, 1], encoding=enc, low_memory=False)
            # Normalize header strings
            df.columns = pd.MultiIndex.from_tuples(
                [(str(a).strip(), str(b).strip()) for a, b in df.columns.to_list()]
            )
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV with tried encodings: {encodings_to_try}. Last error: {last_err}")


def infer_target_key(cols: pd.MultiIndex) -> tuple[str, str]:
    """
    Try to find the target/sample name column under level-0 'meta'.
    Prefers (meta, Target_Name), but falls back to best match.
    """
    # Exact common key first
    if ("meta", "Target_Name") in cols:
        return ("meta", "Target_Name")

    meta_cols = [c for c in cols if c[0].lower() == "meta"]
    if not meta_cols:
        raise ValueError("No 'meta' columns found; cannot locate target/sample name column.")

    # Heuristic: choose a meta column containing "target" and/or "name"
    def score(c):
        s = c[1].lower()
        sc = 0
        if "target" in s: sc += 10
        if "name" in s: sc += 5
        if "sample" in s: sc += 4
        if "rock" in s: sc += 2
        return sc

    meta_cols_sorted = sorted(meta_cols, key=score, reverse=True)
    best = meta_cols_sorted[0]
    if score(best) == 0:
        raise ValueError(
            "Could not infer target column from meta fields. "
            f"Meta columns (sample): {meta_cols[:20]}"
        )
    return best


def build_arrays_from_twoheader(
    df: pd.DataFrame,
    oxides=("SiO2","TiO2","Al2O3","FeOT","MnO","MgO","CaO","Na2O","K2O","P2O5"),
    comp_level0="comp",
    wvl_level0="wvl",
):
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError("Expected df.columns to be a 2-level MultiIndex (from header=[0,1]).")

    target_key = infer_target_key(df.columns)
    print("Using target column:", target_key)

    # Target series
    target = df[target_key].astype(str).str.strip()

    # Oxides columns: (comp, oxide)
    oxide_cols = [(comp_level0, o) for o in oxides if (comp_level0, o) in df.columns]
    missing = [o for o in oxides if (comp_level0, o) not in df.columns]
    if missing:
        raise ValueError(f"Missing oxide columns under '{comp_level0}': {missing}")

    # Wavelength columns: (wvl, "<number>")
    wvl_cols = [c for c in df.columns if c[0] == wvl_level0]
    if not wvl_cols:
        raise ValueError(f"No wavelength columns found with level-0 '{wvl_level0}'.")

    # Parse and sort wavelengths
    def parse_wvl(c):
        try:
            return float(c[1])
        except Exception:
            return np.nan

    wvl_with_vals = [(c, parse_wvl(c)) for c in wvl_cols]
    wvl_with_vals = [(c, v) for c, v in wvl_with_vals if np.isfinite(v)]
    if not wvl_with_vals:
        raise ValueError("Found 'wvl' columns but could not parse numeric wavelengths from level-1.")

    wvl_with_vals.sort(key=lambda t: t[1])
    wvl_cols_sorted = [c for c, _ in wvl_with_vals]
    wavelengths = np.array([v for _, v in wvl_with_vals], dtype=float)

    # Working DF
    work = pd.DataFrame({target_key: target})
    work = pd.concat([work, df[oxide_cols], df[wvl_cols_sorted]], axis=1)

    # Coerce numeric
    for c in oxide_cols + wvl_cols_sorted:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Drop missing target
    work = work.dropna(subset=[target_key])

    # Aggregate: average spectra per target
    X_df = work.groupby(target_key, sort=True)[wvl_cols_sorted].first()

    # Oxides aggregation

    Y_df = work.groupby(target_key, sort=True)[oxide_cols].first()
 

    # Align
    X_df, Y_df = X_df.align(Y_df, join="inner", axis=0)

    targets = [str(t) for t in X_df.index.to_list()]
    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    oxide_names = [c[1] for c in Y_df.columns.to_list()]

    return targets, wavelengths, X, Y, oxide_names

def DownLoadSuperCam(saving_path):
    CSV_URL = (
        "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_supercam/"
        "calibration_supercam/libs_spectral_library_reference.csv"
    )

    OUT_DIR = Path(saving_path + "/PDS_SuperCam/Data")
    OUT_CSV = OUT_DIR / "libs_spectral_library_reference.csv"

    if not OUT_CSV.exists():
        download_file(CSV_URL, OUT_CSV)
    else:
        print("CSV already exists:", OUT_CSV)

    df = load_twoheader_csv(OUT_CSV)
    targets, wavelengths, X, Y, oxide_names = build_arrays_from_twoheader(
        df,
    )

    wvl_index = pd.Index([float(w) for w in wavelengths], name="wavelength")
    ox_index = pd.Index(list(oxide_names[:4] + oxide_names[5:9]), name="oxide") # exlude MnO and P205

    spec_dict = {}
    oxide_dict = {}

    for i, t in enumerate(targets):
        t = str(t)

        # Spectrum series: wavelengths -> intensities
        spec_dict[t] = pd.Series(X[i, :], index=wvl_index, name=t)

        # Oxides series: oxide name -> proportion
        oxide_dict[t] = pd.Series(np.hstack((Y[i, :4], Y[i, 5 : 9])), index=ox_index, name=t)


    with open(OUT_DIR / 'SuperCam_Mars_predictors.pickle', 'wb') as file:
        pickle.dump(spec_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_DIR / 'SuperCam_Mars_labels.pickle', 'wb') as file:
        pickle.dump(oxide_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

def download_and_split_data(args: argparse.Namespace, saving_dir: str) -> None:
    # Set up the environment
    ResultDir = saving_dir + '/%s/SampleSplits/'%(args.DataType)
    if not os.path.exists(ResultDir):
        os.makedirs(ResultDir)

    # Download SuperCam or ChemCam datasets if it does not exist 
    if args.DataType == 'PDS_ChemCam':
        Download = DownLoadChemCam(args, saving_dir)
        Download.Parallel_AcquireData()
    elif args.DataType == 'PDS_SuperCam':
        DownLoadSuperCam(saving_dir)
        
    # set random seed 
    for i in range(args.TrialNum):
        np.random.seed(i)
        random.seed(i)

        # Construct variables and get training, validation and testing splits of the original data
        TrSampleName, ValSampleName, TeSampleName = LoadLIBSDataforNetworkTraining_StandardSplit(args.DataType, saving_dir) 
    
        # Save sample splits
        with open(ResultDir + 'Seed%d_TrSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(TrSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ResultDir + 'Seed%d_TeSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(TeSampleName, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ResultDir + 'Seed%d_ValSampleName.pickle'%(i+1), 'wb') as file:
            pickle.dump(ValSampleName, file, protocol=pickle.HIGHEST_PROTOCOL) 