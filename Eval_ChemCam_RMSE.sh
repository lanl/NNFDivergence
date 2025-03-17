# Evaluate the root mean sqared error (RMSE) of indepedent use of no regularization, L1, L2 and the f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='NoReg'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='ParamSelect_L1'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='ParamSelect_L2'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='ParamSelect_Dropout'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --Lambda=2 --LossType3='ParamSelect_fdiv' 

# Evaluate the root mean sqared error (RMSE) of L1, L2 and droputout with various strength, abd the combination 
# of these methods and the f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='L1'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='L2'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --LossType3='Dropout'
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --Lambda=2 --LossType3='ParamSelect_L1_fdiv' 
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --Lambda=2 --LossType3='ParamSelect_L2_fdiv' 
python Main_PlotResults.py --DataType='PDS_ChemCam' --PrintTE_EachMethod=1 --epochs=50 --lr=1 --TrialNum=2 --Lambda=2 --LossType3='ParamSelect_Dropout_fdiv' 
