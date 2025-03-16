# TTest for no regularization V.S. f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --LossType3='NoReg_fdiv'
# TTest for l2 V.S. f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --LossType3='l2_fdiv'
# TTest for dropout V.S. f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --LossType3='Dropout_fdiv'
# TTest for l1 V.S. f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --LossType3='l1_fdiv'

# TTest for l1 V.S. l1 + f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --Lambda=2 --LossType3='l1_fdivMixture' 
# TTest for l2 V.S. l2 + f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --Lambda=2 --LossType3='l2_fdivMixture' 
# TTest for dropout V.S. dropout + f-divergence regularization
python Main_PlotResults.py --DataType='PDS_ChemCam' --PairTest=1 --epochs=50 --lr=1 --Lambda=2 --TrialNum=2 --Lambda=2 --LossType3='Dropout_fdivMixture' 
