# Download and split the data
python Main_SampleSplit.py --DataType='PDS_ChemCam' --TrialNum=15

# train models 
for i in {1..2}
do 
    # no regularization
    python Main_OneShot_Standardsplit.py --DataType='PDS_ChemCam' --LossType='L2' --seed=$i --epochs=50 --batch-size=16 --lr=1 --delta=0
    
    # l1 regularization 
    for j in 0.00005 0.0001 0.0003 0.0005 0.0007 0.001
    do 
        python Main_OneShot_Standardsplit.py --DataType='PDS_ChemCam' --LossType='L1' --seed=$i --epochs=50 --batch-size=16 --lr=1 --delta=$j
    done 
    
    # l2 regularization
    for j in 0.00005 0.0001 0.0003 0.0005 0.0007 0.001
    do 
        python Main_OneShot_Standardsplit.py --DataType='PDS_ChemCam' --LossType='L2' --seed=$i --epochs=50 --batch-size=16 --lr=1 --delta=$j
    done 
 
    # dropout
    for j in 0.03 0.04 0.05 0.06 0.07 0.08 0.1
    do 
        python Main_OneShot_Standardsplit.py --DataType='PDS_ChemCam' --LossType='Dropout' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Dropout_prob=$j
    done 

    # F-divergence regularization 
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00005 --delta=0.001
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00007 --delta=0.0012
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00009 --delta=0.0014
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.0001 --delta=0.02
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00012 --delta=0.005
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00014 --delta=0.01
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.00016 --delta=0.015
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.0003 --delta=0.015
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.0005 --delta=0.02
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.0007 --delta=0.025
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.005 --delta=0.03
    python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --weight=0.01 --delta=0.02

    # F-divergence regularization + l1 
    for j in 0.0001 0.0003 0.0005 0.0007
    do
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L1_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l1delta=$j --weight=0.00007 --delta=0.0012
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L1_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l1delta=$j --weight=0.0003 --delta=0.015
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L1_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l1delta=$j --weight=0.00014 --delta=0.01
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L1_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l1delta=$j --weight=0.005 --delta=0.03
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L1_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l1delta=$j --weight=0.00016 --delta=0.015
    done

    # F-divergence regularization + l2 
    for j in 0.0001 0.0003 0.0005 0.0007
    do
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L2_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l2delta=$j --weight=0.00007 --delta=0.0012
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L2_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l2delta=$j --weight=0.0003 --delta=0.015
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L2_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l2delta=$j --weight=0.00014 --delta=0.01
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L2_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l2delta=$j --weight=0.005 --delta=0.03
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='L2_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --l2delta=$j --weight=0.00016 --delta=0.015
    done

    # F-divergence regularization + dropout 
    for j in 0.04 0.06 0.08 0.1
    do
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='Dropout_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --Dropout_prob=$j --weight=0.00007 --delta=0.0012
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='Dropout_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --Dropout_prob=$j --weight=0.0003 --delta=0.015
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='Dropout_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --Dropout_prob=$j --weight=0.00014 --delta=0.01
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='Dropout_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --Dropout_prob=$j --weight=0.005 --delta=0.03
        python Main_TwoSampleStatsNN_StandardSplit.py --DataType='PDS_ChemCam' --LossType2='Dropout_fdiv' --seed=$i --epochs=50 --batch-size=16 --lr=1 --Lambda=2 --Dropout_prob=$j --weight=0.00016 --delta=0.015
    done
done 