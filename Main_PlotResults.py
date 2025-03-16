from Options import *
import os

from Utils import MeasurePairedTest,  MeasureTE_EachMethod

def main():
    cwd = os.getcwd() 
    args = parser.parse_args()
    ResultDir = cwd + '/%s/Results/'%args.DataType
    print("Training souce: %s"%args.DataType)

    """
    Calculate Test error for each method including the hyper-parameter selection 
    """
    if args.PrintTE_EachMethod == 1:
        MeasureTE_EachMethod(args, ResultDir)

    if args.PairTest == 1:
        MeasurePairedTest(args, ResultDir)


if __name__ == '__main__':
    main()