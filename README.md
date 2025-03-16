## Code Repository for the Paper: [Regularization via f-Divergence: An Application to Multi-Oxide Spectroscopic Analysis](https://arxiv.org/pdf/2502.03755)

In our study, we introduce f-divergence regularization and conduct a series of comparative analyses:
1. f-divergence vs. no regularization
2. f-divergence vs. L1 regularization
3. f-divergence vs. L2 regularization
4. f-divergence vs. dropout
5. f-divergence + L1 vs. L1
6. f-divergence + L2 vs. L2
7. f-divergence + dropout vs. dropout

The primary objective of the paper is to predict oxide weight composition, which is a regression task. We evaluate our models using the following metrics:
1. Root Mean Squared Error (RMSE)
2. Pairwise t-test

We conduct experiments using data from ChemCam and SuperCam, available from the [PDS Geosciences Node](https://www.example.com). The code in this repository is fully functional with ChemCam data. Efforts to open-source the code for SuperCam data are ongoing.

### Running the Code
1. Execute `./ChemCam_Run.sh` in the terminal. This script performs extensive network training across various configurations mentioned in the paper and handles automatic downloading and processing of ChemCam data.
2. After training, run `./Eval_ChemCam_RMSE.sh` and `./Eval_ChemCam_TTest.sh` to output RMSE and t-test results respectively.

### Citation
If you find our work insightful, please consider citing it:

```bibtex
@article{li2025regularization,
  title={Regularization via f-Divergence: An Application to Multi-Oxide Spectroscopic Analysis},
  author={Li, Weizhi and Klein, Natalie and Gifford, Brendan and Sklute, Elizabeth and Legett, Carey and Clegg, Samuel},
  journal={arXiv preprint arXiv:2502.03755},
  year={2025}
}
