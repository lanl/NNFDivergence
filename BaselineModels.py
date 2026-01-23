import numpy as np 
import multiprocessing as mp
from scipy.sparse import csr_matrix 
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

class Baseline():
    def __init__(self, args, TrFeat, TrLabel, ValFeat, ValLabel, scaler):
        self.TrFeat = TrFeat
        self.TrLabel = TrLabel 
        self.ValFeat = ValFeat
        self.ValLabel = ValLabel
        self.Model = None
        self.EmbeddingDim = args.EmbeddingDim
        self.scaler = scaler

    def fit(self):
   
        self.Model = PLSRegression(n_components=self.EmbeddingDim)
        self.Model.fit(self.TrFeat, self.TrLabel)
          
    def mean_squared_relative_error(self, Y_pred, ValLabel):
        A = (Y_pred - ValLabel) ** 2 / (ValLabel ** 2)
        Err = np.mean(A)
        return Err 

    def cut_edges_mst(self, X, Y): # cut-edge in mst
        n, m = len(X), len(Y)
        labels = np.r_[np.zeros(n, dtype=int), np.ones(m, dtype=int)]  # 0=X, 1=Y
        Z = np.vstack([X, Y])

        NumCores = mp.cpu_count()
        D = pairwise_distances(Z, n_jobs=NumCores)

        # ensure diagonal is zero, and avoid zero-weight off-diagonal edges
        np.fill_diagonal(D, 0.0)
        D[(D == 0) & (~np.eye(len(D), dtype=bool))] = 1e-10

        T = minimum_spanning_tree(csr_matrix(D)).toarray()

        # make undirected adjacency
        A = ((T + T.T) > 0)

        # list undirected edges once
        iu, ju = np.where(np.triu(A, 1))
        E = len(iu)
        assert E == (n + m - 1), f"MST edge count {E}, expected {n+m-1}"

        # cut-edge count
        R = np.sum(labels[iu] != labels[ju])
        return int(R), int(E)    
    def FDivergence(self, X, Y):
        """
        Fridman Rafsky statstic (graph estimate of a f-divergence)
        """

        R, G = self.cut_edges_mst(X, Y)
        N = len(X) + len(Y)
        Dp = 1 - 2 * R / N
    
        return max(Dp, 0)
    
    def predict(self):

        Y_pred = self.Model.predict(self.ValFeat)

        if self.scaler != None:
            Y_pred = Y_pred * self.scaler.scale_ + self.scaler.mean_

        mse = mean_squared_error(Y_pred, self.ValLabel)
        Fdiv = self.FDivergence(Y_pred, self.ValLabel)
        return Y_pred, mse, Fdiv



