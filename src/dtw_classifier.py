from dtaidistance import dtw_ndim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DTWKNNClassifier(BaseEstimator, ClassifierMixin):
    """1-NN classifier menggunakan DTW distance"""
    
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X_series, y):
        """X_series: list of MFCC arrays shape (n_mfcc, T)"""
        self.train_series = X_series
        self.train_labels = np.array(y)
        return self
    
    def _dtw_distance(self, s1, s2):
        # dtaidistance pakai (T, n_mfcc) bukan (n_mfcc, T)
        return dtw_ndim.distance(s1.T, s2.T)
    
    def predict(self, X_series):
        predictions = []
        for query in X_series:
            distances = [self._dtw_distance(query, ref) 
                        for ref in self.train_series]
            nearest = np.argsort(distances)[:self.k]
            # majority vote
            votes = self.train_labels[nearest]
            pred = np.bincount([np.where(self.classes_ == v)[0][0] 
                               for v in votes]).argmax()
            predictions.append(self.classes_[pred])
        return predictions
    
    @property
    def classes_(self):
        return np.unique(self.train_labels)
