import numpy as np

from sklearn.decomposition import TruncatedSVD

# Regular SVD
A = np.array([[3,4,3],[1,2,3],[4,2,1]])

U, D, VT = np.linalg.svd(A)

A_remake = (U @ np.diag(D) @ VT)

print(A_remake)

# Truncated SVD
B = np.array([[4,3,4],[3,2,1],[1,2,4]])

trun_svd = TruncatedSVD(n_components=2)
B_transformed = trun_svd.fit_transform(B, y=None)

print(B_transformed)
