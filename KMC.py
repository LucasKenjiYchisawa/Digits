import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.externals import joblib

#Loading the data from sklearn datasets
digits = load_digits()
data=scale(digits.data)
#Setting the data as the output
y=data.targets

#Obtaining the number of groups the numerical digitals are classified into
k=len(np.unique(y))

samples,features = data.shape

#To measure the accuracy when using the K Means Clustering, we need to use the function
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#Training the method
clf=KMeans(n_clusters=k, init= ‘random’,  n_init=10)
bench_k_means(clk, “l”, data)
