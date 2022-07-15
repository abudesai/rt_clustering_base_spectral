import numpy as np
import sys
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def get_input_vars_lists(data_schema):      
    input_vars = []
    attributes = data_schema["inputDatasets"]["clusteringBaseMainInput"]["inputFields"]   
    for attribute in attributes: 
        if attribute["dataType"] == "CATEGORICAL" or attribute["dataType"] == "NUMERIC":
            input_vars.append(attribute["fieldName"])
    return input_vars


def standardize_data(data, data_schema):
    input_vars = get_input_vars_lists(data_schema)    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[input_vars])
    return scaled_data


def reduce_dims(X): 
    reducer = TSNE(n_components=2)
    # reducer = PCA(n_components=2)
    
    
    reduced_dim_data = reducer.fit_transform(X)
    return reduced_dim_data
    