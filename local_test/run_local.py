import os, shutil
import sys
import time
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score

sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer 
import algorithm.model_server as model_server
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.clustering as clustering 

import scoring_utils as scoring

inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data", "clusteringBaseMainInput")

model_path = "./ml_vol/model/"
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)

prediction_col = 'prediction'


'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os or python version-related issues, so beware. 
'''

model_name = clustering.MODEL_NAME


def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "clusteringBaseMainInput": None
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            }, 
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                
            }
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)



def copy_example_files(dataset_name):
    # data schema
    shutil.copyfile(f"./examples/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # data
    shutil.copyfile(f"./examples/{dataset_name}_test.csv", os.path.join(data_path, f"{dataset_name}_test.csv"))
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json"))



def train_and_predict(num_clusters):        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data
    data = utils.get_data(data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)   
    # get trained preprocessor, model, training history 
    preprocessor, model = model_trainer.get_trained_model(data, data_schema, hyper_parameters, num_clusters)            
    # Save the processing pipeline   
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model 
    clustering.save_model(model, model_artifacts_path)
    print("done with training")
    
    # instantiate the trained model        
    predictor = model_server.ModelServer(model_artifacts_path)
    # make predictions
    predictions = predictor.predict(data, data_schema)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    results, chart_data = score(data, predictions, data_schema)  
    print("done with predictions") 
    return results, chart_data



def set_dataset_variables_for_testing(dataset_name):
    global id_col, num_clusters, target_col
    if dataset_name == "anisotropic": 
        id_col = "id"; num_clusters = 3; target_col = "y"
    elif dataset_name == "equal_variance_blobs": 
        id_col = "id"; num_clusters = 3; target_col = "y"
    elif dataset_name == "noisy_moons": 
        id_col = "id"; num_clusters = 2; target_col = "y"
    elif dataset_name == "noisy_circles": 
        id_col = "id"; num_clusters = 2; target_col = "y"
    elif dataset_name == "unequal_variance_blobs": 
        id_col = "id"; num_clusters = 3; target_col = "y"
    elif dataset_name == "no_structure": 
        id_col = "id"; num_clusters = 3; target_col = "y"
    else: raise Exception(f"Error: Cannot find dataset = {dataset_name}")


def score(test_data, predictions, data_schema): 
    predictions = predictions.merge(test_data[[id_col, target_col]], on=id_col)
        
    # external validation metrics
    purity = scoring.purity_score(predictions[target_col], predictions[prediction_col])
    ami = adjusted_mutual_info_score(predictions[target_col], predictions[prediction_col])     
    
    # standardize the data before doing internal validation
    scaled_data = scoring.standardize_data(test_data, data_schema)
    # internal validation metrics
    dbi = davies_bouldin_score(scaled_data, predictions[prediction_col])
    chi = calinski_harabasz_score(scaled_data, predictions[prediction_col])
    silhouette = silhouette_score(scaled_data, predictions[prediction_col])
    
    results = {
        "purity": np.round(purity, 4),
        "ami": np.round(ami, 4),
        "dbi": np.round(dbi, 4),
        "chi": np.round(chi, 4),
        "silhouette": np.round(silhouette, 4),
    }
    # print(results)
    
    chart_data = {
        "X": scaled_data,
        "y": predictions[prediction_col]
    }
    return results, chart_data


def save_scoring_outputs(results, dataset_name, chart_data=None):    
    df = pd.DataFrame(results) if dataset_name is None else pd.DataFrame([results])        
    df = df[["model", "dataset_name", "purity", "ami", "dbi", "chi", "silhouette", "elapsed_time_in_minutes"]]    
    file_path_and_name = get_file_path_and_name(dataset_name)
    df.to_csv(file_path_and_name, index=False)
    # print(df)
    
    if chart_data is not None:  
        X, y = chart_data["X"], chart_data["y"]  
        if X.shape[1] > 2:  X = scoring.reduce_dims(X)  
        plt.scatter(X[:,0], X[:, 1], alpha=0.3, c=y)
        file_path_and_name = get_file_path_and_name(dataset_name, file_type="scatter")
        plt.title(dataset_name)
        plt.savefig(file_path_and_name)
        plt.clf()       
        
        
def get_file_path_and_name(dataset_name, file_type="scores"): 
    if file_type == 'scores':
        if dataset_name is None: 
            fname = f"_{model_name}_results.csv"
        else: 
            fname = f"{model_name}_{dataset_name}_results.csv"
    elif file_type == 'scatter':
        if dataset_name is None: 
            raise Exception("Cant do this.")
        else: 
            fname = f"{model_name}_{dataset_name}_scatter.png"
    else: 
        raise Exception(f"Invalid file_type for scatter plot: {file_type}")        
    full_path = os.path.join(test_results_path, fname)
    return full_path



def run_train_and_test(dataset_name):
    start = time.time() 
    
    create_ml_vol()   # create the directory which imitates the bind mount on container
    copy_example_files(dataset_name)   # copy the required files for model training    
    set_dataset_variables_for_testing(dataset_name=dataset_name)
    results, chart_data = train_and_predict(num_clusters)        # train and predict 
    
    end = time.time()
    elapsed_time_in_minutes = np.round((end - start)/60.0, 2)
    
    results = { **results, 
               "model": model_name, 
               "dataset_name": dataset_name, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes 
               }
    
    return results, chart_data


if __name__ == "__main__":     
    
    datasets = ["anisotropic", "equal_variance_blobs", "noisy_moons", 
                "noisy_circles", "unequal_variance_blobs", "no_structure"]
    # datasets = ["anisotropic"]
    
    all_results = []
    for dataset_name in datasets:        
        print("-"*60)
        print(f"Running dataset {dataset_name}")
        results, chart_data = run_train_and_test(dataset_name)
        save_scoring_outputs(results, dataset_name, chart_data)            
        all_results.append(results)
        print("-"*60)
    
    save_scoring_outputs(all_results, dataset_name=None)