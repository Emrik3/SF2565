# Kernel Methods vs. Neural Networks


### Loading Data & Preprocessing
The dataset is unfortunatly too large to include in the gid repo but there is a download script in the data folder that pulls the datset from kaggle and unzips it. The next step before any model is trained is to do the feature extraction step. This is done by running the transfrom_data.py script. This took about 2.5 hours on a 4 core CPU but if a GPU is available in the enviroment it should use that. No guarantees though.
> **Note**: The dataset automatically handles CheXpert's uncertain labels (`-1`) by converting them to `0` (negative) for binary classification.

# Neural Network
## Files Overview
- **`run.sh`**: Easy to use shell script for running models and parameter sweeps
- **`feature_dataset.py`**: Dataset class for loading pre-extracted features from `train.pt`
- **`model.py`**: Flexible neural network architectures (simple, deep, and customizable)
- **`config.py`**: Training configuration and metrics tracking
- **`compare_experiments.py`**: Compare different models
- **`trainer.py`**: Main training loop with validation, checkpointing, and visualization
- **`train.py`**: Easy-to-use training script with multiple preset modes
- **`parameter_sweep.py`**: Automated hyperparameter tuning
- **`evaluate.py`**: Model evaluation and inference
- **`plot_time_complexity.py`**: Time complexity analysis and plotting
- **`example_time_analysis.py`**: Example usage of time complexity analysis
- **`data`**: Directory containing feature extracted data


## Usage
./src/run.sh [command]

Command | Effect
------------|--------------
simple            | Train simple 2-layer network
default           | Train default 3-layer network
deep              | Train deep 5-layer network
sweep-lr          | Sweep learning rates
sweep-arch        | Sweep architectures
sweep-dropout     | Sweep dropout rates
compare           | Compare all experiments
help              | Show this help message



# Kernel Methods
The Kernel script is made up of several different functions that initialize and train its subsequent model. To run one or more of these just uncomment the ones in question in the main file and fun the script. 
### kernel.py/run_regularization
Produces the regularization graph for the train and test error for different regularization strengths. The funciton only does one at the time so the body of thefunctio needs to be modified in the sense that a line need to be commented out.
### kernel.pyrun_train_test_err_size
This produces a graph that plots the train test error for different sizes of sample sizes.
### kernel.pyRun_time_mem_test & plot_times
These two work in conjunction to produce the graph compares the execution time of training a kernel SVC with different amouts of samples.
### kernel.pyrun_nyström (not used)
This performs CV and finds the best SVC using  Nyström approximation
### kernel.pyrun_RFF (not used)
This performs CV and finds the best SVC using  RFF approximation

### gs.py/Nystoem_gs & gs.py/RFF_cv
These does the CV for the approximation methods that are used in the report

### gs.py/plot_time_accuracy (AI Generated)
This produces a comparative plot that compare the execution time and accuracy of both the Nyström and RFF method.








