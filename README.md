# AI Pipeline for African Wild Dogs Biologging


## Dependencies & Quickstart
The code uses Python version 3.11. The required Python packages can be installed by setting up the conda environment as follows.
```
conda env create -f environment.yml
conda activate wildlife
```

Next, please install PyTorch following the installation instructions for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Make sure you have [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) installed to create and manage Conda environments.
**Hardware Recommendation:** We recommend using a machine with:
- At least 32 GB of CPU RAM.
- A CUDA-enabled GPU with a minimum of 12 GB RAM for training machine learning models.

> Note: The GPU is only necessary for training models in the PyTorch framework. If you are not training models (e.g., for dataset manipulation or creating train-test splits), a GPU is not required.
To install PyTorch with CUDA support, follow these [instructions](https://pytorch.org/) for your specific CUDA distribution.

A step-by-step guide for setting up the data, generating metadata, creating train and test splits based on the metadata, creating fixed-length windows for training, training the model, and evaluating its performance is available in `quickstart.ipynb`.

## Experiment Setups and Results

The experiment setups differ in their train-test splits chosen specifically so that the train and test data satisfy use-specified controls. For example, the experiment setup with AWD *green* in test set and remaining dogs in the train set is called *interdog*. These filteres for train and test set can be specified by modifying the `get_exp_filter_profiles` function in `src/utils/data.py`. The filter profile received from the function is used to obtain the filtered dataframes using `filter_data` method in `src/data_prep/data_prep_utils.py`.

We use the following four experiment setups.

| Setup | Train split | Test split |
|-------|-------------|------------|
| No split | NA | NA |
| Interdog | Ash, Jessie, Palus, Fossey | Green |
| Interyear | 2021 | 2022|
| InterAMPM| | AM | PM | 

The split details for each experimental setup are:

| Details | No split | Interdog | Interyear | InterAMPM |
|---------|----------|----------|-----------|-----------|
| Train set size | 14978 | 13104 | 9528 | 13712|
| Validation set size | 3745 | 3277 | 2382 | 3429 |
| Test set size | 4645 | 6987 | 11458 | 6227 | 

## Training

To train the classification model, including both the prediction and conformal models, for a predefined experiment setup which specifies the train and test split, use the following command:
```
python scripts/train.py --experiment_name <experiment_name> --window_duration_percentile <window_duration_percentile> --alpha <alpha>
```
where
- `<experiment_name>` is one of: *no split*, *interdog*, *interyear*, and *interAMPM*.
- `<window_duration_percentile>` selects the window duration based on the percentile of all windows in the dataset.
- `<alpha>` specifies the confidence level for the conformal model (default is 0.95).

Additional parameters, such as the prediction model architecture (e.g., number of convolution layers, kernel size) and training settings (e.g., learning rate, batch size), can also be configured. Refer to `scripts/train.py` for a full list of adjustable parameters.

## Reproducing Results

The figures and tables from the main text can be reproduced by running the four notebooks - `no_split.ipynb`, `interdog.ipynb`, `interyear.ipynb`, and `interAMPM.ipynb` for the four experimental setups. 
