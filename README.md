Github repository for "Machine learning to predict microbial community functions: An analysis of dissolved organic carbon from litter decomposition"

PLoS One Link:
https://doi.org/10.1371/journal.pone.0215502

Installation:
To install the required dependencies, make sure that Python is installed:

I recommend using the Anaconda Python distribution:
https://www.anaconda.com/products/individual

Alternatively, install Python and then install required modules:
1. download Python from: https://www.python.org/downloads/
2. Install dependencies: pip install -r requirements.txt
3. To run tutorials:     pip install jupyterlab

Usage:
Once Python is installed, clone this repository into a local directory by navigating to the directory in the terminal and run: git clone https://github.com/MunskyGroup/thompson_etal_plos_one_2019.git
Alternatively, download a .zip file of the repository by clicking on the green "Code" button and then click "Download zip".

After cloning the repository, the Jupyter notebook tutorials can be opened and run locally by running: jupyter-lab

In JupyterLab, run any of the ".ipynb" files. I recommend starting with FeatureSelectionTutorial.ipynb or TargetPredictionTutorial.ipynb 

To apply RFINN to other datasets, training and testing data .csv files must be created manually. Training and testing data must match the same organization as the train_data.csv and test_data.csv files with sample IDs in the first column and features in the following columns. A separate .csv file with targets should have sample ID names in each column, with the corresponding "target" (in this case DOC) value listed under each sample ID. For reference, see the DOC_targets.csv file.
