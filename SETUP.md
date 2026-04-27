# Setup Instructions for T&C Scanner

## Prerequisites
- Conda (Miniconda or Anaconda) installed on your system
- Python 3.9 (managed by conda environment)
- Internet connection (to download models and dependencies on first run)

## Remove the old environment (if it exists)

Open a terminal. Then run:

```bash
conda deactivate
conda env remove -n tc_scanner
```
This deletes the old environment completely.

## Step 1: Create the Conda environment

Open a terminal and navigate to the project root (where `environment.yml` is located). Run:

```bash
conda env create -f environment.yml
```
This will create a new environment named tc_scanner with all required packages. It may take a few minutes.

After it finishes, activate the environment:
```bash
conda activate tc_scanner
```
## Step 2: Run the app again
```bash
streamlit run src/app.py
```
On first run, the embedding model (all-MiniLM-L6-v2) and the language model (distilgpt2) will be downloaded automatically (~500 MB total). Subsequent runs will be much faster. This means the first run might need 2-5 minutes, but all subsquent runs will have wait times closer to 10-30 seconds.

## To Use Evaluate.py:
After it finishes, activate the environment:
```bash
conda activate tc_scanner
```
## Step 2: Run the app again
```bash
python src/evaluate.py
```
This will have the model rnning evaluations on our current test set
