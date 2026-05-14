# Variant Classification

This repository contains the results of a survey study conducted on the genomic variant classification problem. In this study, we benchmarked 4 machine learning models on 36 variations of a SNV dataset, to find the most critical factors driving performance. More information on our methodology and results can be found in the paper in `paper.pdf`.

## Repository structure

- `conf/`, directory of the [Hydra](https://hydra.cc/) configuration files
- `experiments/`, directory containing the results of our experiments
- `plot_scripts/`, directory containing the scripts to generate plots
- `Snakefile`, main [Snakemake](https://snakemake.github.io/) file
- `run_conf.py`, entry point for training using Hydra configs
- `load_preprocess.py`, script with utilities for preprocessing configured using configs under `conf/preprocessing/`
- `create_ds.py`, script to create all the used datasets starting from the [dbNSFP](https://zenodo.org/records/15131632) dataset
- `train_from_yaml.py`, script to re-train and save the model's files from a `optimization_results.yaml`, result of a multirun with optimization
- `run_test.py`, script which tests the models on their respective test set, which also runs 2c and 4c models on the uncertain variants' dataset
- `shapley_vals.py`, script which, for each trained model under the `outputs/` directory, computes the shapley values
- `compare_models.py`, script which compares the predictions on the uncertain dataset of multiple 2c and 4c models
- `benchmark.sh`, shell script which trains some models and logs the respective elapsed time and ram usage
- `requirements.txt`, requirements file

## Usage

This repository requires Python 3.10, a venv with said version can be created using uv like this:

```bash
uv venv --python 3.10
```

Then the venv can be activated and the dependencies installed like this:
```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

Once dependencies are installed, we need to create the datasets directories should be creted and the [dbNSFP](https://zenodo.org/records/15131632) dataset downloaded in `datasets/`.

```bash
mkdir datasets datasets_new
```

**Note**: the downloaded dataset will be in the parquet file format, before using it it should be converted to a csv, like this for example:

```python
import polars as pl

source_path = DS_PARQUET_PATH
df = pl.read_parquet(source_path)
df.write_csv("./datasets/dbnsfp5_full.csv")
```

After this setup is done, we can proceed with the experiments, which are done through Snakemek rules.

## Snakemake rules

The default rule, `all`, can be executed usign this command
```bash
snakemake -c all --stats stats.json
```

This will also produce a `stats.json` file which will contain statistics for each executed rule. As said, the default rule is `all`, which will execute the following rules:

- **create_ds**, which will create the datasets using `create_ds.py`
- **train_all**, which, using `run_conf.py`, will launch the optimization multirun of each possible model, with a maximum of 100 train runs for each, this will create the `multirun/` directory with the optimization results under
- **train_optimized**, runs `train_from_yaml.py` to train the optimized models, creating the respective files under the `outputs/` directory
- **run_tests**, runs `run_tests.py` to get the test score results, saving them to `experiments/test_results.csv`, while also gathering the prediction of uncertain variants of the 2c and 4c models
- **shapley_vals**, runs `shapley_vals.py` to compute the shapley values for each model, saving them in `experiments/shapleys.csv`

**Note**: due to the high number of models, this rule may require several days to fully execute depending on the hardware available, for our experiments we used a machine with an AMD EPYC 9V74 cpu and 128 GiB of ram and it took around 84 hours.

Additionally, there are four additional rules to generate performance plots:

- **plot_performance_results**, which generates the f1 and accuracy plots, saving them to `experiments/performance_plots/`
- **plot_f1_results**, which plots only the f1 scores under `experiments/performance_plots`
- **compare_models**, which compares the uncertain predictions' of different models using `compare_models.py`, saving the results under `experiments/uncertain_predictions_comparison/`
- **plot_shap**, which plots the results of the shapley values analysis

To launch a specific snakemake rule, we can use:
```bash
snakemake -c all <rule_name>
```

## Paper plots
Plots used in the paper can be generated using the following commands:
```bash
cd plot_scripts
python plot_full_results.py
python plot_matrixes.py
python plot_shap_column.py
```

Which will save the plots, respectively, to `experiments/combined_performance.pdf`, `experiments/uncertain_predictions_comparison/combined_results.pdf` and `experiments/shapley_results/relative_importance_variance_top7_column.pdf`.