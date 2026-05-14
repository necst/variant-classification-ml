import re
import multiprocessing

CORES = multiprocessing.cpu_count()

TRAIN_DATASETS = ["dbnfs_2c_bal_100k_100000", "dbnfs_2c_bal_10k_10000", "dbnfs_2c_bal_full_218142",
            "dbnfs_3c_bal_100k_100000", "dbnfs_3c_bal_10k_10000", "dbnfs_3c_bal_full_327213",
            "dbnfs_4c_bal_100k_100000", "dbnfs_4c_bal_10k_10000", "dbnfs_4c_bal_full_123932",
            "dbnfs_5c_bal_100k_100000", "dbnfs_5c_bal_10k_10000", "dbnfs_5c_bal_full_154915"]
NO_UNC_DATASETS = [ds for ds in TRAIN_DATASETS if "2c" in ds or "4c" in ds]
TEST_DATASETS = ["dbnfs_2c_test", "dbnfs_3c_test", "dbnfs_4c_test", "dbnfs_5c_test"]
UNC_DATASET = "dbnfs_uncertain"

BASE_CONF = "config"
MODELS_CONF = ["logistic_regression", "mlp", "random_forest", "xgboost"]
MODELS_NAMES = ["logreg", "mlp", "randfor", "xgboost"]
MODEL_MAPPING = dict(zip(MODELS_NAMES, MODELS_CONF))
PREPROC_CONF = ["all_features", "drop_raw_metadata_keep_clinical_context", "drop_raw_metadata_and_labs"]
FILL_NA_STRATEGY = "median"
SCALER = "standard"
MULTIRUN_RUNS = 100
COMPARE_COMBINATIONS = [
    ("xgboost", "dbnfs_2c_bal_full_218142", "all_features"),
    ("xgboost", "dbnfs_4c_bal_full_123932", "all_features"),
    ("randfor", "dbnfs_2c_bal_full_218142", "all_features"),
    ("randfor", "dbnfs_4c_bal_full_123932", "all_features"),
]

rule all:
    input:
        "experiments/test_results.csv",
        "experiments/shapleys.csv",
        expand("outputs/{model_name}_{ds}_{pc}/uncertain_predictions.csv", model_name=MODELS_NAMES, ds=NO_UNC_DATASETS, pc=PREPROC_CONF),


rule create_ds:
    threads: CORES
    input:
        ds="datasets/dbnsfp5_full.csv"
    output:
        train_datasets=expand("datasets_new/{ds}.csv", ds=TRAIN_DATASETS),
        test_datasets=expand("datasets_new/{ds}.csv", ds=TEST_DATASETS),
        unc_dataset=f"datasets_new/{UNC_DATASET}.csv"
    shell:
        """
        python create_ds.py
        rm datasets_new/temp.csv
        """

rule train_all:
    threads: CORES
    wildcard_constraints:
        model_name = "|".join([re.escape(x) for x in MODELS_NAMES]),
        ds         = "|".join([re.escape(x) for x in TRAIN_DATASETS]),
        pc         = "|".join([re.escape(x) for x in PREPROC_CONF])
    input:
        dataset="datasets_new/{ds}.csv",
        base_conf=f"conf/{BASE_CONF}.yaml",
        preproc_conf="conf/preprocessing/{pc}.yaml",
        ds_conf="conf/dataset/{ds}.yaml",
        model_conf=lambda wildcards: f"conf/model/{MODEL_MAPPING[wildcards.model_name]}.yaml"
    output:
        "multirun/{model_name}_{ds}_{pc}/optimization_results.yaml"
    params:
        hydra_model_name = lambda w: MODEL_MAPPING[w.model_name],
        scaler_flag = lambda w: f"preprocessing.scaler={SCALER}" if w.model_name in ("mlp", "logreg") else "",
        xgb_2c_flag = lambda w: "model.objective=binary:logistic model.eval_metric=logloss" if w.model_name == "xgboost" and "2c" in w.ds else ""
    shell:
        """
        python run_conf.py \
            model={params.hydra_model_name} \
            {params.xgb_2c_flag} \
            dataset={wildcards.ds} \
            preprocessing={wildcards.pc} \
            preprocessing.fill_na_strategy={FILL_NA_STRATEGY} \
            {params.scaler_flag} \
            hydra.sweeper.ax_config.max_trials={MULTIRUN_RUNS} -m 
        """

rule train_optimized:
    threads: CORES
    input:
        opt_res_path = "multirun/{model_name}_{ds}_{pc}/optimization_results.yaml"
    output:
        "outputs/{model_name}_{ds}_{pc}/model.joblib"
    shell:
        "python train_from_yaml.py {input.opt_res_path}"

rule run_tests:
    threads: CORES
    input:
        expand("outputs/{model_name}_{ds}_{pc}/model.joblib", model_name=MODELS_NAMES, ds=TRAIN_DATASETS, pc=PREPROC_CONF)
    output:
        "experiments/test_results.csv",
        expand("outputs/{model_name}_{ds_nu}_{pc}/uncertain_predictions.csv", model_name=MODELS_NAMES, ds_nu=NO_UNC_DATASETS, pc=PREPROC_CONF)
    shell:
        "python run_test.py"

rule shapley_vals:
    threads: CORES
    input:
        expand("outputs/{model_name}_{ds}_{pc}/model.joblib", model_name=MODELS_NAMES, ds=TRAIN_DATASETS, pc=PREPROC_CONF)
    output:
        "experiments/shapleys.csv"
    shell:
        "python shapley_vals.py"



# Rules for plots
rule plot_performance_results:
    threads: CORES
    input:
        "experiments/test_results.csv"
    output:
        directory("experiments/performance_plots")
    shell:
        "cd plot_scripts && python plot_acc_results.py && python plot_f1_results.py"

rule plot_f1_results:
    threads: CORES
    input:
        "experiments/test_results.csv"
    output:
        directory("experiments/performance_plots")
    shell:
        "cd plot_scripts && python plot_f1_results.py"


rule compare_models:
    threads: CORES
    input:
        [f"outputs/{m}_{d}_{p}/uncertain_predictions.csv" for m, d, p in COMPARE_COMBINATIONS]
    output:
        directory("experiments/uncertain_predictions_comparison")
    shell:
        "python compare_models.py"

rule plot_shap:
    threads: CORES
    input:
        "experiments/shapleys.csv",
        "plot_scripts/plot_shap.py"
    output:
        directory("experiments/shapley_results")
    shell:
        "python plot_scripts/plot_shap.py"
