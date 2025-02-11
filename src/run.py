# Required to run multiple processes on Unity for some reason.
import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except:
    pass
mp.freeze_support()

import sys
import traceback
import argparse
import os
import signal
from pathlib import Path
import pickle


from config import *
from data import HydroDataset, HydroDataLoader
from train import Trainer, load_last_state
from evaluate import *

def load_model(run_dir):
    state = load_last_state(run_dir)
    cfg = state[0]
    model = state[1]
    trainer_state = state[2]
    return cfg, model, trainer_state

def start_training(config_yml):
    cfg, _ = read_config(config_yml)
    dataset = HydroDataset(cfg) 
    cfg = set_model_data_args(cfg, dataset)
    dataloader = HydroDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, log_parent=config_yml.parent)
    trainer.start_training()

    return cfg, trainer.model, trainer.log_dir, dataset

def continue_training(run_dir):
    cfg, _, _ = load_model(run_dir)
    dataset = HydroDataset(cfg) 
    dataloader = HydroDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer.model, dataset

def finetune(finetune_yml:Path):
    finetune = read_yml(finetune_yml)
    run_dir = Path(finetune_yml).parent

    # Load the config and manipulate it a bit
    cfg, _, trainer_state = load_model(run_dir)
    stop_epoch = trainer_state['epoch']
    cfg['num_epochs'] = stop_epoch + finetune.get('additional_epochs',0)
    cfg['transition_begin'] = stop_epoch if finetune.get('reset_lr') else 0
    cfg['cfg_path'] = finetune_yml
    # Insert these params directly.
    cfg.update(finetune.get('config_update',{}))

    dataset = HydroDataset(cfg) 
    dataloader = HydroDataLoader(cfg, dataset)
    trainer = Trainer(cfg, dataloader, log_parent=finetune_yml.parent, continue_from=run_dir)
    trainer.start_training()

    return cfg, trainer.model, trainer.log_dir, dataset

def hyperparam_grid_search(config_yml:Path, idx, k_folds=None):
    cfg, _ = read_config(config_yml)
    cfg = update_cfg_from_grid(cfg, idx)

    k = 4 if k_folds is None else k_folds
    for i in range(k):
        cfg['test_basin_file'] = f"metadata/site_lists/k_folds/test_{i}_{k}.txt"
        cfg['train_basin_file'] = f"metadata/site_lists/k_folds/train_{i}_{k}.txt"
        dataset = HydroDataset(cfg) 

        cfg = set_model_data_args(cfg, dataset)
        dataloader = HydroDataLoader(cfg, dataset)
        
        log_dir = config_yml.parent / f"index_{idx}" / f"fold_{i}"
        if log_dir.is_dir():
            trainer = Trainer(cfg, dataloader, continue_from=log_dir)
        else:
            trainer = Trainer(cfg, dataloader, log_dir=log_dir)

        start_epoch = trainer.epoch
        while trainer.epoch < trainer.num_epochs:
            start_epoch = trainer.epoch
            trainer.start_training()
            if trainer.epoch == start_epoch:
                break

        # Check if we actually did some training this run.
        out_file = trainer.log_dir / "test_data.pkl"
        if (start_epoch != trainer.epoch) or (not out_file.is_file()):
            eval_model(cfg, trainer.model, dataset, trainer.log_dir, run_predict=False, run_train=False, make_plots=False)

def load_prediction_model(run_dir, chunk_idx):
    cfg, model, _ = load_model(run_dir)
    train_dataset = HydroDataset(cfg) 

    cfg['data_subset'] = 'predict'
    cfg['basin_file'] = f'metadata/site_lists/predictions/chunk_{chunk_idx:02}.txt'
    cfg['shuffle'] = False # No need to shuffle for inference

    predict_dataset = HydroDataset(cfg, train_ds=train_dataset, use_cache=False)

    eval_dir = run_dir / 'inference'
    eval_dir.mkdir(parents=True, exist_ok=True)

    return cfg, model, predict_dataset, eval_dir


def make_all_plots(cfg, results, bulk_metrics, basin_metrics, data_subset, log_dir):
    fig_dir = log_dir / "figures" / data_subset
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = mosaic_scatter(cfg, results, bulk_metrics, str(log_dir))
    fig.savefig(fig_dir / f"density_scatter.png",  dpi=300)

    metric_args = {
        'R2': {'range':[-1,1]},
        'nBias':{'range':[-100,100]},
        'rRMSE':{'range':[0,500]},
        'KGE':{'range':[-1,1]},
        'NSE':{'range':[-4,1]},
        'Agreement':{'range':[0,1]}
    }
    figs = basin_metric_histograms(basin_metrics, metric_args)
    for target, fig in figs.items():
        fig.savefig(fig_dir / f"{target}_metrics_hist.png",  dpi=300)


def eval_model(cfg, model, dataset, log_dir, 
               run_test: bool | str = True, 
               run_predict: bool | str = True, 
               run_train: bool | str = True, 
               make_plots=True):
    """
    Evaluates a model on specified data subsets, calculates metrics, and optionally generates plots.

    Parameters:
    ----------
    cfg : dict
        Configuration dictionary with parameters for the model and evaluation process.
    model : object
        The trained model to evaluate.
    dataset : object
        The dataset object containing data and methods for creating data subsets.
    log_dir : Path
        Path to the directory where results and logs will be saved.
    run_test, run_predict, run_train : bool | str, optional
        If `True`, evaluates the model on the data subset. If a string, it is treated 
        as the output file stem for test/predict/train subset results. Defaults to `True`.
    make_plots : bool, optional
        If `True`, generates and saves evaluation plots for the specified subsets. 
        Defaults to `True`.

    Notes:
    -----
    - The `predict` subset does not generate plots, regardless of the value of `make_plots`.
      These would be identical to test, as 'predict' covers the same basins and time period
      as 'test' but without validation data.

    Saves:
    ------
    - Pickled results, bulk metrics, and basin metrics for each data subset.
    - Plots in the `log_dir` directory if `make_plots` is `True`.
    """
    
    def eval_data_subset(data_subset, out_stem=None):
        if out_stem is False:
            return
        elif isinstance(out_stem, str):
            if out_stem[-4:]!='.pkl':
                out_stem += '.pkl'
            else:
                print("Pass string")
        else:
            out_stem = f"{data_subset}_data.pkl"
            
        results_file = log_dir / out_stem
        print(f'Evaluating {data_subset} subset and saving to: {results_file}')

        dataset.cfg['exclude_target_from_index'] = None
        dataset.update_indices(data_subset)
        dataloader = HydroDataLoader(cfg, dataset)

        results = predict(model, dataloader, quiet=cfg.get('quiet',True), denormalize=True)

        if data_subset != "predict":
            bulk_metrics = get_all_metrics(results)
            basin_metrics = get_basin_metrics(results)
            if make_plots:
                make_all_plots(cfg, results, bulk_metrics, basin_metrics, data_subset, log_dir) 
            out = (results, bulk_metrics, basin_metrics)
        else:
            out = results

        with open(results_file, 'wb') as f:
            pickle.dump(out, f)

    eval_data_subset('test', run_test)
    eval_data_subset('predict', run_predict)
    eval_data_subset('train', run_train)    


def main(args):
    # Default values
    run_test = run_predict = run_train = True

    if args.train:
        config_yml = Path(args.train).resolve()
        cfg, model, eval_dir, dataset = start_training(config_yml)
    elif args.continue_training:
        run_dir = Path(args.continue_training).resolve()
        cfg, model, dataset = continue_training(run_dir)
        eval_dir = run_dir
    elif args.finetune:
        finetune_yml = Path(args.finetune).resolve()
        cfg, model, eval_dir, dataset = finetune(finetune_yml)
    elif args.grid_search:
        config_yml = Path(args.grid_search).resolve()
        hyperparam_grid_search(config_yml, args.grid_index, args.k_folds)
        return
    elif args.test:
        run_dir = Path(args.test).resolve()
        cfg, model, _ = load_model(run_dir)
        dataset = HydroDataset(cfg) 
        eval_dir = run_dir
    elif args.prediction_model:
        run_test = run_train = False
        run_predict = f'chunk_{args.basin_chunk_index:02}'
        run_dir = Path(args.prediction_model).resolve()
        cfg, model, dataset, eval_dir = load_prediction_model(run_dir, args.basin_chunk_index)

    eval_model(cfg, model, dataset, eval_dir, run_test, run_predict, run_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model based on the command line arguments.")

    # Create a mutually exclusive arg group for train/continue/test. 
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', 
                    type=str, 
                    help='Path to the training configuration file.')
    group.add_argument('--continue', 
                    dest='continue_training', 
                    type=str, 
                    help='Directory path to continue training.')
    group.add_argument('--finetune',
                        type=str,
                        help='Path to the finetune configuration yml file.')
    group.add_argument('--grid_search',
                    type=str,
                    help='Path to the grid search configuration file.')
    group.add_argument('--test',  
                    type=str, 
                    help='Path to directory with model to test.')
    group.add_argument('--prediction_model',
                       type=str,
                       help='Path to directory with model to use for predictions.')
    
    # Add a new argument for grid search index
    parser.add_argument('--grid_index', 
                        type=int, 
                        help='Index in the hyperparameter grid to evaluate (required if --grid_search is used)',
                        required='--grid_search' in sys.argv)
    
    # Add optional argument for number of k-fold validation folds
    parser.add_argument('--k_folds', 
                        type=int, 
                        default=None,
                        help='Number of folds for k-fold cross-validation (optional, can be used with --train or --continue)')
    
    # Add a new argument for prediction chunk index
    parser.add_argument('--basin_chunk_index', 
                        type=int, 
                        help='Index of the chunked basin list to predict on',
                        required='--prediction_model' in sys.argv)
    
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        # Cleanup all processes (prevents zombie dataloader workers)
        os.killpg(os.getpgid(0), signal.SIGKILL)
    