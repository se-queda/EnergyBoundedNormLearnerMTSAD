import argparse
import csv
import gc
import os
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import seed_manager
from train import train_on_machine, load_config, _resolve_data_root

SMAP_SKIP_IDS = {

}

def worker_task(mid, config, perf_path):
    try:
        parent = os.path.dirname(perf_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        auc, pr_auc, p_best, r_best, f1_best, vusauc, vuspr, aff_p, aff_r, aff1, train_stats, inference_stats, diagnosis_stats, parameter_stats, models = train_on_machine(mid, config)
        inf_path = os.path.join(os.path.dirname(perf_path), "inference.csv")
        diag_path = os.path.join(os.path.dirname(perf_path), "diagnosis.csv")
        param_path = os.path.join(os.path.dirname(perf_path), "parameter.csv")
        
        def _ensure_header(path, header):
            if not os.path.isfile(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(header)
                return
            try:
                with open(path, "r", newline="") as f:
                    rows = list(csv.reader(f))
                if not rows or rows[0] != header:
                    backup = path + ".bak"
                    os.replace(path, backup)
                    with open(path, "w", newline="") as f:
                        csv.writer(f).writerow(header)
            except Exception:
                backup = path + ".bak"
                try:
                    os.replace(path, backup)
                except Exception:
                    pass
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(header)
        perf_header = ["id", "auc", "prauc", "p_best", "r_best", "f1_best", "vusaucc", "vuspr", "aff_p", "aff_r", "aff1"]
        _ensure_header(perf_path, perf_header)
        with open(perf_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                mid,
                f"{auc:.4f}",
                f"{pr_auc:.4f}",
                f"{p_best:.4f}",
                f"{r_best:.4f}",
                f"{f1_best:.4f}",
                f"{vusauc:.4f}",
                f"{vuspr:.4f}",
                f"{aff_p:.4f}",
                f"{aff_r:.4f}",
                f"{aff1:.4f}",
            ])
        _compute_and_append_avg(perf_path)

        inf_header = ["id", "elapsed_minutes", "ms_per_sample"]
        _ensure_header(inf_path, inf_header)
        with open(inf_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                mid,
                f"{inference_stats['elapsed_minutes']:.6f}",
                f"{inference_stats['ms_per_sample']:.6f}",
            ])
        _compute_and_append_avg(inf_path)

        if diagnosis_stats is not None:
            diag_header = ["id", "hr_100", "hr_150", "ndcg_100", "ndcg_150", "ips_100", "ips_150"]
            _ensure_header(diag_path, diag_header)
            with open(diag_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    mid,
                    f"{diagnosis_stats['hr_100']:.6f}",
                    f"{diagnosis_stats['hr_150']:.6f}",
                    f"{diagnosis_stats['ndcg_100']:.6f}",
                    f"{diagnosis_stats['ndcg_150']:.6f}",
                    f"{diagnosis_stats['ips_100']:.6f}",
                    f"{diagnosis_stats['ips_150']:.6f}",
                ])
            _compute_and_append_avg(diag_path)

        param_header = ["id", "total_params", "trainable_params", "non_trainable_params"]
        _ensure_header(param_path, param_header)
        with open(param_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                mid,
                str(parameter_stats["total_params"]),
                str(parameter_stats["trainable_params"]),
                str(parameter_stats["non_trainable_params"]),
            ])
        _compute_and_append_avg(param_path)

    except Exception as e:
        print(f" Failed on {mid}: {e}")
        traceback.print_exc()
        raise
    finally:
        try:
            K.clear_session()
        except Exception:
            pass
        gc.collect()

def _compute_and_append_avg(csv_path):
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return
    rows = [r for r in rows if not (r and r[0].strip().upper() == "AVG")]
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        return
    vals = []
    for r in data_rows:
        try:
            vals.append([float(x) for x in r[1:]])
        except Exception:
            continue
    if not vals:
        return
    avg = np.mean(np.array(vals, dtype=float), axis=0)
    rows.append(["AVG"] + [f"{x:.6f}" for x in avg])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def run_all_entities(config, out_dir):
    dataset_type = config.get("dataset", "SMD").upper()
    data_root = _resolve_data_root(config, dataset_type)
    if not data_root:
        raise ValueError(f"data_root not set for dataset {dataset_type}")
    if dataset_type == "PSM":
        machine_ids = ["PSM_Pooled"]
    elif dataset_type == "SMD" and config.get("smd_compact", False):
        machine_ids = ["SMD_Compact"]
    elif dataset_type == "SWAT":
        machine_ids = ["SWAT"]
    else:
        ext = ".txt" if dataset_type == "SMD" else ".npy"
        train_path = os.path.join(data_root, "train")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Could not find train folder at {train_path}")
            
        machine_ids = sorted([f.replace(ext, "") for f in os.listdir(train_path) if f.endswith(ext)])
    
    if dataset_type == "SMAP":
        before = len(machine_ids)
        machine_ids = [mid for mid in machine_ids if mid not in SMAP_SKIP_IDS]
        if before != len(machine_ids):
            print(f"🧹 Skipped {before - len(machine_ids)} SMAP entities via hardcoded resume filter")

    print(f"Found {len(machine_ids)} entities for {dataset_type}")

    os.makedirs(out_dir, exist_ok=True)
    perf_path = os.path.join(out_dir, "performance.csv")
    for mid in machine_ids:
        worker_task(mid, config, perf_path)

    _compute_and_append_avg(perf_path)


def _get_data_root_for_dataset(config, dataset):
    return _resolve_data_root(config, dataset)


def _dataset_cfg(config, dataset):
    cfg = dict(config)
    cfg["dataset"] = dataset
    cfg["data_root"] = _get_data_root_for_dataset(cfg, dataset)
    return cfg


def _collect_seed_avg(perf_path, seed, out_list):
    if not os.path.isfile(perf_path):
        return
    with open(perf_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[-1][0].strip().upper() == "AVG":
        avg_row = rows[-1]
        try:
            avg_vals = [float(x) for x in avg_row[1:]]
            out_list.append([seed] + avg_vals)
        except Exception:
            pass


def _write_final_results(dataset, seed_avgs):
    if not seed_avgs:
        return
    dataset_dir = os.path.join("results", dataset)
    final_path = os.path.join(dataset_dir, "final_results.csv")
    header = ["seed", "auc", "prauc", "p_best", "r_best", "f1_best", "vusaucc", "vuspr", "aff_p", "aff_r", "aff1"]
    with open(final_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in seed_avgs:
            writer.writerow([row[0]] + [f"{x:.6f}" for x in row[1:]])
        vals = np.array([r[1:] for r in seed_avgs], dtype=float)
        mean_vals = np.mean(vals, axis=0)
        std_vals = np.std(vals, axis=0)
        writer.writerow(["MEAN"] + [f"{x:.6f}" for x in mean_vals])
        writer.writerow(["STD"] + [f"{x:.6f}" for x in std_vals])


def run_experiments(config, datasets, seeds):
    datasets = [d.upper() for d in datasets]
    seed_avgs_by_dataset = {d: [] for d in datasets}

    for s in seeds:
        os.environ["SUITE_SEED"] = str(s)
        seed_manager.initialize_seeds(int(s))
        seed_label = f"seed_{s}"

        for dataset in datasets:
            cfg = _dataset_cfg(config, dataset)
            if not cfg.get("data_root"):
                continue
            out_dir = os.path.join("results", dataset, seed_label)
            os.makedirs(out_dir, exist_ok=True)
            run_all_entities(cfg, out_dir)

        for dataset in datasets:
            perf_path = os.path.join("results", dataset, seed_label, "performance.csv")
            _collect_seed_avg(perf_path, s, seed_avgs_by_dataset[dataset])

    for dataset in datasets:
        _write_final_results(dataset, seed_avgs_by_dataset[dataset])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--all", action="store_true", help="Run all entities once for the dataset in config.yaml")
    parser.add_argument("--datasets", nargs="+", help="Datasets for the final experimental run, e.g. --datasets MSL SMAP SMD")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds for the final experimental run, e.g. --seeds 42 43")
    parser.add_argument("--id", type=str, help="Specify a single machine/channel ID to test")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_type = config.get("dataset", "SMD").upper()

    if (args.datasets is None) ^ (args.seeds is None):
        parser.error("--datasets and --seeds must be provided together")

    if args.datasets and args.seeds:
        run_experiments(config, args.datasets, args.seeds)
        return

    if args.all:
        dataset = config.get("dataset", "SMD").upper()
        cfg = _dataset_cfg(config, dataset)
        if not cfg.get("data_root"):
            raise ValueError(f"data_root not set for dataset {dataset}")
        out_dir = os.path.join("results", dataset, "seed_single")
        os.makedirs(out_dir, exist_ok=True)
        run_all_entities(cfg, out_dir)
        return

    else:
        # Logic for a single-point test run
        if args.id:
            mid = args.id
        else:
            # defaults for single runs based on dataset
            defaults = {"MSL": "C-1", "SMAP": "A-1"}
            mid = defaults.get(dataset_type, "test_entity")
            if dataset_type == "SMD" and config.get("smd_compact", False):
                mid = "SMD_Compact"
        print(f"starting single-entity test run: {mid}")
        csv_path = f"results/{dataset_type}_performance_metrics.csv"
        worker_task(mid, config, csv_path)
        _compute_and_append_avg(csv_path)


if __name__ == "__main__":
    main()
