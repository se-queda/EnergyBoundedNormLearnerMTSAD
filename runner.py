import argparse
import subprocess
import sys
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
    'A-1',
    'A-2',
    'A-3',
    'A-4',
    'A-5',
    'A-6',
    'A-7',
    'A-8',
    'A-9',
    'B-1',
    'D-1',
    'D-11',
    'D-12',
    'D-13',
    'D-2',
    'D-3',
    'D-4',
    'D-5',
    'D-6',
    'D-7',
    'D-8',
    'D-9',
    'E-1',
    'E-10',
    'E-11',
    'E-12',
    'E-13',
    'E-2',
    'E-3',
    'E-4',
    'E-5',
    'E-6',
    'E-7',
    'E-8',
    'E-9',
    'F-1',
    'F-2',
    'F-3',
    'G-1',
    'G-2',
    'G-3',
    'G-4',
    'G-6',
    'G-7',
    'P-1',
    'P-2',
    'P-3',
    'P-4',
    'P-7',
    'R-1',
}

SMD_SKIP_IDS = {
    'machine-1-1',
    'machine-1-2',
    'machine-1-3',
    'machine-1-4',
    'machine-1-5',
    'machine-1-6',
    'machine-1-7',
    'machine-1-8',
    'machine-2-1',
    'machine-2-2',
    'machine-2-3',
    'machine-2-4',
    'machine-2-5',
    'machine-2-6',
    'machine-2-7',
    'machine-2-8',
    'machine-2-9',
    'machine-3-1',
    'machine-3-10',
    'machine-3-11',
    'machine-3-2',
    'machine-3-3',
    'machine-3-4',
    'machine-3-5',
}

def worker_task(mid, config, perf_path, comp_path, infer_path):
    """Child process executor with clean TF initialization."""
    try:
        for p in (perf_path, comp_path, infer_path):
            parent = os.path.dirname(p)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # GPU Setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Run Training & Scoring
        auc, pr_auc, p_best, r_best, f1_best, vusauc, vuspr, aff_p, aff_r, aff1, train_stats, models, num_samples, inference_stats = train_on_machine(mid, config)
        
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

        # Save performance metrics
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

        # Save computation metrics
        comp_header = [
            "id",
            "total_params",
            "peak_vram_mb",
            "approx_tflops_per_epoch",
            "avg_gpu_util_percent",
            "avg_mem_util_percent",
            "avg_power_w",
            "energy_wh",
            "energy_wh_total",
            "sec_per_epoch",
            "total_time_converge",
        ]
        _ensure_header(comp_path, comp_header)

        total_params = 0
        for m in models:
            total_params += int(np.sum([np.prod(v.shape) for v in m.trainable_variables]))

        steps_per_epoch = int(np.ceil(num_samples / config.get("batch_size", 128)))
        approx_tflops_per_epoch = (2.0 * total_params * steps_per_epoch) / 1e12
        sec_per_epoch = float(np.mean(train_stats["epoch_times"])) if train_stats["epoch_times"] else 0.0
        avg_gpu_util = train_stats.get("avg_gpu_util", float("nan"))
        avg_mem_util = train_stats.get("avg_mem_util", float("nan"))
        avg_power_w = train_stats.get("avg_power_w", float("nan"))
        energy_wh = train_stats.get("energy_wh", float("nan"))
        total_time = train_stats.get("total_time", float("nan"))
        energy_wh_total = avg_power_w * (total_time / 3600.0) if not np.isnan(avg_power_w) else float("nan")

        peak_vram_mb = float("nan")
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            peak_vram_mb = info["peak"] / (1024 * 1024)
        except Exception:
            pass

        with open(comp_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                mid,
                f"{total_params}",
                f"{peak_vram_mb:.2f}",
                f"{approx_tflops_per_epoch:.6f}",
                f"{avg_gpu_util:.2f}",
                f"{avg_mem_util:.2f}",
                f"{avg_power_w:.2f}",
                f"{energy_wh:.4f}",
                f"{energy_wh_total:.4f}",
                f"{sec_per_epoch:.4f}",
                f"{train_stats['converge_time']:.4f}",
            ])
        _compute_and_append_avg(comp_path)

        infer_header = [
            "id",
            "avg_fwd_s",
            "p99_fwd_s",
            "avg_window_s",
            "p99_window_s",
            "throughput_windows_per_s",
            "per_point_ms",
            "per_point_fwd_ms",
        ]
        _ensure_header(infer_path, infer_header)
        if inference_stats is not None:
            with open(infer_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    mid,
                    f"{inference_stats['avg_fwd_s']:.6f}",
                    f"{inference_stats['p99_fwd_s']:.6f}",
                    f"{inference_stats['avg_window_s']:.6f}",
                    f"{inference_stats['p99_window_s']:.6f}",
                    f"{inference_stats['throughput_windows_per_s']:.2f}",
                    f"{inference_stats['per_point_ms']:.4f}",
                    f"{inference_stats['per_point_fwd_ms']:.4f}",
                ])
            _compute_and_append_avg(infer_path)
            
    except Exception as e:
        print(f"❌ Failed on {mid}: {e}")
        traceback.print_exc()
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
    # Drop any previously appended AVG rows anywhere in the file
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
    
    # --- 1. ID Discovery ---
    if dataset_type == "PSM":
        machine_ids = ["PSM_Pooled"]
    # elif dataset_type == "SWAT":
    #     machine_ids = ["SWAT"]
    else:
        # Determine file extension based on dataset
        ext = ".txt" if dataset_type == "SMD" else ".npy"
        train_path = os.path.join(data_root, "train")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Could not find train folder at {train_path}")
            
        # Get all IDs (filenames without extension)
        machine_ids = sorted([f.replace(ext, "") for f in os.listdir(train_path) if f.endswith(ext)])
    
    if dataset_type == "SMAP":
        before = len(machine_ids)
        machine_ids = [mid for mid in machine_ids if mid not in SMAP_SKIP_IDS]
        if before != len(machine_ids):
            print(f"🧹 Skipped {before - len(machine_ids)} SMAP entities via hardcoded resume filter")
    elif dataset_type == "SMD":
        before = len(machine_ids)
        machine_ids = [mid for mid in machine_ids if mid not in SMD_SKIP_IDS]
        if before != len(machine_ids):
            print(f"🧹 Skipped {before - len(machine_ids)} SMD entities via hardcoded resume filter")

    print(f"📂 Found {len(machine_ids)} entities for {dataset_type}")

    os.makedirs(out_dir, exist_ok=True)
    perf_path = os.path.join(out_dir, "performance.csv")
    comp_path = os.path.join(out_dir, "computation.csv")
    infer_path = os.path.join(out_dir, "inference.csv")
    # Serial execution: one entity at a time, no multiprocessing.
    for mid in machine_ids:
        worker_task(mid, config, perf_path, comp_path, infer_path)

    _compute_and_append_avg(perf_path)
    _compute_and_append_avg(comp_path)
    _compute_and_append_avg(infer_path)


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


def _run_dataset_process(dataset, cfg, seed_dir):
    run_all_entities(cfg, seed_dir)


def _serial_schedule(config, seed_label):
    datasets = ["MSL", "SMAP", "SMD"]
    for dataset in datasets:
        cfg = _dataset_cfg(config, dataset)
        if not cfg.get("data_root"):
            continue
        out_dir = os.path.join("results", dataset, seed_label)
        os.makedirs(out_dir, exist_ok=True)
        run_all_entities(cfg, out_dir)


def _parallel_schedule(config, seed_label):
    ctx = mp.get_context("spawn")
    procs = {}

    def start_dataset(ds):
        cfg = _dataset_cfg(config, ds)
        if not cfg.get("data_root"):
            return None
        out_dir = os.path.join("results", ds, seed_label)
        os.makedirs(out_dir, exist_ok=True)
        p = ctx.Process(target=_run_dataset_process, args=(ds, cfg, out_dir))
        p.start()
        return p

    p_smd = start_dataset("SMD")
    p_smap = start_dataset("SMAP")
    p_msl = start_dataset("MSL")

    if p_smap:
        p_smap.join()
    if p_msl:
        p_msl.join()

    p_psm = start_dataset("PSM")

    if p_smd:
        p_smd.join()
    if p_psm:
        p_psm.join()

    # SWAT excluded from training pipeline


def run_suite(config, seeds, parallel=False):
    datasets = ["SMD", "SMAP", "MSL", "PSM"]
    seed_avgs_by_dataset = {d: [] for d in datasets}

    for i, s in enumerate(seeds):
        os.environ["SUITE_SEED"] = str(s)
        if i == 0:
            os.environ["SUITE_SAVE_SCORES"] = "1"
        else:
            os.environ["SUITE_SAVE_SCORES"] = "0"
        seed_manager.initialize_seeds(int(s))
        seed_label = f"seed_{s}"

        if parallel:
            _parallel_schedule(config, seed_label)
        else:
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
    parser.add_argument("--suite", action="store_true", help="Run all datasets for N seeds")
    parser.add_argument("--all", action="store_true", help="Run all datasets once (seed_single)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for suite runs")
    parser.add_argument("--parallel", action="store_true", help="Use parallel dataset schedule")
    parser.add_argument("--serial", action="store_true", help="With --all, run MSL/SMAP/SMD serially into seed_single dirs")
    # Optional override for quick testing
    parser.add_argument("--id", type=str, help="Specify a single machine/channel ID to test")
    args = parser.parse_args()

    # Assuming load_config is available in your environment
    config = load_config(args.config)
    dataset_type = config.get("dataset", "SMD").upper()
    
    if args.suite:
        rng = np.random.default_rng()
        seeds_path = "results/suite_seeds.txt"
        os.makedirs("results", exist_ok=True)
        if os.path.isfile(seeds_path):
            with open(seeds_path, "r") as f:
                seeds = [int(x.strip()) for x in f.read().split(",") if x.strip()]
        else:
            seeds = rng.integers(0, 1_000_000, size=args.seeds).tolist()
            with open(seeds_path, "w") as f:
                f.write(",".join(str(s) for s in seeds))
        run_suite(config, seeds, parallel=args.parallel)
        return

    if args.all:
        if args.serial:
            _serial_schedule(config, "seed_single")
            return
        dataset = config.get("dataset", "SMD").upper()
        cfg = _dataset_cfg(config, dataset)
        if not cfg.get("data_root"):
            raise ValueError(f"data_root not set for dataset {dataset}")
        out_dir = os.path.join("results", dataset, "seed_single")
        os.makedirs(out_dir, exist_ok=True)
        run_all_entities(cfg, out_dir)
        return

    if args.parallel:
        _parallel_schedule(config, "seed_single")
        return
    else:
        # Logic for a single-point test run
        if args.id:
            mid = args.id
        else:
            # Smart defaults for single runs based on dataset
            defaults = {"SMD": "machine-1-1", "MSL": "T-10", "SMAP": "D-12", "PSM": "PSM_Pooled"}
            mid = defaults.get(dataset_type, "test_entity")
        print(f"🧪 Starting single-entity test run: {mid}")
        csv_path = f"results/{dataset_type}_performance_metrics.csv"
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=worker_task,
            args=(
                mid,
                config,
                csv_path,
                f"results/{dataset_type}_computation_metrics.csv",
                f"results/{dataset_type}_inference.csv",
            ),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            if p.exitcode < 0:
                print(f"❌ Worker for {mid} terminated by signal {-p.exitcode}. Likely OOM or a fatal TF error.")
            else:
                print(f"❌ Worker for {mid} exited with code {p.exitcode}.")
        p.close()
        _compute_and_append_avg(csv_path)
        _compute_and_append_avg(f"results/{dataset_type}_computation_metrics.csv")
        _compute_and_append_avg(f"results/{dataset_type}_inference.csv")

    # Clean legacy score dumps
    try:
        legacy_scores = os.path.join("scores", dataset_type, f"{mid}_scores.csv")
        if os.path.isfile(legacy_scores):
            os.remove(legacy_scores)
    except Exception:
        pass

if __name__ == "__main__":
    main()
