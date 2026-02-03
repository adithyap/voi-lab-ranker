"""
Lab Test Value Ranking with a Transformer backbone on the PhysioNet 2012 ICU dataset.

Designed for a single Colab cell run:
- Downloads a public ICU time-series dataset (PhysioNet/CinC Challenge 2012).
- Preprocesses and caches resampled trajectories.
- Trains a Transformer-based mortality risk model with dynamic batch sizing.
- Derives per-lab expected utility by masking each lab and measuring cross-entropy gain.
- Benchmarks against simple baselines (global utility frequency and random ranking, plus tabular classifiers).
- Produces plots/tables for paper-ready artifacts and a verbose textual summary.
- Saves artifacts to disk and provides a ready-to-run zip+download cell.

Usage in Colab:
    %run lab_test_ranker.py
All configuration is inside CONFIG below—no CLI flags needed.
"""

import io
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier

# -------------------------- Configuration -------------------------- #

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()

CONFIG = {
    "base_url": "https://physionet.org/files/challenge-2012/1.0.0",
    "splits": ["set-a"],  # accessible without credential
    "max_records": 2000,  # use more patients for stronger signal
    "max_time_hours": 48,
    "resample_minutes": 30,  # finer temporal granularity than hourly
    "invalidate_cache": False,  # keep preprocessed/cache by default
    "force_redownload_raw": False,  # set True to re-pull raw txt even if cached
    "cache_dir": ROOT / "lab_rank_cache",
    "results_dir": ROOT / "lab_rank_results",
    "seed": 42,
    "train": {
        "epochs": 80,
        "initial_batch_size": 64,
        "min_batch_size": 4,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "patience": 10,  # early stopping patience (epochs)
        "num_workers": 0,  # set >0 for speed; 0 avoids multiprocessing teardown issues in notebooks
    },
    "model": {"d_model": 128, "nhead": 8, "num_layers": 6, "dropout": 0.2},
    "force_cpu": False,  # set True to disable CUDA entirely (e.g., on AMD/Intel GPUs)
    # Secondary intervention target for treatment-conditional VOI
    "intervention_target": "MechVent",
    "intervention_horizon_hours": 6,  # predict whether intervention starts within horizon
    "loss_weights": {"mortality": 1.0, "intervention": 0.5},
    # Optional lab costs (relative units); defaults to 1.0 if unspecified
    "lab_costs": {
        "Lactate": 3.0,
        "PaO2": 3.0,
        "PaCO2": 3.0,
        "pH": 3.0,
        "Cholesterol": 1.0,
        "Glucose": 1.0,
    },
    "lambda_cost": 0.0,  # set >0 to activate cost-aware ranking in evaluate_budgeted
    "reuse_artifacts": True,  # skip retrain if matching metrics already exist
}

# Canonical variable list from the Challenge 2012 definition.
VARIABLES = [
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "ICUType",
    "K",
    "Lactate",
    "MAP",
    "MechVent",
    "Mg",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Age",
    "Gender",
    "Height",
    "Weight",
]

LAB_LIKE = [
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "Glucose",
    "HCO3",
    "HCT",
    "K",
    "Lactate",
    "Mg",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "TroponinI",
    "TroponinT",
    "WBC",
]


# -------------------------- Utilities -------------------------- #

def set_seed(seed: int, prefer_cuda: bool = True) -> bool:
    """
    Set RNG seeds; return whether CUDA seeding succeeded. If CUDA seeding fails,
    we force CPU-only by masking CUDA devices and reseed CPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    if prefer_cuda and torch.cuda.is_available():
        try:
            torch.manual_seed(seed)  # seeds CPU and triggers CUDA seed internally
            torch.cuda.manual_seed_all(seed)
            return True
        except RuntimeError as e:
            print(f"CUDA seeding failed ({e}); forcing CPU-only run.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.manual_seed(seed)
            return False
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.manual_seed(seed)
        return False


def ensure_dirs(config):
    config["cache_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    (config["results_dir"] / "figs").mkdir(parents=True, exist_ok=True)
    (config["results_dir"] / "tables").mkdir(parents=True, exist_ok=True)


def hash_config(config: Dict) -> str:
    """
    Generate a stable hash for arbitrary (possibly nested) config dicts that may
    contain Path objects or numpy types. Paths are stringified via default=str.
    """
    serializable = json.dumps(config, sort_keys=True, default=str)
    return md5(serializable.encode()).hexdigest()[:10]


def maybe_invalidate(config):
    if config["invalidate_cache"]:
        pre = config["cache_dir"] / "preprocessed.npz"
        if pre.exists():
            pre.unlink()
            print("Preprocessed cache cleared.")


# -------------------------- Data Download -------------------------- #


def download_challenge_records(config) -> Tuple[pd.DataFrame, List[Path]]:
    cache_dir = config["cache_dir"]
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = raw_dir / "Outcomes-a.csv"

    if outcomes_path.exists():
        outcomes = pd.read_csv(outcomes_path)
    else:
        url = f"{config['base_url']}/Outcomes-a.txt"
        print(f"Downloading labels from {url}")
        text = requests.get(url, timeout=30).text
        outcomes = pd.read_csv(io.StringIO(text))
        outcomes.to_csv(outcomes_path, index=False)

    outcomes = outcomes[outcomes["In-hospital_death"].isin([0, 1])]
    record_ids = outcomes["RecordID"].tolist()[: config["max_records"]]

    # If raw cached files already cover desired records and no force_redownload, reuse
    cached_files = {int(p.stem): p for p in raw_dir.glob("*.txt")}
    file_paths = []
    if not config.get("force_redownload_raw", False):
        available = [rid for rid in record_ids if rid in cached_files]
        if len(available) == len(record_ids):
            file_paths = [cached_files[rid] for rid in available]
            return outcomes.set_index("RecordID"), file_paths

    # Otherwise download missing ones
    for rid in tqdm(record_ids, desc="Downloading ICU records"):
        fname = raw_dir / f"{rid}.txt"
        if fname.exists() and not config.get("force_redownload_raw", False):
            file_paths.append(fname)
            continue
        url = f"{config['base_url']}/set-a/{rid}.txt"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"Failed {rid} ({resp.status_code}); skipping")
            continue
        fname.write_bytes(resp.content)
        file_paths.append(fname)
    return outcomes.set_index("RecordID"), file_paths


# -------------------------- Preprocessing -------------------------- #


def _to_minutes(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 2:
        h, m = parts
        return int(h) * 60 + int(m)
    return 0.0


def clean_value(val):
    try:
        val = float(val)
    except ValueError:
        return np.nan
    if val <= -1:
        return np.nan
    return val


def resample_record(
    path: Path, label: int, cfg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(path)
    df["minutes"] = df["Time"].apply(_to_minutes)
    df["Value"] = df["Value"].apply(clean_value)

    max_minutes = cfg["max_time_hours"] * 60
    step = cfg["resample_minutes"]
    times = np.arange(0, max_minutes + step, step)
    n_steps = len(times)
    n_feat = len(VARIABLES)
    values = np.full((n_steps, n_feat), np.nan, dtype=np.float32)
    mask = np.zeros_like(values, dtype=np.float32)
    deltas = np.full((n_steps, n_feat), cfg["max_time_hours"] * 60, dtype=np.float32)

    # Static parameters we forward fill
    static_vars = {"Age", "Gender", "Height", "ICUType", "Weight"}

    for var_idx, var in enumerate(VARIABLES):
        var_df = df[df["Parameter"] == var]
        if var_df.empty:
            continue
        # Collapse duplicate timestamps (mean) to avoid reindex duplicate-label error
        series = (
            var_df.groupby("minutes")["Value"]
            .mean()
            .sort_index()
        )
        # Use last observation carried forward within window
        resampled = series.reindex(times, method="ffill")
        values[:, var_idx] = resampled.values
        mask[:, var_idx] = ~np.isnan(resampled.values)

        # time since last observation in minutes
        last_seen = np.full(n_steps, cfg["max_time_hours"] * 60, dtype=np.float32)
        prev = None
        for t_idx, present in enumerate(mask[:, var_idx]):
            if present:
                prev = times[t_idx]
                last_seen[t_idx] = 0.0
            elif prev is not None:
                last_seen[t_idx] = times[t_idx] - prev
        deltas[:, var_idx] = last_seen

    # Forward fill static vars if present
    for var in static_vars:
        if var in VARIABLES:
            idx = VARIABLES.index(var)
            if np.all(np.isnan(values[:, idx])):
                continue
            first_val = values[:, idx][~np.isnan(values[:, idx])]
            if len(first_val) > 0:
                values[:, idx] = first_val[0]
                mask[:, idx] = 1.0

    return values, mask, deltas, label


def build_datasets(config):
    preprocess_path = config["cache_dir"] / "preprocessed.npz"
    if preprocess_path.exists():
        data = np.load(preprocess_path, allow_pickle=True)
        return (
            data["values"],
            data["masks"],
            data["deltas"],
            data["labels"],
        )

    outcomes, file_paths = download_challenge_records(config)
    X, M, D, y = [], [], [], []
    for path in tqdm(file_paths, desc="Parsing + resampling"):
        rid = int(path.stem)
        if rid not in outcomes.index:
            continue
        label = int(outcomes.loc[rid, "In-hospital_death"])
        v, m, d, l = resample_record(path, label, config)
        X.append(v)
        M.append(m)
        D.append(d)
        y.append(l)

    values = np.stack(X)
    masks = np.stack(M)
    deltas = np.stack(D)
    labels = np.array(y, dtype=np.int64)
    np.savez_compressed(preprocess_path, values=values, masks=masks, deltas=deltas, labels=labels)
    return values, masks, deltas, labels


# -------------------------- Dataset -------------------------- #


class ICUSequence(Dataset):
    def __init__(
        self,
        values,
        masks,
        deltas,
        labels,
        stats,
        max_minutes,
        intervention_labels=None,
        onset_steps=None,
    ):
        self.values = values
        self.masks = masks
        self.deltas = deltas
        self.labels = labels
        self.stats = stats
        self.max_minutes = max_minutes
        self.intervention_labels = (
            intervention_labels if intervention_labels is not None else np.zeros_like(labels)
        )
        self.onset_steps = onset_steps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        v = self.values[idx]
        m = self.masks[idx]
        d = self.deltas[idx]
        # Normalize using training stats
        v_norm = (v - self.stats["mean"]) / (self.stats["std"] + 1e-6)
        v_norm = np.nan_to_num(v_norm, nan=0.0)
        d_norm = np.clip(d / self.max_minutes, 0, 1.0)
        x = np.concatenate([v_norm, m, d_norm], axis=1)  # [T, 3F]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(m).float(),
            torch.tensor(float(bool(self.labels[idx]))).float(),
            torch.tensor(float(self.intervention_labels[idx])).float(),
        )


def compute_stats(train_values, train_masks):
    mean = np.nanmean(np.where(train_masks == 1, train_values, np.nan), axis=(0, 1))
    std = np.nanstd(np.where(train_masks == 1, train_values, np.nan), axis=(0, 1))
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    return {"mean": mean, "std": std}


def compute_intervention_labels(values, masks, var_name: str, horizon_steps: int):
    """
    Build binary intervention labels:
    - target = 1 if the intervention variable turns on (value>0.5) at any point.
    - onset_idx captures the first observed on-timestamp.
    - horizon label is 1 if onset occurs within horizon_steps of sequence end.
    """
    labels = []
    onset_steps = []
    if var_name not in VARIABLES:
        return np.zeros(len(values)), np.full(len(values), np.inf)
    idx = VARIABLES.index(var_name)
    for i in range(len(values)):
        mask_i = masks[i][:, idx]
        vals_i = values[i][:, idx]
        on_steps = np.where((mask_i > 0.5) & (vals_i > 0.5))[0]
        if len(on_steps) == 0:
            labels.append(0.0)
            onset_steps.append(np.inf)
        else:
            onset = int(on_steps[0])
            labels.append(1.0 if onset <= values.shape[1] - 1 else 0.0)
            onset_steps.append(onset)
    labels = np.array(labels, dtype=np.float32)
    onset_steps = np.array(onset_steps)
    horizon_labels = (onset_steps <= horizon_steps).astype(np.float32)
    return labels, onset_steps, horizon_labels


# -------------------------- Model -------------------------- #


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class LabTransformer(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg["nhead"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg["num_layers"]
        )
        self.pos = PositionalEncoding(d_model, max_len=500)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_mortality = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.head_intervention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, attn_mask=None):
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.transformer(h, src_key_padding_mask=attn_mask)
        # attn_mask: True where we want to mask (padding)
        h = h.transpose(1, 2)  # [B, D, T]
        pooled = self.pool(h).squeeze(-1)
        return {
            "mortality": self.head_mortality(pooled).squeeze(-1),
            "intervention": self.head_intervention(pooled).squeeze(-1),
        }


# -------------------------- Training -------------------------- #


def build_loaders(values, masks, deltas, labels, config):
    idx = np.arange(len(labels))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.3, random_state=config["seed"], stratify=labels
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=config["seed"], stratify=y_temp
    )

    X_train, M_train, D_train = values[idx_train], masks[idx_train], deltas[idx_train]
    X_val, M_val, D_val = values[idx_val], masks[idx_val], deltas[idx_val]
    X_test, M_test, D_test = values[idx_test], masks[idx_test], deltas[idx_test]

    stats = compute_stats(X_train, M_train)
    max_minutes = config["max_time_hours"] * 60
    horizon_steps = int((config.get("intervention_horizon_hours", 6) * 60) / config["resample_minutes"])
    int_labels, onset_steps, horizon_labels = compute_intervention_labels(
        values, masks, var_name=config.get("intervention_target", "MechVent"), horizon_steps=horizon_steps
    )
    train_ds = ICUSequence(
        X_train, M_train, D_train, y_train, stats, max_minutes, intervention_labels=int_labels[idx_train], onset_steps=onset_steps[idx_train]
    )
    val_ds = ICUSequence(
        X_val, M_val, D_val, y_val, stats, max_minutes, intervention_labels=int_labels[idx_val], onset_steps=onset_steps[idx_val]
    )
    test_ds = ICUSequence(
        X_test, M_test, D_test, y_test, stats, max_minutes, intervention_labels=int_labels[idx_test], onset_steps=onset_steps[idx_test]
    )
    return train_ds, val_ds, test_ds, stats, horizon_steps


def make_loader(dataset, batch_size, shuffle, device=None, num_workers=None):
    if num_workers is None:
        num_workers = CONFIG["train"].get("num_workers", 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
    )


def tune_batch_size(model, dataset, config, device):
    bs = config["train"]["initial_batch_size"]
    min_bs = config["train"]["min_batch_size"]
    nw = config["train"].get("num_workers", 0)
    while bs >= min_bs:
        try:
            loader = make_loader(dataset, bs, shuffle=True, device=device, num_workers=nw)
            batch = next(iter(loader))
            if len(batch) == 3:
                x, mask, _ = batch
            else:
                x, mask, _, _ = batch
            x, mask = x.to(device), mask.to(device)
            with torch.no_grad():
                _ = model(x, attn_mask=None)
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                bs = bs // 2
                print(f"OOM at batch {bs*2}, retrying with {bs}")
            else:
                raise e
    return min_bs


def train_model(model, train_ds, val_ds, criteria, loss_weights, config, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"], eta_min=config["train"]["lr"] * 0.1
    )

    best_val = -1
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    patience = config["train"].get("patience", 0)
    epochs_since_improve = 0

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        loader = make_loader(
            train_ds, config["train"]["batch_size"], shuffle=True, device=device, num_workers=config["train"].get("num_workers", 0)
        )
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        running = 0.0
        for step, batch in enumerate(pbar, 1):
            if len(batch) == 3:
                x, _, y = batch
                y_int = None
            else:
                x, _, y, y_int = batch
            x, y = x.to(device), y.to(device)
            y_int = y_int.to(device) if y_int is not None else None
            outputs = model(x)
            loss = criteria["mortality"](outputs["mortality"], y)
            if y_int is not None:
                loss = loss + loss_weights.get("intervention", 0.0) * criteria["intervention"](
                    outputs["intervention"], y_int
                )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()
            running += loss.item()
            if step % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / len(loader)
        val_loss, val_auc = evaluate_model(model, val_ds, criteria, device)
        if math.isnan(val_auc):
            val_auc = 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.3f}"
        )
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), config["results_dir"] / "best_model.pt")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        scheduler.step()
        if patience and epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val AUROC improvement for {patience} epochs).")
            break
    return history, best_val


def evaluate_model(model, dataset, criteria, device, return_preds=False):
    model.eval()
    loader = make_loader(
        dataset,
        batch_size=64,
        shuffle=False,
        device=device,
        num_workers=CONFIG["train"].get("num_workers", 0),
    )
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _, y = batch
                y_int = None
            else:
                x, _, y, y_int = batch
            x, y = x.to(device), y.to(device)
            y_int = y_int.to(device) if y_int is not None else None
            outputs = model(x)
            loss = criteria["mortality"](outputs["mortality"], y)
            if y_int is not None:
                loss = loss + criteria["intervention"](outputs["intervention"], y_int)
            losses.append(loss.item())
            probs.append(torch.sigmoid(outputs["mortality"]).cpu().numpy())
            labels.append(y.cpu().numpy())
    probs = np.concatenate(probs) if probs else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    try:
        auc = metrics.roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    if return_preds:
        return float(np.mean(losses)), auc, probs, labels
    return float(np.mean(losses)), auc


# -------------------------- Utility Estimation -------------------------- #


def cross_entropy_loss(prob, label):
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    if label == 1:
        return -math.log(prob)
    return -math.log(1 - prob)


def compute_utilities(model, dataset, device, target="mortality"):
    model.eval()
    utilities = defaultdict(list)  # key: lab -> list of utilities per patient
    patient_utils = []  # per patient dict
    patient_probs = []  # base probabilities for selected target
    loader = make_loader(
        dataset,
        batch_size=32,
        shuffle=False,
        device=device,
        num_workers=CONFIG["train"].get("num_workers", 0),
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc="Utility scoring"):
            if len(batch) == 3:
                x, mask, y = batch
                y_int = None
            else:
                x, mask, y, y_int = batch
            x = x.to(device)
            y_np = y.numpy()
            y_int_np = y_int.numpy() if y_int is not None else None
            outputs = model(x)
            base_logit = outputs[target]
            base_prob = torch.sigmoid(base_logit).cpu().numpy()
            batch_size, T, feat2 = x.shape
            F = len(VARIABLES)
            for i in range(batch_size):
                if target == "mortality":
                    label = int(y_np[i])
                else:
                    label = int(y_int_np[i]) if y_int_np is not None else 0
                patient_probs.append(float(base_prob[i]))
                sample_utils = {}
                for f_name in LAB_LIKE:
                    if f_name not in VARIABLES:
                        continue
                    f_idx = VARIABLES.index(f_name)
                    xi = x[i].clone()
                    # zero out value and mask; set delta to max-normalized (1.0)
                    xi[:, f_idx] = 0.0
                    xi[:, f_idx + F] = 0.0
                    xi[:, f_idx + 2 * F] = 1.0
                    prob_masked = torch.sigmoid(
                        model(xi.unsqueeze(0).to(device))[target]
                    ).cpu().item()
                    ce_full = cross_entropy_loss(base_prob[i], label)
                    ce_mask = cross_entropy_loss(prob_masked, label)
                    util = ce_mask - ce_full  # positive => benefit
                    utilities[f_name].append(util)
                    sample_utils[f_name] = util
                patient_utils.append(sample_utils)
    return utilities, patient_utils, patient_probs


def compute_lab_propensity(masks: np.ndarray) -> Dict[str, float]:
    """Estimate probability a lab is observed (anytime in window)."""
    props = {}
    for f_name in LAB_LIKE:
        if f_name not in VARIABLES:
            continue
        idx = VARIABLES.index(f_name)
        obs = masks[:, :, idx]  # [N, T]
        props[f_name] = float(np.mean(obs > 0.5))
    return props


def aggregate_global_utils(patient_utils, propensities=None, min_prop=1e-3):
    """Aggregate per-patient utilities into global scores with optional IPW."""
    totals = defaultdict(float)
    counts = defaultdict(float)
    for utils in patient_utils:
        for lab, val in utils.items():
            w = 1.0
            if propensities and lab in propensities:
                w = 1.0 / max(propensities[lab], min_prop)
            totals[lab] += w * val
            counts[lab] += w
    return {lab: (totals[lab] / counts[lab]) if counts[lab] > 0 else 0.0 for lab in totals}


def evaluate_budgeted(patient_utils, global_scores, budgets=(1, 3, 5), seed=42, costs=None, lambda_cost=0.0):
    """Simulate budgeted selection utility retention (optionally cost-aware)."""
    rng = random.Random(seed)
    results = {}
    for b in budgets:
        oracle = []
        global_sel = []
        random_sel = []
        for utils in patient_utils:
            ranked_oracle = sorted(utils, key=lambda k: utils[k], reverse=True)
            ranked_global = sorted(utils, key=lambda k: max(global_scores.get(k, 0.0), 0.0), reverse=True)
            ranked_random = list(utils.keys())
            rng.shuffle(ranked_random)
            def utility_with_cost(test):
                base = utils.get(test, 0.0)
                c = costs.get(test, 1.0) if costs else 1.0
                return base - lambda_cost * c
            def sum_top(ranked): return float(np.sum([utility_with_cost(t) for t in ranked[:b]]))
            oracle.append(sum_top(ranked_oracle))
            global_sel.append(sum_top(ranked_global))
            random_sel.append(sum_top(ranked_random))
        oracle_mean = np.mean(oracle)
        results[b] = {
            "oracle": float(oracle_mean),
            "global": float(np.mean(global_sel)),
            "random": float(np.mean(random_sel)),
            "retention_global": float(np.mean(global_sel) / oracle_mean) if oracle_mean else float("nan"),
            "retention_random": float(np.mean(random_sel) / oracle_mean) if oracle_mean else float("nan"),
        }
    return results


def redundancy_audit(patient_utils, masks, threshold=0.0):
    """Count ordered labs whose utility is <= threshold."""
    counts = {"ordered": 0, "low_utility": 0}
    per_lab_low = defaultdict(int)
    for i, utils in enumerate(patient_utils):
        m = masks[i]  # [T, F]
        for f_name in LAB_LIKE:
            if f_name not in VARIABLES:
                continue
            idx = VARIABLES.index(f_name)
            ordered = np.any(m[:, idx] > 0.5)
            if not ordered:
                continue
            counts["ordered"] += 1
            if utils.get(f_name, 0.0) <= threshold:
                counts["low_utility"] += 1
                per_lab_low[f_name] += 1
    return counts, dict(per_lab_low)


def risk_trajectory(model, sample_x, device, step_stride=4):
    """Compute rolling risk p(t) by truncating sequence after each step."""
    model.eval()
    T = sample_x.shape[0]
    traj = []
    with torch.no_grad():
        for t in range(0, T, step_stride):
            x_t = sample_x.clone()
            # zero out future timepoints
            x_t[t + 1 :, :] = 0.0
            prob = torch.sigmoid(model(x_t.unsqueeze(0).to(device))["mortality"]).cpu().item()
            traj.append((t, prob))
    return traj


def temporal_utility_heatmap(model, sample_x, device, labs, step_stride=4):
    """Compute utility over time for selected labs."""
    model.eval()
    T = sample_x.shape[0]
    F = len(VARIABLES)
    rows = []
    with torch.no_grad():
        for t in range(0, T, step_stride):
            x_base = sample_x.clone()
            x_base[t + 1 :, :] = 0.0
            base_prob = torch.sigmoid(model(x_base.unsqueeze(0).to(device))["mortality"]).cpu().item()
            for lab in labs:
                if lab not in VARIABLES:
                    continue
                idx = VARIABLES.index(lab)
                x_mask = x_base.clone()
                x_mask[:, idx] = 0.0
                x_mask[:, idx + F] = 0.0
                x_mask[:, idx + 2 * F] = 1.0
                prob_mask = torch.sigmoid(model(x_mask.unsqueeze(0).to(device))["mortality"]).cpu().item()
                util = cross_entropy_loss(prob_mask, 0) - cross_entropy_loss(base_prob, 0)
                rows.append({"t": t, "lab": lab, "utility": util})
    return pd.DataFrame(rows)


def compute_lead_time_gain(model, dataset, global_scores, config, device, risk_threshold=0.5):
    """
    Estimate how early high-utility labs appear relative to a risk spike.
    - Detect first time mortality risk > threshold.
    - Detect first observation time of any top-5 global lab.
    - Lead time = baseline_daily_time - detection_time, baseline_daily_time ~24h.
    """
    step_minutes = config["resample_minutes"]
    baseline_minutes = min(config["max_time_hours"] * 60, 24 * 60)
    gains = []
    detector_times = []
    spike_times = []
    top_labs = sorted(global_scores, key=global_scores.get, reverse=True)[:5]
    for i in range(len(dataset)):
        x, mask, _, _ = dataset[i]
        traj = risk_trajectory(model, x.clone(), device, step_stride=1)
        spike_idx = next((t for t, p in traj if p >= risk_threshold), None)
        if spike_idx is None:
            continue
        # earliest observation of any top lab
        m_np = mask.numpy()
        obs_indices = []
        for lab in top_labs:
            if lab not in VARIABLES:
                continue
            idx = VARIABLES.index(lab)
            obs = np.where(m_np[:, idx] > 0.5)[0]
            if len(obs):
                obs_indices.append(obs[0])
        if not obs_indices:
            continue
        detect_idx = min(obs_indices)
        spike_minutes = spike_idx * step_minutes
        detect_minutes = detect_idx * step_minutes
        spike_times.append(spike_minutes)
        detector_times.append(detect_minutes)
        gains.append(baseline_minutes - detect_minutes)
    return {
        "mean_lead_minutes": float(np.mean(gains)) if gains else float("nan"),
        "median_lead_minutes": float(np.median(gains)) if gains else float("nan"),
        "n_patients": len(gains),
        "baseline_minutes": baseline_minutes,
        "mean_spike_minutes": float(np.mean(spike_times)) if spike_times else float("nan"),
        "mean_detect_minutes": float(np.mean(detector_times)) if detector_times else float("nan"),
    }


def ndcg_at_k(true_utils: Dict[str, float], ranked_tests: List[str], k: int = 5):
    dcg = 0.0
    for i, test in enumerate(ranked_tests[:k], 1):
        rel = max(true_utils.get(test, 0.0), 0.0)  # NDCG expects non-negative relevance
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal = sorted([max(v, 0.0) for v in true_utils.values()], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], 1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(patient_utils, strategy: str, global_scores=None, seed=42):
    rng = random.Random(seed)
    ndcgs = []
    topk_utils = {1: [], 3: [], 5: []}
    for utils in patient_utils:
        if strategy == "oracle":
            ranked = sorted(utils, key=lambda k: utils[k], reverse=True)
        elif strategy == "global":
            ranked = sorted(
                utils, key=lambda k: max(global_scores.get(k, 0.0), 0.0), reverse=True
            )
        elif strategy == "random":
            ranked = list(utils.keys())
            rng.shuffle(ranked)
        else:
            raise ValueError(strategy)
        ndcgs.append(ndcg_at_k(utils, ranked, k=5))
        for k in topk_utils.keys():
            chosen = ranked[:k]
            avg_util = np.mean([utils.get(t, 0) for t in chosen])
            topk_utils[k].append(avg_util)
    return {
        "mean_ndcg@5": float(np.mean(ndcgs)),
        "top1": float(np.mean(topk_utils[1])),
        "top3": float(np.mean(topk_utils[3])),
        "top5": float(np.mean(topk_utils[5])),
    }


def bootstrap_auc_ci(probs, labels, n_boot=200, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    for _ in range(n_boot):
        idx = rng.randint(0, len(labels), len(labels))
        try:
            aucs.append(metrics.roc_auc_score(labels[idx], probs[idx]))
        except ValueError:
            continue
    if not aucs:
        return (float("nan"), float("nan"))
    aucs = np.array(aucs)
    lower = np.percentile(aucs, 100 * (alpha / 2))
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


# -------------------------- Baselines -------------------------- #


def aggregate_static_features(values, masks):
    # Take last observed value per variable
    last_vals = []
    for v, m in zip(values, masks):
        last = []
        for j in range(v.shape[1]):
            obs = v[:, j][m[:, j] == 1]
            last.append(obs[-1] if len(obs) else np.nan)
        last_vals.append(last)
    arr = np.array(last_vals, dtype=np.float32)
    fill = np.nanmean(arr, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    arr = np.where(np.isnan(arr), fill, arr)
    return arr


def run_tabular_baselines(train_X, test_X, train_y, test_y, config):
    baseline_cache = config["cache_dir"] / "baseline_results.json"
    key = hash_config({"config": config, "baseline": "tabular"})
    if baseline_cache.exists():
        saved = json.loads(baseline_cache.read_text())
        if saved.get("key") == key:
            return saved

    results = {}
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X)
    test_Xs = scaler.transform(test_X)

    log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
    log_reg.fit(train_Xs, train_y)
    probs = log_reg.predict_proba(test_Xs)[:, 1]
    results["logreg_auc"] = metrics.roc_auc_score(test_y, probs)

    gbdt = GradientBoostingClassifier()
    gbdt.fit(train_X, train_y)
    probs = gbdt.predict_proba(test_X)[:, 1]
    results["gbdt_auc"] = metrics.roc_auc_score(test_y, probs)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced", n_jobs=-1
    )
    rf.fit(train_X, train_y)
    probs = rf.predict_proba(test_X)[:, 1]
    results["rf_auc"] = metrics.roc_auc_score(test_y, probs)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=50,
        alpha=1e-4,
        random_state=config["seed"],
    )
    mlp.fit(train_Xs, train_y)
    probs = mlp.predict_proba(test_Xs)[:, 1]
    results["mlp_auc"] = metrics.roc_auc_score(test_y, probs)

    baseline_cache.write_text(json.dumps({"key": key, "results": results}, indent=2))
    return {"key": key, "results": results}


# -------------------------- Plotting -------------------------- #


def plot_losses(history, outdir):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["val_auc"], label="Val AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Validation AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "val_auc.png", dpi=200)
    plt.close()


def plot_top_tests(global_util, outdir):
    util_items = sorted(global_util.items(), key=lambda kv: kv[1], reverse=True)[:15]
    if not util_items:
        return
    tests, vals = zip(*util_items)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(vals), y=list(tests), palette="Blues_r")
    plt.xlabel("Average utility (CE gain)")
    plt.ylabel("Lab test")
    plt.title("Most informative lab tests")
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "top_tests.png", dpi=200)
    plt.close()


def plot_ranking_metrics(metrics_dict, outdir):
    labels = list(metrics_dict.keys())
    ndcgs = [metrics_dict[k]["mean_ndcg@5"] for k in labels]
    plt.figure(figsize=(5, 4))
    sns.barplot(x=labels, y=ndcgs, palette="Set2")
    plt.ylabel("Mean NDCG@5")
    plt.title("Ranking quality by strategy")
    plt.tight_layout()
    plt.savefig(outdir / "figs" / "ndcg.png", dpi=200)
    plt.close()


# -------------------------- Reporting -------------------------- #


def build_summary(
    val_auc,
    test_auc,
    ranking_metrics,
    baseline_res,
    global_util,
    intervention_util,
    policy_gains,
    budget_metrics,
    redundancy_counts,
    test_ci,
    baseline_ci,
    lead_time_stats,
    config,
    outdir,
):
    top_tests = sorted(global_util.items(), key=lambda kv: kv[1], reverse=True)[:10]
    lines = []
    lines.append("=== Training run summary ===")
    lines.append(f"Val AUROC (Transformer): {val_auc:.3f}")
    lines.append(
        f"Test AUROC (Transformer): {test_auc:.3f} "
        f"[95% CI {test_ci[0]:.3f}, {test_ci[1]:.3f}]"
    )
    lines.append("")
    lines.append("Baselines (tabular):")
    for k, v in baseline_res["results"].items():
        ci = baseline_ci.get(k)
        if ci:
            lines.append(f"  {k}: {v:.3f} [95% CI {ci[0]:.3f}, {ci[1]:.3f}]")
        else:
            lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("Ranking (utility) metrics:")
    for name, m in ranking_metrics.items():
        lines.append(
            f"  {name}: NDCG@5={m['mean_ndcg@5']:.3f} top1={m['top1']:.4f} top3={m['top3']:.4f} top5={m['top5']:.4f}"
        )
    lines.append("")
    lines.append("Decision-centered utility (avg CE gain, nats):")
    for k, v in policy_gains.items():
        lines.append(f"  Top-{k} vs random: {v:.5f}")
    lines.append("")
    lines.append("Budgeted selection (utility retention vs oracle):")
    for b, res in budget_metrics.items():
        lines.append(
            f"  Budget {b}: retention_global={res['retention_global']:.3f} retention_random={res['retention_random']:.3f}"
        )
    lines.append("")
    lines.append(
        f"Redundancy audit: {redundancy_counts.get('low_utility', 0)} / {redundancy_counts.get('ordered', 1)} ordered labs had non-positive utility."
    )
    lines.append("")
    lines.append("Top-10 informative labs (avg CE gain; positive=more predictive info):")
    for lab, val in top_tests:
        lines.append(f"  {lab}: {val:.6f}")
    lines.append("")
    lines.append("Treatment-conditional utilities (intervention head) top-5:")
    top_int = sorted(intervention_util.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for lab, val in top_int:
        lines.append(f"  {lab}: {val:.6f}")
    lines.append("")
    all_utils = list(global_util.values())
    if all_utils:
        lines.append(
            f"Mean CE gain across labs (nats): {float(np.mean(all_utils)):.6f}; median: {float(np.median(all_utils)):.6f}"
        )
        lines.append(
            "Interpretation: CE gain ~0.01 nats ≈ 0.014 bits reduction in predictive uncertainty for mortality."
        )
        lines.append("")
    if lead_time_stats and not math.isnan(lead_time_stats.get("mean_lead_minutes", float('nan'))):
        lines.append(
            f"Lead-time to risk spike using top-5 labs: mean {lead_time_stats['mean_lead_minutes']:.1f} min, "
            f"median {lead_time_stats['median_lead_minutes']:.1f} min (n={lead_time_stats['n_patients']}); "
            f"baseline daily labs assumed at {lead_time_stats['baseline_minutes']} min."
        )
        lines.append("")
    lines.append("Methodology notes:")
    lines.append(
        "  - Outcome label: in-hospital death (binary) from PhysioNet 2012 Outcomes-a."
    )
    lines.append(
        "  - Expected clinical value proxy: delta in per-patient cross-entropy when a lab is masked vs observed; CE gain > 0 implies the lab reduces predictive uncertainty/risk."
    )
    lines.append(
        "  - Masking mechanism: for each lab, we zero value, zero mask, and set time-since-last to max (1.0 normalized); baseline CE is computed on the full feature set for that patient/time window."
    )
    lines.append(
        "  - Relevance scores are non-negative for ranking (clipped at 0) to keep NDCG well-defined in [0,1]."
    )
    lines.append(
        "  - Train/val/test splits are stratified by label; normalization stats computed on the training split only to avoid leakage."
    )
    lines.append("")
    lines.append("Config snapshot:")
    lines.append(json.dumps(config, indent=2, default=str))
    summary_text = "\n".join(lines)
    (outdir / "summary.txt").write_text(summary_text)
    print(summary_text)
    return summary_text


# -------------------------- Main Pipeline -------------------------- #


def main():
    config = CONFIG
    prefer_cuda = (not config.get("force_cpu", False)) and torch.cuda.is_available()
    run_hash = hash_config(config)
    metrics_path = config["results_dir"] / "tables" / "metrics.json"
    if config.get("reuse_artifacts", True) and metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text())
            if cached.get("config_hash") == run_hash:
                print(f"Reuse enabled and matching artifacts found at {metrics_path}. Skipping training.")
                print("Summary:\n", (config["results_dir"] / "summary.txt").read_text())
                return
        except Exception:
            pass
    use_cuda = set_seed(config["seed"], prefer_cuda=prefer_cuda)
    ensure_dirs(config)
    maybe_invalidate(config)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    values, masks, deltas, labels = build_datasets(config)

    train_ds, val_ds, test_ds, stats, horizon_steps = build_loaders(values, masks, deltas, labels, config)
    print(
        f"Class balance - train: {train_ds.labels.mean():.3f}, "
        f"val: {val_ds.labels.mean():.3f}, "
        f"test: {test_ds.labels.mean():.3f}"
    )
    model = LabTransformer(input_dim=len(VARIABLES) * 3, cfg=config["model"]).to(device)

    # Dynamic batch sizing
    bs = tune_batch_size(model, train_ds, config, device)
    config["train"]["batch_size"] = bs
    print(f"Using batch size: {bs}")

    # Loss with class imbalance correction shared across train/val/test
    pos_frac = train_ds.labels.mean()
    pos_frac_clamped = float(np.clip(pos_frac, 1e-3, 1 - 1e-3))
    pos_weight = torch.tensor((1 - pos_frac_clamped) / pos_frac_clamped, device=device)

    int_frac = float(np.mean(train_ds.intervention_labels)) if hasattr(train_ds, "intervention_labels") else 0.5
    int_frac_clamped = float(np.clip(int_frac, 1e-3, 1 - 1e-3))
    int_weight = torch.tensor((1 - int_frac_clamped) / int_frac_clamped, device=device)

    criteria = {
        "mortality": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "intervention": nn.BCEWithLogitsLoss(pos_weight=int_weight),
    }
    loss_weights = config.get("loss_weights", {"mortality": 1.0, "intervention": 0.5})

    try:
        history, best_val = train_model(
            model, train_ds, val_ds, criteria, loss_weights, config, device
        )
        plot_losses(history, config["results_dir"])

        # Load best and evaluate test with the same logits-aware criterion
        model.load_state_dict(
            torch.load(config["results_dir"] / "best_model.pt", map_location=device)
        )
        _, test_auc, test_probs, test_labels = evaluate_model(
            model, test_ds, criteria, device, return_preds=True
        )
    except RuntimeError as e:
        if "device-side assert triggered" in str(e).lower() and device.type == "cuda":
            print(
                "Detected CUDA device-side assert. Falling back to CPU and retrying training/eval."
            )
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            set_seed(config["seed"], prefer_cuda=False)
            model = LabTransformer(input_dim=len(VARIABLES) * 3, cfg=config["model"]).to(
                device
            )
            bs = min(config["train"]["batch_size"], config["train"]["initial_batch_size"])
            config["train"]["batch_size"] = bs
            pos_weight = torch.tensor((1 - pos_frac_clamped) / pos_frac_clamped, device=device)
            int_weight = torch.tensor((1 - int_frac_clamped) / int_frac_clamped, device=device)
            criteria = {
                "mortality": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "intervention": nn.BCEWithLogitsLoss(pos_weight=int_weight),
            }
            history, best_val = train_model(
                model, train_ds, val_ds, criteria, loss_weights, config, device
            )
            plot_losses(history, config["results_dir"])
            model.load_state_dict(
                torch.load(config["results_dir"] / "best_model.pt", map_location=device)
            )
            _, test_auc, test_probs, test_labels = evaluate_model(
                model, test_ds, criteria, device, return_preds=True
            )
        else:
            raise
    print(f"Test AUROC: {test_auc:.3f}")

    # Utility-based ranking
    val_utils, val_patient_utils, _ = compute_utilities(model, val_ds, device, target="mortality")
    val_propensity = compute_lab_propensity(val_ds.masks)
    global_util_mean_val = aggregate_global_utils(val_patient_utils, propensities=val_propensity)

    test_utils, patient_utils, patient_probs = compute_utilities(model, test_ds, device, target="mortality")
    # Weighted global util on test for reporting; rankings use validation-only scores
    test_propensity = compute_lab_propensity(test_ds.masks)
    global_util_mean_test = aggregate_global_utils(patient_utils, propensities=test_propensity)
    # Treatment-conditional VOI (intervention head)
    int_utils, int_patient_utils, int_probs = compute_utilities(model, test_ds, device, target="intervention")
    global_util_intervention = aggregate_global_utils(int_patient_utils, propensities=test_propensity)
    global_util_intervention = aggregate_global_utils(int_patient_utils, propensities=test_propensity)

    ranking_metrics = {
        "oracle": evaluate_ranking(patient_utils, "oracle"),
        "global": evaluate_ranking(patient_utils, "global", global_scores=global_util_mean_val),
        "random": evaluate_ranking(patient_utils, "random"),
    }

    budget_metrics = evaluate_budgeted(
        patient_utils,
        global_util_mean_val,
        budgets=(1, 3, 5),
        costs=config.get("lab_costs"),
        lambda_cost=config.get("lambda_cost", 0.0),
    )

    # Policy simulation: CE gain of taking top-k (global strategy) minus random
    policy_gains = {}
    for k in [1, 3, 5]:
        gain_topk = ranking_metrics["global"][f"top{k}"]
        gain_rand = ranking_metrics["random"][f"top{k}"]
        policy_gains[k] = gain_topk - gain_rand

    lead_time_stats = compute_lead_time_gain(
        model, test_ds, global_util_mean_val, config, device, risk_threshold=0.5
    )

    plot_top_tests(global_util_mean_test, config["results_dir"])
    plot_ranking_metrics(ranking_metrics, config["results_dir"])

    # Redundancy audit: how many ordered labs have non-positive utility
    red_counts, red_per_lab = redundancy_audit(patient_utils, test_ds.masks, threshold=0.0)
    (config["results_dir"] / "tables" / "redundancy.json").write_text(
        json.dumps({"counts": red_counts, "per_lab": red_per_lab}, indent=2)
    )

    # Temporal case visualizations (one survivor, one non-survivor)
    try:
        top5 = sorted(global_util_mean_test, key=global_util_mean_test.get, reverse=True)[:5]
        figs_dir = config["results_dir"] / "figs"
        # pick cases
        survivor_idx = int(np.where(test_ds.labels == 0)[0][0])
        nonsurv_idx = int(np.where(test_ds.labels == 1)[0][0])
        for idx, tag in [(survivor_idx, "survivor"), (nonsurv_idx, "nonsurvivor")]:
            sample_x, sample_mask, _, _ = test_ds[idx]
            traj = risk_trajectory(model, sample_x.clone(), device, step_stride=4)
            df_traj = pd.DataFrame(traj, columns=["t", "p"])
            df_utils = temporal_utility_heatmap(model, sample_x.clone(), device, top5, step_stride=4)
            plt.figure(figsize=(5, 3))
            plt.plot(df_traj["t"], df_traj["p"])
            plt.xlabel("Time index")
            plt.ylabel("Mortality risk")
            plt.title(f"Risk trajectory ({tag})")
            plt.tight_layout()
            plt.savefig(figs_dir / f"risk_traj_{tag}.png", dpi=200)
            plt.close()
            if not df_utils.empty:
                piv = df_utils.pivot_table(index="lab", columns="t", values="utility", aggfunc=np.mean)
                plt.figure(figsize=(6, 3))
                sns.heatmap(piv, cmap="coolwarm", center=0)
                plt.xlabel("Time index")
                plt.ylabel("Lab")
                plt.title(f"Utility heatmap ({tag})")
                plt.tight_layout()
                plt.savefig(figs_dir / f"utility_heatmap_{tag}.png", dpi=200)
                plt.close()
    except Exception as e:
        print(f"Case visualization skipped: {e}")

    # Tabular baselines
    agg = aggregate_static_features(values, masks)
    X_train, X_test, y_train, y_test = train_test_split(
        agg, labels, test_size=0.2, random_state=config["seed"], stratify=labels
    )
    baseline_res = run_tabular_baselines(X_train, X_test, y_train, y_test, config)
    baseline_ci = {}
    for name, auc in baseline_res["results"].items():
        # crude bootstrap CIs on tabular test split
        probs = None
        if name == "logreg_auc":
            # recompute prob scores for CI
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_train)
            probs = LogisticRegression(max_iter=500, class_weight="balanced").fit(
                scaler.transform(X_train), y_train
            ).predict_proba(scaler.transform(X_test))[:, 1]
        elif name == "gbdt_auc":
            probs = GradientBoostingClassifier().fit(X_train, y_train).predict_proba(
                X_test
            )[:, 1]
        elif name == "rf_auc":
            from sklearn.ensemble import RandomForestClassifier

            probs = RandomForestClassifier(
                n_estimators=400, class_weight="balanced", n_jobs=-1
            ).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        elif name == "mlp_auc":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_train)
            probs = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                max_iter=50,
                alpha=1e-4,
                random_state=config["seed"],
            ).fit(scaler.transform(X_train), y_train).predict_proba(
                scaler.transform(X_test)
            )[:, 1]
        if probs is not None:
            baseline_ci[name] = bootstrap_auc_ci(probs, y_test)

    # Build summary and save metrics
    test_ci = bootstrap_auc_ci(test_probs, test_labels)

    summary_text = build_summary(
        val_auc=best_val,
        test_auc=test_auc,
        ranking_metrics=ranking_metrics,
        baseline_res=baseline_res,
        global_util=global_util_mean_test,
        intervention_util=global_util_intervention,
        policy_gains=policy_gains,
        budget_metrics=budget_metrics,
        redundancy_counts=red_counts,
        test_ci=test_ci,
        baseline_ci=baseline_ci,
        lead_time_stats=lead_time_stats,
        config=config,
        outdir=config["results_dir"],
    )

    metrics_path = config["results_dir"] / "tables" / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "config_hash": run_hash,
                "history": history,
                "ranking": ranking_metrics,
                "global_util_val": global_util_mean_val,
                "global_util_test": global_util_mean_test,
                "global_util_intervention": global_util_intervention,
                "budgets": budget_metrics,
                "redundancy": {"counts": red_counts, "per_lab": red_per_lab},
                "baselines": baseline_res,
                "lead_time": lead_time_stats,
            },
            indent=2,
        )
    )

    zip_target = config["results_dir"]
    print("\nArtifacts saved under:", zip_target)
    print(
        "\nTo download everything from Colab run the following cell:\n"
        f"!zip -r /content/file.zip {zip_target}\n"
        "from google.colab import files\n"
        'files.download("/content/file.zip")'
    )


if __name__ == "__main__":
    main()
