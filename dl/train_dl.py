#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dl.py

Train RNN/LSTM/GRU models for hurricane impact prediction using JSONL from dl_prep_city_split.py
(fields: X, y, meta.city, meta.top_category). Produces rounded integer predictions and extensive reports.

Outputs per model in <output-dir>/<model>/:
  - best_<model>.pth
  - <model>_predictions.csv                        # placekey, city, category, y_true, y_pred (ints)
  - <model>_category_metrics.csv                   # per-category metrics + counts
  - <model>_city_metrics.csv                       # per-city metrics + counts
  - <model>_summary.json                           # overall metrics + top/bottom categories/cities
  - <model>_dataset_summary.json                   # dataset counts (samples, placekeys)

Also writes, at <output-dir>/ :
  - all_models_city_metrics.csv                    # combined per-city metrics (long)
  - all_models_city_metrics_wide.csv               # combined per-city metrics (wide, rounded to 2 decimals)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ===================== ARGS =====================

def parse_args():
    p = argparse.ArgumentParser(description="Train DL models for hurricane impact prediction")
    # data
    p.add_argument("--train-file", required=True, help="Path to train JSONL")
    p.add_argument("--test-file",  required=True, help="Path to test JSONL")
    p.add_argument("--task", choices=["d14", "d21"], default="d14", help="Task type (for logging only)")
    # training
    p.add_argument("--models", nargs="+", default=["LSTM"], choices=["RNN", "LSTM", "GRU"])
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-split", type=float, default=0.2, help="Validation split from TRAIN samples")
    p.add_argument("--normalize", action="store_true", help="Normalize input sequences with StandardScaler")
    # output
    p.add_argument("--output-dir", default="results", help="Root output directory")
    return p.parse_args()


# ===================== DATA =====================

def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class HurricaneDataset(Dataset):
    """
    Consumes items like:
      {"placekey": "...", "X":[...], "y": 7, "meta":{"city":"...", "top_category":"..."}}
    """
    def __init__(self, data, encoders=None, scaler=None, normalize=False):
        self.seq, self.tgt, self.cat_lbl, self.city_lbl, self.placekeys = [], [], [], [], []
        for item in data:
            X = item.get("X")
            y = item.get("y")
            meta = item.get("meta", {}) or {}
            if not isinstance(X, list) or y is None:
                continue
            self.seq.append(X)
            self.tgt.append(int(round(y)))  # ensure integer true visits
            self.cat_lbl.append(meta.get("top_category", "Unknown"))
            self.city_lbl.append(meta.get("city", "Unknown"))
            self.placekeys.append(item.get("placekey", ""))

        self.seq = np.asarray(self.seq, dtype=np.float32)          # [N, T]
        self.tgt = np.asarray(self.tgt, dtype=np.int32)            # [N]

        # normalization (optional)
        if normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                flat = self.seq.reshape(-1, 1)
                self.scaler.fit(flat)
                self.seq = self.scaler.transform(flat).reshape(self.seq.shape)
            else:
                self.scaler = scaler
                flat = self.seq.reshape(-1, 1)
                self.seq = self.scaler.transform(flat).reshape(self.seq.shape)
        else:
            self.scaler = None

        # encoders
        if encoders is None:
            self.encoders = {
                "category": LabelEncoder().fit(self.cat_lbl),
                "city": LabelEncoder().fit(self.city_lbl)
            }
            self.cat_enc = self.encoders["category"].transform(self.cat_lbl)
            self.city_enc = self.encoders["city"].transform(self.city_lbl)
        else:
            self.encoders = encoders

            def safe_transform(encoder, values):
                known = set(encoder.classes_.tolist())
                out = []
                for v in values:
                    out.append(encoder.transform([v])[0] if v in known else 0)
                return np.asarray(out)

            self.cat_enc = safe_transform(self.encoders["category"], self.cat_lbl)
            self.city_enc = safe_transform(self.encoders["city"], self.city_lbl)

        # tensors
        self.seq = torch.FloatTensor(self.seq)            # [N, T]
        self.tgt = torch.FloatTensor(self.tgt)            # keep float for MSE loss
        self.cat_enc = torch.LongTensor(self.cat_enc)
        self.city_enc = torch.LongTensor(self.city_enc)

    def __len__(self): return len(self.seq)

    def __getitem__(self, i):
        return {
            "sequence": self.seq[i],
            "target": self.tgt[i],
            "category": self.cat_enc[i],
            "city": self.city_enc[i],
            "placekey": self.placekeys[i]
        }


# ===================== MODEL =====================

class HurricaneRNN(nn.Module):
    def __init__(self, model_type, hidden_size, num_layers, num_categories, num_cities, dropout=0.2):
        super().__init__()
        if model_type == "LSTM":
            self.rnn = nn.LSTM(1, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(1, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        else:
            self.rnn = nn.RNN(1, hidden_size, num_layers, nonlinearity="tanh",
                              dropout=dropout if num_layers > 1 else 0, batch_first=True)

        embed_dim = 16
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        self.city_embed = nn.Embedding(num_cities, embed_dim)

        feat = hidden_size + 2 * embed_dim
        self.head = nn.Sequential(
            nn.Linear(feat, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(32, 1)
        )

    def forward(self, sequence, category, city):
        x = sequence.unsqueeze(-1)   # [B, T, 1]
        rnn_out, _ = self.rnn(x)     # [B, T, H]
        h = rnn_out[:, -1, :]        # [B, H]
        cat_e = self.category_embed(category)
        city_e = self.city_embed(city)
        return self.head(torch.cat([h, cat_e, city_e], dim=1)).squeeze(-1)


# ===================== METRICS =====================

def mae(y_true, y_pred):   return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred):  return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def smape_mean_median(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    vals = np.zeros_like(y_true, dtype=float)
    vals[mask] = 200.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return float(np.mean(vals[mask])) if np.any(mask) else 0.0, float(np.median(vals[mask])) if np.any(mask) else 0.0
def rmsle(y_true, y_pred):
    yt = np.clip(y_true, 0, None); yp = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt))**2)))
def exact_match_acc(y_true, y_pred): return float(np.mean((y_true == y_pred).astype(float)))


# ===================== TRAIN / EVAL =====================

def train_one(model, train_loader, val_loader, args, save_dir, tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.7)

    best_val = float("inf")
    wait = 0

    for epoch in range(args.epochs):
        # train
        model.train()
        tr = 0.0
        for b in train_loader:
            seq, tgt = b["sequence"].to(device), b["target"].to(device)
            cat, city = b["category"].to(device), b["city"].to(device)
            opt.zero_grad()
            pred = model(seq, cat, city)
            loss = crit(pred, tgt)
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))

        # val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                seq, tgt = b["sequence"].to(device), b["target"].to(device)
                cat, city = b["category"].to(device), b["city"].to(device)
                pred = model(seq, cat, city)
                vl += crit(pred, tgt).item()
        vl /= max(1, len(val_loader))
        sched.step(vl)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[{tag}] Epoch {epoch+1:3d}/{args.epochs}  Train={tr:.4f}  Val={vl:.4f}")

        if vl < best_val:
            best_val, wait = vl, 0
            torch.save(model.state_dict(), save_dir / f"best_{tag}.pth")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{tag}] Early stopping at epoch {epoch+1}")
                break

    return best_val


def predict_and_report(model, test_loader, test_dataset, save_dir, tag):
    """Return predictions DataFrame and write per-category/city metrics & summary."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    preds, tgts, pks, cats, cities = [], [], [], [], []

    with torch.no_grad():
        for b in test_loader:
            seq, tgt = b["sequence"].to(device), b["target"].to(device)
            cat, city = b["category"].to(device), b["city"].to(device)
            out = model(seq, cat, city)
            out = torch.clamp(out, min=0).cpu().numpy()
            out = np.rint(out).astype(int)     # round to integer visits
            preds.extend(out)
            tgts.extend(b["target"].cpu().numpy().astype(int))
            pks.extend(b["placekey"])
            cats.extend([test_dataset.encoders["category"].inverse_transform([c.cpu().item()])[0] for c in cat])
            cities.extend([test_dataset.encoders["city"].inverse_transform([c.cpu().item()])[0] for c in city])

    df_pred = pd.DataFrame({
        "placekey": pks,
        "city": cities,
        "category": cats,
        "y_true": tgts,
        "y_pred": preds
    })
    df_pred.to_csv(save_dir / f"{tag}_predictions.csv", index=False)

    # ---- Per-category metrics
    cat_rows = []
    for cat in sorted(df_pred["category"].unique()):
        sub = df_pred[df_pred["category"] == cat]
        y_true = sub["y_true"].to_numpy(); y_pred = sub["y_pred"].to_numpy()
        m_mean, m_med = smape_mean_median(y_true, y_pred)
        cat_rows.append({
            "category": cat,
            "count": int(len(sub)),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "smape_mean": m_mean,
            "smape_median": m_med,
            "rmsle": rmsle(y_true, y_pred),
            "exact_match_acc": exact_match_acc(y_true, y_pred)
        })
    df_cat = pd.DataFrame(cat_rows).sort_values(["mae", "rmse", "rmsle"])
    df_cat.to_csv(save_dir / f"{tag}_category_metrics.csv", index=False)

    # ---- Per-city metrics
    city_rows = []
    for c in sorted(df_pred["city"].unique()):
        sub = df_pred[df_pred["city"] == c]
        y_true = sub["y_true"].to_numpy(); y_pred = sub["y_pred"].to_numpy()
        m_mean, m_med = smape_mean_median(y_true, y_pred)
        city_rows.append({
            "city": c,
            "count": int(len(sub)),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "smape_mean": m_mean,
            "smape_median": m_med,
            "rmsle": rmsle(y_true, y_pred),
            "exact_match_acc": exact_match_acc(y_true, y_pred)
        })
    df_city = pd.DataFrame(city_rows).sort_values(["mae", "rmse", "rmsle"])
    df_city.to_csv(save_dir / f"{tag}_city_metrics.csv", index=False)

    # ---- Summary JSON (overall + highlights)
    overall_y_true = df_pred["y_true"].to_numpy()
    overall_y_pred = df_pred["y_pred"].to_numpy()
    o_mean, o_med = smape_mean_median(overall_y_true, overall_y_pred)
    summary = {
        "samples_test": int(len(df_pred)),
        "overall": {
            "mae": round(mae(overall_y_true, overall_y_pred), 2),
            "rmse": round(rmse(overall_y_true, overall_y_pred), 2),
            "smape_mean": round(o_mean, 2),
            "smape_median": round(o_med, 2),
            "rmsle": round(rmsle(overall_y_true, overall_y_pred), 2),
            "exact_match_acc": round(exact_match_acc(overall_y_true, overall_y_pred), 4)
        },
        "best_categories_by_mae": df_cat.nsmallest(5, "mae")[["category", "count", "mae"]].to_dict(orient="records"),
        "worst_categories_by_mae": df_cat.nlargest(5, "mae")[["category", "count", "mae"]].to_dict(orient="records"),
        "best_cities_by_mae": df_city.nsmallest(5, "mae")[["city", "count", "mae"]].to_dict(orient="records"),
        "worst_cities_by_mae": df_city.nlargest(5, "mae")[["city", "count", "mae"]].to_dict(orient="records"),
    }
    with open(save_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console highlights (2 decimals)
    print("\nTop 5 categories (by MAE):")
    print(df_cat.nsmallest(5, "mae")[["category", "count", "mae", "rmse", "exact_match_acc"]]
          .to_string(index=False, float_format="%.2f"))

    print("\nWorst 5 categories (by MAE):")
    print(df_cat.nlargest(5, "mae")[["category", "count", "mae", "rmse", "exact_match_acc"]]
          .to_string(index=False, float_format="%.2f"))

    return df_pred, df_cat, df_city


# ===================== MAIN =====================

def main():
    args = parse_args()
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Load data
    train_raw = load_jsonl(Path(args.train_file))
    test_raw  = load_jsonl(Path(args.test_file))

    # Report placekey counts in splits (based on provided files)
    pk_train = {it.get("placekey") for it in train_raw if it.get("placekey")}
    pk_test  = {it.get("placekey") for it in test_raw if it.get("placekey")}
    print(f"Placekeys: train={len(pk_train)}  test={len(pk_test)}")

    # Split train into train/val
    train_split, val_split = train_test_split(train_raw, test_size=args.val_split, random_state=42)
    print(f"Samples: train={len(train_split)}  val={len(val_split)}  test={len(test_raw)}")

    # Build datasets / loaders
    ds_train = HurricaneDataset(train_split, normalize=args.normalize)
    ds_val   = HurricaneDataset(val_split, encoders=ds_train.encoders, scaler=ds_train.scaler, normalize=args.normalize)
    ds_test  = HurricaneDataset(test_raw,  encoders=ds_train.encoders, scaler=ds_train.scaler, normalize=args.normalize)

    # Unique placekeys actually usable (complete samples)
    pk_train_used = len({pk for pk in ds_train.placekeys})
    pk_val_used   = len({pk for pk in ds_val.placekeys})
    pk_test_used  = len({pk for pk in ds_test.placekeys})
    print(f"Usable placekeys: train={pk_train_used}  val={pk_val_used}  test={pk_test_used}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    num_categories = len(ds_train.encoders["category"].classes_)
    num_cities     = len(ds_train.encoders["city"].classes_)
    print(f"Embeddings: categories={num_categories}  cities={num_cities}")

    # Train + report for each model
    for model_type in args.models:
        print("\n" + "="*60)
        print(f"TRAINING {model_type}  (task={args.task})")
        print("="*60)
        save_dir = out_root / model_type.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        model = HurricaneRNN(
            model_type=model_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_categories=num_categories,
            num_cities=num_cities,
            dropout=args.dropout
        )

        # Train
        best_val = train_one(model, dl_train, dl_val, args, save_dir, tag=model_type.lower())
        print(f"[{model_type}] Best Val Loss: {best_val:.4f}")

        # Load best and evaluate
        model.load_state_dict(torch.load(save_dir / f"best_{model_type.lower()}.pth"))
        _df_pred, _df_cat, _df_city = predict_and_report(model, dl_test, ds_test, save_dir, tag=model_type.lower())

        # Also write a dataset summary per model
        ds_summary = {
            "task": args.task,
            "train_file": str(Path(args.train_file).resolve()),
            "test_file":  str(Path(args.test_file).resolve()),
            "counts": {
                "samples": {
                    "train": len(ds_train),
                    "val": len(ds_val),
                    "test": len(ds_test)
                },
                "placekeys_provided": {
                    "train": len(pk_train),
                    "test": len(pk_test)
                },
                "placekeys_usable": {
                    "train": pk_train_used,
                    "val": pk_val_used,
                    "test": pk_test_used
                }
            }
        }
        with open(save_dir / f"{model_type.lower()}_dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(ds_summary, f, indent=2)

        print(f"[{model_type}] Reports saved in: {save_dir.resolve()}")

    # --- COMBINE CITY METRICS ACROSS MODELS (Option B) ---
    def _round_cols(df: pd.DataFrame, cols, nd=2):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(nd)
        return df

    root = out_root
    rows = []
    for model_dir in root.iterdir():
        if model_dir.is_dir():
            name = model_dir.name.lower()
            city_csv = model_dir / f"{name}_city_metrics.csv"
            if city_csv.exists():
                df = pd.read_csv(city_csv)
                df.insert(0, "model", name.upper())
                rows.append(df)

    if rows:
        long_df = pd.concat(rows, ignore_index=True)

        # keep + round to 2 decimals
        metric_cols = ["mae", "rmse", "smape_mean", "smape_median", "rmsle"]
        keep_cols = ["model", "city", "count"] + metric_cols
        long_df = long_df[[c for c in keep_cols if c in long_df.columns]]
        long_df = _round_cols(long_df, metric_cols, nd=2)

        # write long
        long_out = root / "all_models_city_metrics.csv"
        long_df.to_csv(long_out, index=False)

        # wide pivot (one row per city, columns per metric__MODEL), rounded 2 decimals
        wide = long_df.pivot_table(index="city", columns="model", values=metric_cols)
        wide = wide.round(2)
        wide.columns = [f"{m}__{mdl}" for m, mdl in wide.columns]
        wide = wide.reset_index()
        wide_out = root / "all_models_city_metrics_wide.csv"
        wide.to_csv(wide_out, index=False)

        print("\nCombined city metrics written to:")
        print(f"  {long_out}")
        print(f"  {wide_out}")
    else:
        print("\n[warn] No per-model city metrics found to combine.")

    print("\nDone.")


if __name__ == "__main__":
    main()
