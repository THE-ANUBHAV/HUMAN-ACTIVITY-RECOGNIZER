"""
==========================================================================
  Human Activity Recognition — LSTM & GRU Model Training
  -------------------------------------------------------
  Uses raw inertial signals (9 channels × 128 time-steps) from the
  UCI HAR Dataset to train LSTM and GRU classifiers with PyTorch.
==========================================================================
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")

# ──────────────────────────── CONFIG ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "UCI HAR Dataset", "UCI HAR Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVITY_LABELS = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING"
}
ACTIVITY_NAMES = [ACTIVITY_LABELS[i] for i in range(1, 7)]

N_TIMESTEPS = 128
N_CHANNELS = 9
N_CLASSES = 6
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────── DATA LOADING ───────────────────────────────
INERTIAL_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

def load_signals(subset: str) -> np.ndarray:
    """Load 9 inertial signal files → (n_samples, 128, 9)"""
    signals = []
    for signal in INERTIAL_SIGNALS:
        fname = f"{signal}_{subset}.txt"
        fpath = os.path.join(DATA_DIR, subset, "Inertial Signals", fname)
        data = pd.read_csv(fpath, delim_whitespace=True, header=None).values
        signals.append(data)
    # Stack along channel axis → (n, 128, 9)
    return np.transpose(np.array(signals), (1, 2, 0)).astype(np.float32)

def load_labels(subset: str) -> np.ndarray:
    fpath = os.path.join(DATA_DIR, subset, f"y_{subset}.txt")
    return pd.read_csv(fpath, header=None).values.flatten() - 1   # 0-indexed

def load_subjects(subset: str) -> np.ndarray:
    fpath = os.path.join(DATA_DIR, subset, f"subject_{subset}.txt")
    return pd.read_csv(fpath, header=None).values.flatten()

def load_features(subset: str) -> np.ndarray:
    fpath = os.path.join(DATA_DIR, subset, f"X_{subset}.txt")
    return pd.read_csv(fpath, delim_whitespace=True, header=None).values.astype(np.float32)

print("[*] Loading data ...")
X_train_raw = load_signals("train")
X_test_raw  = load_signals("test")
y_train     = load_labels("train")
y_test      = load_labels("test")
subjects_train = load_subjects("train")
subjects_test  = load_subjects("test")

# Also load the 561-feature vectors for EDA in the dashboard
X_train_feat = load_features("train")
X_test_feat  = load_features("test")

print(f"    Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")
print(f"    Labels -- Train: {y_train.shape}  |  Test: {y_test.shape}")

# ─────────────── SAVE DATA STATS FOR DASHBOARD ───────────────────
data_stats = {
    "train_samples": int(X_train_raw.shape[0]),
    "test_samples": int(X_test_raw.shape[0]),
    "n_timesteps": N_TIMESTEPS,
    "n_channels": N_CHANNELS,
    "n_classes": N_CLASSES,
    "n_features": int(X_train_feat.shape[1]),
    "activity_labels": ACTIVITY_NAMES,
    "train_class_dist": {ACTIVITY_NAMES[i]: int(c) for i, c in enumerate(np.bincount(y_train))},
    "test_class_dist": {ACTIVITY_NAMES[i]: int(c) for i, c in enumerate(np.bincount(y_test))},
    "train_subjects": sorted(list(set(subjects_train.tolist()))),
    "test_subjects": sorted(list(set(subjects_test.tolist()))),
}

# Save sample raw signals for dashboard visualisation (first sample per activity)
sample_signals = {}
for act_idx in range(N_CLASSES):
    idx = np.where(y_test == act_idx)[0][0]
    sample_signals[ACTIVITY_NAMES[act_idx]] = {
        ch: X_test_raw[idx, :, ch_i].tolist()
        for ch_i, ch in enumerate(INERTIAL_SIGNALS)
    }
data_stats["sample_signals"] = sample_signals

with open(os.path.join(OUTPUT_DIR, "data_stats.json"), "w") as f:
    json.dump(data_stats, f, indent=2)
print("[OK] Saved data_stats.json")

# ──────────────── PYTORCH DATASETS & LOADERS ─────────────────────
train_ds = TensorDataset(
    torch.from_numpy(X_train_raw), torch.from_numpy(y_train).long()
)
test_ds = TensorDataset(
    torch.from_numpy(X_test_raw), torch.from_numpy(y_test).long()
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────── MODEL DEFINITIONS ──────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_CHANNELS, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT, bidirectional=False
        )
        self.bn = nn.BatchNorm1d(HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, N_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H)
        out = out[:, -1, :]             # last time-step
        out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=N_CHANNELS, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, batch_first=True,
            dropout=DROPOUT, bidirectional=False
        )
        self.bn = nn.BatchNorm1d(HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, N_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out)


# ──────────────────── TRAINING LOOP ──────────────────────────────
def train_model(model, model_name: str):
    print(f"\n{'='*60}")
    print(f"  [>] Training {model_name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": [], "learning_rates": []}

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # ── Evaluate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(round(train_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))
        history["epoch_time"].append(round(epoch_time, 2))
        history["learning_rates"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_best.pth"))

        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"Time: {epoch_time:.1f}s")

    # ── Final Evaluation ──
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{model_name}_best.pth"),
                                     weights_only=True))
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=ACTIVITY_NAMES, output_dict=True)
    acc = accuracy_score(all_true, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_preds, average="weighted")

    results = {
        "model_name": model_name,
        "best_val_accuracy": round(best_val_acc, 5),
        "test_accuracy": round(acc, 5),
        "weighted_precision": round(prec, 5),
        "weighted_recall": round(rec, 5),
        "weighted_f1": round(f1, 5),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "history": history,
        "predictions": all_preds.tolist(),
        "true_labels": all_true.tolist(),
        "prediction_probabilities": all_probs.tolist(),
        "hyperparameters": {
            "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS,
            "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
        }
    }
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [OK] {model_name} -- Test Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(f"       Saved: {model_name}_results.json + {model_name}_best.pth")
    return results


# ──────────────── TRAIN BOTH MODELS ──────────────────────────────
if __name__ == "__main__":
    lstm_results = train_model(LSTMModel(), "LSTM")
    gru_results  = train_model(GRUModel(),  "GRU")

    # Save comparison summary
    comparison = {
        "LSTM": {
            "test_accuracy": lstm_results["test_accuracy"],
            "weighted_f1": lstm_results["weighted_f1"],
            "weighted_precision": lstm_results["weighted_precision"],
            "weighted_recall": lstm_results["weighted_recall"],
        },
        "GRU": {
            "test_accuracy": gru_results["test_accuracy"],
            "weighted_f1": gru_results["weighted_f1"],
            "weighted_precision": gru_results["weighted_precision"],
            "weighted_recall": gru_results["weighted_recall"],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'='*60}")
    print("  [DONE] Training Complete!")
    print(f"  LSTM Accuracy: {lstm_results['test_accuracy']:.4f}")
    print(f"  GRU  Accuracy: {gru_results['test_accuracy']:.4f}")
    print(f"{'='*60}")
