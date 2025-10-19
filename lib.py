from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


# =========================================================
# Model (from your lib.py) — kept as-is
# =========================================================
class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# =========================================================
# Scaler/Splitter (from your lib.py) — with scale_target option
# =========================================================
class DataScalerSplitter:
    def __init__(self, features, target, val_ratio=0.15, test_ratio=0.15, *, scale_target: bool = False):
        self.features = list(features)
        self.target = target
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

        self.scaler = StandardScaler()
        self.target_scaler: Optional[StandardScaler] = StandardScaler() if scale_target else None

        self.train = None
        self.val = None
        self.test = None
        self.val_index = None
        self.test_index = None

    def split_and_scale(self, df: pd.DataFrame):
        n = len(df)
        train_size = int((1 - self.val_ratio - self.test_ratio) * n)
        val_size = int(self.val_ratio * n)

        train = df.iloc[:train_size].copy()
        val   = df.iloc[train_size:train_size + val_size].copy()
        test  = df.iloc[train_size + val_size:].copy()

        self.val_index = min(val.index, default=None)
        self.test_index = min(test.index, default=None)

        # Feature scaling
        self.scaler.fit(train[self.features])
        train[self.features] = self.scaler.transform(train[self.features])
        val[self.features]   = self.scaler.transform(val[self.features])
        test[self.features]  = self.scaler.transform(test[self.features])

        # Optional target scaling
        if self.target_scaler is not None:
            self.target_scaler.fit(train[[self.target]])
            train[self.target] = self.target_scaler.transform(train[[self.target]])
            val[self.target]   = self.target_scaler.transform(val[[self.target]])
            test[self.target]  = self.target_scaler.transform(test[[self.target]])

        self.train, self.val, self.test = train, val, test
        return (
            train[self.features].values, train[self.target].values,
            val[self.features].values,   val[self.target].values,
            test[self.features].values,  test[self.target].values
        )

    # Fit on train; transform arrays (used by the trainer)
    def fit_transform(self, X, y):
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        if self.target_scaler is not None:
            ys = self.target_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
        else:
            ys = np.asarray(y).ravel()
        return Xs.astype(np.float32), ys.astype(np.float32)

    def transform(self, X, y):
        Xs = self.scaler.transform(X)
        if self.target_scaler is not None:
            ys = self.target_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()
        else:
            ys = np.asarray(y).ravel()
        return Xs.astype(np.float32), ys.astype(np.float32)

    def inverse_y(self, y_scaled):
        if self.target_scaler is None:
            return np.asarray(y_scaled).ravel()
        return self.target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

class Metrics:
    """
    Comprehensive regression/forecast metrics (no MASE).
    Includes:
      - ME, MAE, RMSE, MPE, MAPE, RMSLE, sMAPE, MBD(=ME)
      - Theil's U, ACF1 of residuals
      - R², Adjusted R²
      - AIC, BIC (Gaussian assumption)
      - Time_seconds, Param_count, n_obs
    """

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(float).ravel()
        return np.asarray(x, dtype=float).ravel()

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _rmsle(y_true, y_pred):
        yt = np.maximum(Metrics._to_numpy(y_true), 0.0)
        yp = np.maximum(Metrics._to_numpy(y_pred), 0.0)
        return float(np.sqrt(mean_squared_error(np.log1p(yt), np.log1p(yp))))

    @staticmethod
    def _smape(y_true, y_pred, eps=1e-9):
        yt = Metrics._to_numpy(y_true)
        yp = Metrics._to_numpy(y_pred)
        denom = np.abs(yt) + np.abs(yp) + eps
        return float(np.mean(2.0 * np.abs(yp - yt) / denom) * 100.0)

    @staticmethod
    def _theils_u(y_true, y_pred, y_naive_pred=None):
        yt = Metrics._to_numpy(y_true)
        yp = Metrics._to_numpy(y_pred)
        if y_naive_pred is None:
            if len(yt) < 2:
                return np.nan
            actual = yt[1:]
            pred_model = yp[1:]
            pred_naive = yt[:-1]
        else:
            yn = Metrics._to_numpy(y_naive_pred)
            m = min(len(yt), len(yp), len(yn))
            actual, pred_model, pred_naive = yt[:m], yp[:m], yn[:m]
        rmse_m = Metrics._rmse(actual, pred_model)
        rmse_n = Metrics._rmse(actual, pred_naive)
        return float(rmse_m / rmse_n) if rmse_n > 0 else np.nan

    @staticmethod
    def _acf1_residuals(y_true, y_pred):
        res = Metrics._to_numpy(y_true) - Metrics._to_numpy(y_pred)
        if len(res) < 2:
            return np.nan
        r = res - np.mean(res)
        num = np.sum(r[1:] * r[:-1])
        den = np.sum(r**2)
        return float(num / den) if den > 0 else np.nan

    @staticmethod
    def _count_params(model):
        if model is None:
            return 0
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    @staticmethod
    def _adj_r2(r2, n, p):
        if p is None or n - p - 1 <= 0:
            return np.nan
        return float(1 - (1 - r2) * (n - 1) / (n - p - 1))

    def evaluate(
        self,
        y_true,
        y_pred,
        *,
        model=None,
        insample=None,            # ignored (kept for signature compatibility)
        y_naive_pred=None,
        elapsed_time_sec=None,
        n_features_for_adj_r2=None,
    ):
        yt = self._to_numpy(y_true)
        yp = self._to_numpy(y_pred)
        n = len(yt)
        eps = 1e-9

        ME = float(np.mean(yp - yt))
        MAE = float(mean_absolute_error(yt, yp))
        RMSE = self._rmse(yt, yp)
        MPE = float(np.mean((yp - yt) / (yt + eps)) * 100.0)
        MAPE = float(mean_absolute_percentage_error(yt + eps, yp + eps) * 100.0)
        RMSLE = self._rmsle(yt, yp)
        sMAPE = self._smape(yt, yp)
        MBD = ME
        Theils_U = self._theils_u(yt, yp, y_naive_pred=y_naive_pred)
        ACF1 = self._acf1_residuals(yt, yp)

        R2 = float(r2_score(yt, yp)) if np.var(yt) > 0 else np.nan
        Adj_R2 = self._adj_r2(R2, n, n_features_for_adj_r2)

        mse = float(mean_squared_error(yt, yp))
        k_params = self._count_params(model)
        if mse <= 0:
            AIC = BIC = np.nan
        else:
            logL = -0.5 * n * (np.log(2 * np.pi * mse) + 1.0)
            AIC = float(-2 * logL + 2 * k_params)
            BIC = float(-2 * logL + k_params * np.log(n))

        return {
            "ME": ME,
            "MAE": MAE,
            "RMSE": RMSE,
            "MPE (%)": MPE,
            "MAPE (%)": MAPE,
            "RMSLE": RMSLE,
            "sMAPE (%)": sMAPE,
            "MBD": MBD,
            "Theils_U": Theils_U,
            "ACF1_residuals": ACF1,
            "R2": R2,
            "Adj_R2": Adj_R2,
            "AIC": AIC,
            "BIC": BIC,
            "Time_seconds": float(elapsed_time_sec) if elapsed_time_sec is not None else np.nan,
            "Param_count": k_params,
            "n_obs": int(n),
        }
    

@dataclass
class GRUTrainerConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 250

    patience: int = 30
    min_delta: float = 1e-4
    clip_grad: Optional[float] = 1.0
    train_test_split: float = 0.15
    train_val_split: float = 0.15
    use_amp: bool = False
    seed: Optional[int] = 42
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    window: int = 30
    horizon: int = 1
    shuffle: bool = False

class GRUTrainer:
    def __init__(self, cfg: GRUTrainerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        self.scaler = torch.amp.GradScaler(enabled=cfg.use_amp)
        self.best_state = None
        self.history = {"train_loss": [], "val_loss": []}
        self.preprocessor = None

        if cfg.seed is not None:
            self._set_seed(cfg.seed)

    def build(self, input_dim: int):
        self.model = GRUNet(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.cfg.lr, 
                weight_decay=self.cfg.weight_decay
            )
        return self

    def make_sliding_windows(self, 
                X: np.ndarray,
                y: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(X) == len(y)
        N, _ = X.shape
        seq_X, seq_y = [], []
        last = N - self.cfg.window - (self.cfg.horizon - 1)
        for start in range(last):
            end = start + self.cfg.window
            seq_X.append(X[start:end, :])
            seq_y.append(y[end + self.cfg.horizon - 1])
        return np.asarray(seq_X, dtype=np.float32), np.asarray(seq_y, dtype=np.float32).reshape(-1, 1)

    def make_loader(
                self, 
                X: np.ndarray, 
                y: np.ndarray, 
                batch_size: int,
            ) -> DataLoader:
        
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        if yt.ndim == 1:
            yt = yt.unsqueeze(-1)
        ds = TensorDataset(xt, yt)
        return DataLoader(ds, batch_size=batch_size, shuffle=self.cfg.shuffle, drop_last=False)


    def fit(self, tr_X, tr_y, vl_X, vl_y, *, verbose=False) -> "GRUTrainer":

        tr_X, tr_y = self.make_sliding_windows(tr_X, tr_y)
        vl_X, vl_y = self.make_sliding_windows(vl_X, vl_y)

        tr_loader = self.make_loader(tr_X, tr_y, self.cfg.batch_size)
        vl_loader = self.make_loader(vl_X, vl_y, self.cfg.batch_size)

        if self.model is None:
            self.build(input_dim=tr_X.shape[-1])

        best_val = np.inf
        wait = 0

        for epoch in range(self.cfg.max_epochs):            
            
            self.model.train()
            total, n = 0.0, 0

            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                if self.cfg.use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.model(xb)
                        loss = self.criterion(preds, yb)
                    
                    self.scaler.scale(loss).backward()

                    if self.cfg.clip_grad is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb)
                    loss.backward()
                    if self.cfg.clip_grad is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
                    self.optimizer.step()

                bs = xb.size(0)
                total += float(loss.item()) * bs
                n += bs
            train_loss = total / max(n, 1)
            
            self.history["train_loss"].append(train_loss)

            val_loss = self._eval_loss(vl_loader)

            self.history["val_loss"].append(val_loss)

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

            
            if best_val - val_loss > self.cfg.min_delta:
                best_val = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}. Best Val: {best_val:.6f}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self

    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                bs = xb.size(0)
                total += float(loss.item()) * bs
                n += bs
        return total / max(n, 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_seq, _ = self.make_sliding_windows(X, np.zeros((len(X),)))
        loader = self.make_loader(X_seq, np.zeros((len(X_seq), 1)), batch_size=self.cfg.batch_size)
        preds_list = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                preds = self.model(xb)
                preds_list.append(preds.cpu().numpy())
        return np.vstack(preds_list)

    @staticmethod
    def _set_seed(seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class PlotResults:
    def __init__(self, val_index, test_index):
        self.val_index = val_index
        self.test_index = test_index

    
    def plot(self, df):

        df_train = df[df["Date"] < self.val_index]
        df_val   = df[(df["Date"] >= self.val_index) & (df["Date"] < self.test_index)]
        df_test  = df[df["Date"] >= self.test_index]

        plt.figure(figsize=(14, 7))

        plt.plot(df_train["Date"], df_train["Actual"], label="Actual (Train)", color="blue")
        plt.plot(df_train["Date"], df_train["Predicted"], label="Predicted (Train)", color="black")

        plt.plot(df_val["Date"], df_val["Actual"], label="Actual (Val)", color="green")
        plt.plot(df_val["Date"], df_val["Predicted"], label="Predicted (Val)", color="black")

        plt.plot(df_test["Date"], df_test["Actual"], label="Actual (Test)", color="red")
        plt.plot(df_test["Date"], df_test["Predicted"], label="Predicted (Test)", color="black")

        plt.axvline(x=self.val_index, color='green', linestyle='--', label='Validation Start')
        plt.axvline(x=self.test_index, color='red', linestyle='--', label='Test Start')

        plt.xlabel("Date")
        plt.ylabel("Treasury Yield")
        plt.title("Treasury Yield Prediction")
        
        plt.show()
