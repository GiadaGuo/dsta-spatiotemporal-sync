#!/usr/bin/env python3
"""Train a small temporal/spatial state autoencoder and compare to PCA.
Run from any working directory: python scripts/train_state_autoencoder.py
"""
from __future__ import annotations
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/city_state_embeddings.parquet'
OUTPUT = ROOT / 'data/processed/city_state_ae_embeddings.parquet'
MODELS = ROOT / 'outputs/models'
FIGURES = ROOT / 'outputs/figures'
REPORTS = ROOT / 'outputs/reports'
SEED = 42
LATENT_DIM = 8
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64

FEATURES = [
    'log_density', 'mean_speed_mps', 'speed_std_mps', 'vehicle_share',
    'delta_log_density', 'delta_mean_speed_mps', 'has_previous_window',
    'neighbor_density', 'neighbor_speed',
]

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    for p in (MODELS, FIGURES, REPORTS, OUTPUT.parent):
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()
    missing_cols = [c for c in FEATURES + ['window_id'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns: {missing_cols}. Regenerate build_state_embeddings.py output.')
    df = df.sort_values(['window_id', 'grid_id']).reset_index(drop=True)
    windows = sorted(df.window_id.dropna().unique())
    if len(windows) < 5:
        raise ValueError('Need at least 5 time windows for temporal train/validation split.')
    cut = max(1, min(len(windows)-1, int(len(windows)*0.7)))
    train_windows, val_windows = windows[:cut], windows[cut:]
    train_mask = df.window_id.isin(train_windows).to_numpy()
    val_mask = ~train_mask
    print('Train windows:', list(map(int, train_windows)))
    print('Validation windows:', list(map(int, val_windows)))
    print('Train/validation rows:', int(train_mask.sum()), int(val_mask.sum()))

    raw = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    missing_flags = raw.isna().astype(np.float32).to_numpy()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    imputed_train = imputer.fit_transform(raw.iloc[np.flatnonzero(train_mask)])
    imputed_all = imputer.transform(raw)
    x_unscaled = np.column_stack([imputed_all, missing_flags]).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_unscaled[train_mask])
    x = scaler.transform(x_unscaled).astype(np.float32)
    x = np.clip(x, -10.0, 10.0)  # robustify very rare extreme states
    input_dim = x.shape[1]
    if input_dim <= LATENT_DIM:
        raise ValueError('Input dimensionality must exceed bottleneck dimension.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, 'Input dimensions:', input_dim)
    model = Autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_mask])), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    train_tensor = torch.from_numpy(x[train_mask]).to(device)
    val_tensor = torch.from_numpy(x[val_mask]).to(device)
    best_state = None
    best_val = float('inf')
    wait = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(train_tensor)[0], train_tensor).item()
            val_loss = criterion(model(val_tensor)[0], val_tensor).item()
        history.append({'epoch': epoch, 'train_mse': train_loss, 'val_mse': val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch:3d} train={train_loss:.5f} val={val_loss:.5f}')
        if wait >= PATIENCE:
            print('Early stopping at epoch', epoch)
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon, z = model(torch.from_numpy(x).to(device))
        recon = recon.cpu().numpy()
        z = z.cpu().numpy()
    ae_mse = ((recon-x)**2).mean(axis=1)

    pca = PCA(n_components=LATENT_DIM, random_state=SEED)
    pca.fit(x[train_mask])
    pca_z = pca.transform(x)
    pca_recon = pca.inverse_transform(pca_z)
    pca_mse = ((pca_recon-x)**2).mean(axis=1)
    for j in range(LATENT_DIM):
        df[f'ae_embedding_{j}'] = z[:, j]
    df['ae_reconstruction_mse'] = ae_mse
    df['pca_reconstruction_mse'] = pca_mse
    df['ae_split'] = np.where(train_mask, 'train', 'validation')
    df.to_parquet(OUTPUT, index=False)

    comparison = pd.DataFrame([
        {'model': name, 'split': split, 'mse': float(scores[mask].mean()), 'n_states': int(mask.sum())}
        for name, scores in [('PCA', pca_mse), ('Autoencoder', ae_mse)]
        for split, mask in [('train', train_mask), ('validation', val_mask)]
    ])
    comparison.to_csv(REPORTS/'embedding_comparison.csv', index=False)
    pd.DataFrame(history).to_csv(REPORTS/'ae_training_history.csv', index=False)
    torch.save({'model_state_dict': best_state, 'input_dim': input_dim,
                'latent_dim': LATENT_DIM, 'features': FEATURES, 'seed': SEED,
                'train_windows': list(map(int, train_windows)),
                'val_windows': list(map(int, val_windows)),
                'imputer_statistics': imputer.statistics_.tolist(),
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()},
               MODELS/'state_autoencoder.pt')
    metadata = {'best_epoch': best_epoch, 'best_validation_mse': best_val,
                'device': str(device), 'train_rows': int(train_mask.sum()),
                'validation_rows': int(val_mask.sum()), 'input_dim': input_dim,
                'latent_dim': LATENT_DIM, 'pca_train_explained_variance': float(pca.explained_variance_ratio_.sum())}
    (REPORTS/'ae_training_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r['epoch'] for r in history], [r['train_mse'] for r in history], label='Train', color='#2563eb')
    ax.plot([r['epoch'] for r in history], [r['val_mse'] for r in history], label='Validation', color='#ea580c')
    ax.axvline(best_epoch, color='#7c3aed', linestyle='--', alpha=0.7, label=f'Best epoch {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='MSE (standardized feature space)', title='State Autoencoder Training')
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(FIGURES/'13_ae_training_curve.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = np.where(val_mask, '#ea580c', '#2563eb')
    ax.scatter(z[:, 0], z[:, 1], c=colors, s=16, alpha=0.65)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#2563eb', markersize=7, label='Train'),
                       Line2D([0],[0], marker='o', color='w', markerfacecolor='#ea580c', markersize=7, label='Validation')])
    ax.set(xlabel='AE latent dimension 1', ylabel='AE latent dimension 2', title='Spatiotemporal State AE Embedding')
    ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(FIGURES/'14_ae_embedding.png', dpi=220)
    plt.close(fig)
    print('\n=== TRAINING SUMMARY ===')
    print(json.dumps(metadata, indent=2))
    print('\n=== PCA VS AE (same split, same inputs) ===')
    print(comparison.to_string(index=False))
    print('Saved:', OUTPUT)
    print('Saved:', MODELS/'state_autoencoder.pt')
    print('Saved figures 13 and 14 in:', FIGURES)

if __name__ == '__main__':
    main()
