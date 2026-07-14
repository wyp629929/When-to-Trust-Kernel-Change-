"""
Regularized task-aligned diagnostic: test whether eigenvalue shrinkage
recovers positive cross-validated correlation.
Poly width=64, 5-fold CV, 5 seeds.
Truncation: keep top k components (by eigenvalue magnitude).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')

N, EPOCHS = 300, 500
WIDTH = 64

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, WIDTH, bias=False)
        self.fc2 = nn.Linear(WIDTH, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def run_fold(train_idx, test_idx, seed, keep_frac=0.5):
    rng = np.random.RandomState(42 + seed)
    Xall = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    yall = Xall[:, 0]**2 + Xall[:, 1]
    yall = (yall - yall.mean()) / yall.std()
    X_tr = Xall[train_idx]; y_tr = yall[train_idx]
    X_te = Xall[test_idx]; y_te = yall[test_idx]
    n_tr = len(train_idx)

    m = Net()
    torch.manual_seed(42 + seed)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/WIDTH))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * rng.randn(n_tr))

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / WIDTH
    evals0, evecs0 = np.linalg.eigh(K0)
    evals0 = evals0[::-1]; evecs0 = evecs0[:, ::-1]

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / WIDTH
    evalsT, _ = np.linalg.eigh(K1)
    evalsT = evalsT[::-1]

    # Unregularized Δalign
    proj = evecs0.T @ y_tr
    vi = (proj ** 2) / n_tr
    dalign_raw = np.sum(vi * (evalsT - evals0)) / (np.sum(vi * evals0) + 1e-12)

    # Regularized: truncate bottom (1-keep_frac) eigen-directions
    k = max(1, int(n_tr * keep_frac))
    vi_reg = vi.copy()
    vi_reg[k:] = 0
    vi_reg = vi_reg / (np.sum(vi_reg) + 1e-12)
    dalign_reg = np.sum(vi_reg * (evalsT - evals0)) / (np.sum(vi_reg * evals0) + 1e-12)

    # Gain on held-out
    K0_te = (m.get_features(X_te) @ H1.T) / WIDTH
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    gain = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {'seed': seed, 'dalign_raw': float(dalign_raw), 'dalign_reg': float(dalign_reg), 'gain': gain}

if __name__ == '__main__':
    from scipy.stats import spearmanr
    n_total = 240; K = 5
    all_folds = []
    for s in range(5):
        idx = np.random.RandomState(42 + s).permutation(n_total)
        fold_size = n_total // K
        for fold in range(K):
            test_idx = idx[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(idx, test_idx)
            r = run_fold(train_idx, test_idx, s)
            r['fold'] = fold; all_folds.append(r)

    raw_d = [r['dalign_raw'] for r in all_folds]
    reg_d = [r['dalign_reg'] for r in all_folds]
    gains = [r['gain'] for r in all_folds]
    rho_raw, p_raw = spearmanr(raw_d, gains)
    rho_reg, p_reg = spearmanr(reg_d, gains)
    print(f"Unregularized: ρ={rho_raw:.4f}, p={p_raw:.4e}")
    print(f"Regularized (top 50%): ρ={rho_reg:.4f}, p={p_reg:.4e}")

    out = os.path.expanduser('~/reg_align_results.json')
    with open(out, 'w') as f:
        json.dump(all_folds, f)
    print(f"Saved to {out}")
