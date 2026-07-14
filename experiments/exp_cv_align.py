"""
Cross-validated task-aligned metric.
For poly width=64: compute Δalign from 4/5 of training data,
predict improvement on held-out 1/5. 5-fold CV, 5 seeds.
Truly predictive: v_i and improvement never share the same data.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

N, EPOCHS = 300, 500
WIDTH = 64
SEEDS = 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, WIDTH, bias=False)
        self.fc2 = nn.Linear(WIDTH, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def run_fold(train_idx, test_idx, seed):
    rng = np.random.RandomState(42 + seed)
    Xall = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    yall = Xall[:, 0]**2 + Xall[:, 1]
    yall = (yall - yall.mean()) / yall.std()

    X_tr_all = Xall[train_idx]
    y_tr_all = yall[train_idx]
    X_te = Xall[test_idx]
    y_te = yall[test_idx]
    n_tr = len(train_idx)

    m = Net()
    torch.manual_seed(42 + seed)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/WIDTH))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr_all)
    yt = torch.FloatTensor(y_tr_all + 0.1 * rng.randn(n_tr))

    # Initial features & kernel on TRAINING data
    H0 = m.get_features(X_tr_all)
    K0 = H0 @ H0.T / WIDTH

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr_all)
    K1 = H1 @ H1.T / WIDTH

    # Spectral quantities from TRAINING data only
    evals0, evecs0 = np.linalg.eigh(K0)
    evals0 = evals0[::-1]; evecs0 = evecs0[:, ::-1]
    evalsT, evecsT = np.linalg.eigh(K1)
    evalsT = evalsT[::-1]; evecsT = evecsT[:, ::-1]

    # v_i from training data projection
    proj = evecs0.T @ y_tr_all
    vi = (proj ** 2) / n_tr
    dalign = np.sum(vi * (evalsT - evals0))
    dalign_norm = dalign / (np.sum(vi * evals0) + 1e-12)

    # Generalization on TEST data
    K0_te = (m.get_features(X_te) @ H1.T) / WIDTH
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr_all)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr_all)
    err0 = float(np.mean((pred0 - y_te)**2))
    err1 = float(np.mean((pred1 - y_te)**2))
    gain = err0 - err1

    return {'dalign': float(dalign_norm), 'gain': float(gain), 'err0': float(err0), 'err1': float(err1)}

if __name__ == '__main__':
    all_folds = []
    n_total = 240  # training samples per split
    k = 5

    for s in range(SEEDS):
        fold_size = n_total // k
        idx = np.random.RandomState(42 + s).permutation(n_total)
        for fold in range(k):
            test_idx = idx[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(idx, test_idx)
            r = run_fold(train_idx, test_idx, s)
            r['seed'] = s; r['fold'] = fold
            all_folds.append(r)
            print(f"[seed={s} fold={fold}] Δalign={r['dalign']:.4f} gain={r['gain']:.4f}")

    from scipy.stats import spearmanr
    daligns = [r['dalign'] for r in all_folds]
    gains = [r['gain'] for r in all_folds]
    rho, p = spearmanr(daligns, gains)
    print(f"\n=== Cross-validated: ρ(Δalign, Gain) = {rho:.4f}, p={p:.4e} (n={len(all_folds)})")

    out = os.path.expanduser('~/cv_align_results.json')
    with open(out, 'w') as f:
        json.dump(all_folds, f)
    print(f"Saved to {out}")
