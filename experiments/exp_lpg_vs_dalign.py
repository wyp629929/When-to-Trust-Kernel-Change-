"""
Direct comparison: LPG vs D_align under 5-fold CV (poly task).
Exactly replicates exp_reg_align_v3.py procedure, but also computes LPG.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import json, time, warnings, os
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
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad(): return torch.relu(self.fc1(x)).numpy()

def run_fold(train_idx, test_idx, seed):
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

    # Initial features
    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / WIDTH
    evals0, evecs0 = np.linalg.eigh(K0)
    evals0 = evals0[::-1]; evecs0 = evecs0[:, ::-1]

    # LPG init: linear probe on train, eval on test
    H0_te = m.get_features(X_te)
    lr0 = LinearRegression().fit(H0, y_tr)
    r2_init = lr0.score(H0_te, y_te)

    # Train
    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    # Post-training features
    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / WIDTH
    evalsT, _ = np.linalg.eigh(K1)
    evalsT = evalsT[::-1]

    # LPG post
    H1_te = m.get_features(X_te)
    lr1 = LinearRegression().fit(H1, y_tr)
    r2_post = lr1.score(H1_te, y_te)
    lpg = r2_post - r2_init

    # D_align (exactly as paper, c=0.05)
    proj = evecs0.T @ y_tr
    vi_raw = (proj ** 2) / n_tr
    vi_reg = vi_raw + 0.05
    vi_reg = vi_reg / (np.sum(vi_reg) + 1e-12)
    dalign = np.sum(vi_reg * (evalsT - evals0)) / (np.sum(vi_reg * evals0) + 1e-12)

    # Gain (exactly as paper)
    K0_te = (m.get_features(X_te) @ H1.T) / WIDTH
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    gain = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {'seed': seed, 'fold': 0, 'lpg': lpg, 'dalign': dalign, 'gain': gain}

if __name__ == '__main__':
    t0 = time.time()
    n_total = 240; K = 5; SEEDS = 5
    all_results = []
    for s in range(SEEDS):
        idx = np.random.RandomState(42 + s).permutation(n_total)
        fold_size = n_total // K
        for fold in range(K):
            test_idx = idx[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(idx, test_idx)
            r = run_fold(train_idx, test_idx, s)
            r['fold'] = fold
            all_results.append(r)

    lpg_vals = [r['lpg'] for r in all_results]
    da_vals = [r['dalign'] for r in all_results]
    g_vals = [r['gain'] for r in all_results]

    rho_lpg, p_lpg = spearmanr(lpg_vals, g_vals)
    rho_da, p_da = spearmanr(da_vals, g_vals)

    print("="*60)
    print("5-fold CV: LPG vs D_align (exact paper replication)")
    print("="*60)
    print(f"n = {len(all_results)} (5 folds x {SEEDS} seeds)")
    print(f"\nD_align vs Gain: rho = {rho_da:.3f}, p = {p_da:.4f}")
    print(f"LPG vs Gain:     rho = {rho_lpg:.3f}, p = {p_lpg:.4f}")
    print(f"\nD_align mean: {np.mean(da_vals):.4f}, std: {np.std(da_vals):.4f}")
    print(f"LPG mean:     {np.mean(lpg_vals):.4f}, std: {np.std(lpg_vals):.4f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    out = os.path.expanduser('~/lpg_vs_dalign_results.json')
    with open(out, 'w') as f:
        # Convert numpy types
        clean = [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for k, v in r.items()} for r in all_results]
        json.dump(clean, f)
    print(f"Saved to {out}")
