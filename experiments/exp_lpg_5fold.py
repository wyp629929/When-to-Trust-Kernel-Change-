"""
5-fold CV comparison: LPG vs Delta_align on the polynomial task.
Directly compares their cross-validated Spearman correlations,
replicating the D_align CV procedure from exp_reg_align_v3.py.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import json, time, warnings, os
warnings.filterwarnings('ignore')

N, EPOCHS, WIDTH = 300, 500, 64
K, SEEDS = 5, 5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, WIDTH, bias=False)
        self.fc2 = nn.Linear(WIDTH, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(x)).cpu().numpy()

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

    # --- Initial measures ---
    H0_tr = m.get_features(xt)
    H0_te = m.get_features(torch.FloatTensor(X_te))
    K0 = H0_tr @ H0_tr.T / WIDTH
    evals0, evecs0 = np.linalg.eigh(K0)
    evals0 = evals0[::-1]; evecs0 = evecs0[:, ::-1]

    # LPG init
    lr0 = LinearRegression().fit(H0_tr, y_tr)
    r2_init = lr0.score(H0_te, y_te)

    # Kernel-based init error
    reg = 0.01 * n_tr
    pred0 = (H0_te @ H0_tr.T / WIDTH) @ np.linalg.solve(K0 + reg * np.eye(n_tr), y_tr)
    err_init = float(np.mean((pred0 - y_te)**2))

    # D_align init
    proj = evecs0.T @ y_tr
    vi_raw = (proj ** 2) / n_tr

    # --- Train ---
    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    # --- Post-training measures ---
    H1_tr = m.get_features(xt)
    H1_te = m.get_features(torch.FloatTensor(X_te))
    K1 = H1_tr @ H1_tr.T / WIDTH
    evalsT, _ = np.linalg.eigh(K1)
    evalsT = evalsT[::-1]

    # D_align
    dalign = np.sum(vi_raw * (evalsT - evals0)) / (np.sum(vi_raw * evals0) + 1e-12)

    # LPG post
    lr1 = LinearRegression().fit(H1_tr, y_tr)
    r2_post = lr1.score(H1_te, y_te)
    lpg = r2_post - r2_init

    # Kernel-based post error
    pred1 = (H1_te @ H1_tr.T / WIDTH) @ np.linalg.solve(K1 + reg * np.eye(n_tr), y_tr)
    err_post = float(np.mean((pred1 - y_te)**2))
    delta_err = float(err_init - err_post)

    return {'seed': seed, 'fold': 0, 'lpg': lpg, 'dalign': dalign, 'delta_err': delta_err}

if __name__ == '__main__':
    t0 = time.time()
    n_total = 240
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

    # Spearman correlations (5-fold x 5 seeds = 25 points)
    lpg_vals = [r['lpg'] for r in all_results]
    da_vals = [r['dalign'] for r in all_results]
    de_vals = [r['delta_err'] for r in all_results]

    rho_lpg, p_lpg = spearmanr(lpg_vals, de_vals)
    rho_da, p_da = spearmanr(da_vals, de_vals)

    print("="*60)
    print("5-fold CV comparison: LPG vs D_align (poly task)")
    print("="*60)
    print(f"n = {len(all_results)} (5 folds x {SEEDS} seeds)")
    print(f"\nD_align vs DeltaError: rho = {rho_da:.3f}, p = {p_da:.4f}")
    print(f"LPG vs DeltaError:     rho = {rho_lpg:.3f}, p = {p_lpg:.4f}")
    print(f"\nD_align mean: {np.mean(da_vals):.4f}, std: {np.std(da_vals):.4f}")
    print(f"LPG mean:     {np.mean(lpg_vals):.4f}, std: {np.std(lpg_vals):.4f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    out = os.path.expanduser('~/lpg_5fold_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"Saved to {out}")
