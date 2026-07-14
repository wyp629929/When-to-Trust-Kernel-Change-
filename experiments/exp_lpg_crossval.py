"""
LPG cross-validation experiment.
Computes Linear Probe Gain for each (dataset, width, seed) and its
Spearman correlation with DeltaError, to verify that LPG avoids the
negative CV correlation problem (rho=-0.45 for D_align).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import json, time, warnings, os
warnings.filterwarnings('ignore')

N, D = 300, 10
EPOCHS = 500
WIDTHS = [32, 64, 128, 256, 512, 1024]
SEEDS = 5
LR = 0.05

class TwoLayerNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(D, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(x)).cpu().numpy()

def make_data(name, n, rng):
    X = rng.uniform(-1, 1, (n, D)).astype(np.float32)
    if name == 'poly':
        y = X[:, 0]**2 + X[:, 1]
    elif name == 'highfreq':
        y = np.sin(5*X[:, 0]) + np.cos(7*X[:, 1])
    elif name == 'gmm':
        X[:n//2] = rng.randn(n//2, D)*0.5 + 1.0
        X[n//2:] = rng.randn(n-n//2, D)*0.5 - 1.0
        y = np.array([1]*(n//2) + [0]*(n-n//2))
    else:
        raise ValueError(name)
    y = (y - y.mean()) / y.std()
    return X, y

def compute_lpg_and_delta(ds_name, width, seed):
    rng = np.random.RandomState(42 + seed)
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    X, y = make_data(ds_name, N, rng)
    y_noisy = y + 0.1 * rng.randn(N)
    n_tr = N - 60  # 80/20 split
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]
    y_noisy_tr = y_noisy[:n_tr]

    model = TwoLayerNet(width).to(device)
    nn.init.normal_(model.fc1.weight, std=np.sqrt(2/D))
    nn.init.normal_(model.fc2.weight, std=np.sqrt(2/width))
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    x_t = torch.FloatTensor(X_tr).to(device)
    y_t = torch.FloatTensor(y_noisy_tr).to(device)

    # Initial features + linear probe
    H0_tr = model.get_features(x_t)
    H0_te = model.get_features(torch.FloatTensor(X_te).to(device))
    lr0 = LinearRegression().fit(H0_tr, y_tr)
    r2_init = lr0.score(H0_te, y_te)

    # Initial kernel predictor (for DeltaError)
    K0 = H0_tr @ H0_tr.T / width
    n_tr = len(y_tr)
    reg = 0.01 * n_tr
    pred0 = (H0_te @ H0_tr.T / width) @ np.linalg.solve(K0 + reg * np.eye(n_tr), y_tr)
    err_init = float(np.mean((pred0 - y_te)**2))

    # Train
    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(model(x_t), y_t).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Post-training features + linear probe
    H1_tr = model.get_features(x_t)
    H1_te = model.get_features(torch.FloatTensor(X_te).to(device))
    lr1 = LinearRegression().fit(H1_tr, y_tr)
    r2_post = lr1.score(H1_te, y_te)

    # Post-training kernel predictor (for DeltaError)
    K1 = H1_tr @ H1_tr.T / width
    pred1 = (H1_te @ H1_tr.T / width) @ np.linalg.solve(K1 + reg * np.eye(n_tr), y_tr)
    err_post = float(np.mean((pred1 - y_te)**2))

    lpg = r2_post - r2_init
    delta_err = err_init - err_post

    return {'ds': ds_name, 'width': int(width), 'seed': int(seed),
            'r2_init': r2_init, 'r2_post': r2_post, 'lpg': lpg,
            'err_init': err_init, 'err_post': err_post, 'delta_err': delta_err}

if __name__ == '__main__':
    t0 = time.time()
    results = []
    for ds in ['poly', 'highfreq', 'gmm']:
        for w in WIDTHS:
            for s in range(SEEDS):
                r = compute_lpg_and_delta(ds, w, s)
                results.append(r)
                print(f"[{ds}] w={w} seed={s}: LPG={r['lpg']:.4f} DeltaErr={r['delta_err']:.4f} ({time.time()-t0:.0f}s)")

    # Save raw results
    out_path = os.path.expanduser('~/lpg_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved to {out_path}")

    # Compute Spearman correlations
    print("\n" + "="*60)
    print("LPG vs DeltaError Spearman correlations")
    print("="*60)

    # Per-dataset, across widths (using configuration means)
    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [r for r in results if r['ds'] == ds]
        # Configuration means: average over seeds per width
        means = {}
        for r in pts:
            key = r['width']
            if key not in means:
                means[key] = {'lpg': [], 'delta_err': []}
            means[key]['lpg'].append(r['lpg'])
            means[key]['delta_err'].append(r['delta_err'])
        w_list = sorted(means.keys())
        lpg_means = [np.mean(means[w]['lpg']) for w in w_list]
        de_means = [np.mean(means[w]['delta_err']) for w in w_list]

        if len(w_list) >= 4:
            rho, p = spearmanr(lpg_means, de_means)
            print(f"\n{ds} (n={len(w_list)} config means): rho={rho:.3f}, p={p:.4f}")
            # Also seed-level correlation
            all_lpg = [r['lpg'] for r in pts]
            all_de = [r['delta_err'] for r in pts]
            rho_s, p_s = spearmanr(all_lpg, all_de)
            print(f"  Seed-level (n={len(pts)}): rho={rho_s:.3f}, p={p_s:.4f}")
        else:
            print(f"\n{ds}: insufficient data points")

    # Pooled across all datasets (18 configuration means)
    all_means = {}
    for r in results:
        key = (r['ds'], r['width'])
        if key not in all_means:
            all_means[key] = {'lpg': [], 'delta_err': []}
        all_means[key]['lpg'].append(r['lpg'])
        all_means[key]['delta_err'].append(r['delta_err'])

    keys = sorted(all_means.keys())
    pooled_lpg = [np.mean(all_means[k]['lpg']) for k in keys]
    pooled_de = [np.mean(all_means[k]['delta_err']) for k in keys]
    rho_p, p_p = spearmanr(pooled_lpg, pooled_de)
    print(f"\nPooled (n={len(keys)} config means): rho={rho_p:.3f}, p={p_p:.4f}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
