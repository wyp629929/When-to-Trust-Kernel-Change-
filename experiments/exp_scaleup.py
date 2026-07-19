"""
Multi-seed width sweep with CKA, SVCCA, PWCCA.
3 datasets x 6 widths x 10 seeds = 180 runs.
Saves results to JSON for figure generation.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import svcca, pwcca, cka_from_features

SEEDS = 10
WIDTHS = [32, 64, 128, 256, 512, 1024]
N, EPOCHS = 300, 500

class Net(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.fc1 = nn.Linear(d, w, bias=False)
        self.fc2 = nn.Linear(w, 1, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad():
            return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def gen_data(name, n, rng):
    X = rng.uniform(-1, 1, (n, 10)).astype(np.float32)
    if name == 'poly':
        y = X[:, 0]**2 + X[:, 1]
    elif name == 'highfreq':
        y = np.sin(5*X[:, 0]) + np.cos(7*X[:, 1])
    elif name == 'gmm':
        X = np.vstack([rng.randn(n//2, 10)*0.5 + 1.0, rng.randn(n - n//2, 10)*0.5 - 1.0]).astype(np.float32)
        y = np.hstack([np.zeros(n//2), np.ones(n - n//2)])
    elif name == 'poly3':
        y = X[:, 0]**3 + X[:, 1]**2 + X[:, 2]
    elif name == 'mixedfreq':
        y = np.sin(2*X[:, 0]) + np.cos(3*X[:, 1]) + np.sin(X[:, 2])
    else:
        raise ValueError(name)
    return X, y

def run_one(ds, w, seed):
    rng = np.random.RandomState(42 + seed)
    Xall, yall = gen_data(ds, N, rng)
    idx = rng.choice(len(yall), N, replace=False)
    Xi, yi = Xall[idx], yall[idx]
    yi = (yi - yi.mean()) / yi.std()
    n_tr = N - 60
    X_tr, X_te = Xi[:n_tr], Xi[n_tr:]
    y_tr, y_te = yi[:n_tr], yi[n_tr:]

    m = Net(10, w)
    torch.manual_seed(42 + seed)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/w))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * np.random.RandomState(seed).randn(n_tr))

    # Initial features & kernel
    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / w

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    # Final features & kernel
    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / w

    # Metrics
    cka_val = float(cka_from_features(H0, H1))
    svcca_val = float(svcca(H0, H1))
    pwcca_val = float(pwcca(H0, H1, y_tr))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)

    # KRR test error
    K0_te = (m.get_features(X_te) @ H1.T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {
        'ds': ds, 'width': int(w), 'seed': int(seed),
        'cka': cka_val, 'svcca': svcca_val, 'pwcca': pwcca_val,
        'frob': frob, 'delta_err': delta_err,
    }

if __name__ == '__main__':
    DATASETS = ['poly', 'highfreq', 'gmm']
    all_results = []

    total = len(DATASETS) * len(WIDTHS) * SEEDS
    done = 0
    t0 = time.time()

    for ds in DATASETS:
        for w in WIDTHS:
            for s in range(SEEDS):
                r = run_one(ds, w, s)
                all_results.append(r)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total}] {ds} w={w} seed={s}: "
                      f"CKA={r['cka']:.4f} SVCCA={r['svcca']:.4f} "
                      f"gain={r['delta_err']:.4f}  {elapsed:.0f}s")

    out = './figures/exp_scaleup_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"\n{total} runs complete. Saved to {out}")
    print(f"Total time: {time.time() - t0:.0f}s")

    # Summary per dataset
    for ds in DATASETS:
        pts = [r for r in all_results if r['ds'] == ds]
        cka_vals = [r['cka'] for r in pts]
        svcca_vals = [r['svcca'] for r in pts]
        gains = [r['delta_err'] for r in pts]
        print(f"\n{ds}: mean CKA={np.mean(cka_vals):.4f}±{np.std(cka_vals):.4f}, "
              f"mean SVCCA={np.mean(svcca_vals):.4f}±{np.std(svcca_vals):.4f}, "
              f"mean gain={np.mean(gains):.4f}±{np.std(gains):.4f}")
