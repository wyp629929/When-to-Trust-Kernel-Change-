"""Null baseline: shuffled labels for highfreq and GMM.
Shows kernel change under null persists for all datasets, not just poly.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import cka_from_features

SEEDS = 5
WIDTHS = [32, 64, 128, 256, 512, 1024]
N, EPOCHS = 300, 500

class Net(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.fc1 = nn.Linear(d, w, bias=False)
        self.fc2 = nn.Linear(w, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def run_one(ds_name, w, seed):
    rng = np.random.RandomState(42 + seed)
    if ds_name == 'highfreq':
        X = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
        y = np.sin(5*X[:, 0]) + np.cos(7*X[:, 1])
    elif ds_name == 'gmm':
        X = np.vstack([rng.randn(N//2, 10)*0.5 + 1.0, rng.randn(N - N//2, 10)*0.5 - 1.0]).astype(np.float32)
        y = np.hstack([np.zeros(N//2), np.ones(N - N//2)])
    rng.shuffle(y)  # shuffle labels — null signal
    y = (y - y.mean()) / y.std()

    n_tr = N - 60
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    m = Net(10, w)
    torch.manual_seed(42 + seed)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/w))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * np.random.RandomState(seed).randn(n_tr))

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / w

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / w

    cka_val = float(cka_from_features(H0, H1))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    K0_te = (m.get_features(X_te) @ H1.T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {'ds': ds_name, 'width': int(w), 'seed': int(seed), 'cka': cka_val, 'frob': frob, 'delta_err': delta_err}

if __name__ == '__main__':
    all_results = []
    t0 = time.time()
    for ds in ['highfreq', 'gmm']:
        for w in WIDTHS:
            for s in range(SEEDS):
                r = run_one(ds, w, s)
                all_results.append(r)
                print(f"[{ds}] w={w} seed={s}: CKA={r['cka']:.4f} Frob={r['frob']:.1f}% ΔErr={r['delta_err']:.4f}")

    out = './figures/null_baseline_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f)

    from scipy.stats import spearmanr
    for ds in ['highfreq', 'gmm']:
        pts = [r for r in all_results if r['ds'] == ds]
        frobs = [r['frob'] for r in pts]
        ders = [r['delta_err'] for r in pts]
        rho, p = spearmanr(frobs, ders)
        cka_mean = np.mean([r['cka'] for r in pts])
        frob_mean = np.mean([r['frob'] for r in pts])
        print(f"\n{ds} null (n={len(pts)}): ρ={rho:.3f}, p={p:.4e}")
        print(f"  mean CKA={cka_mean:.3f}, mean Frob={frob_mean:.1f}%")
        for w in WIDTHS:
            wp = [r for r in pts if r['width'] == w]
            print(f"  w={w}: CKA={np.mean([r['cka'] for r in wp]):.3f} Frob={np.mean([r['frob'] for r in wp]):.1f}% ΔErr={np.mean([r['delta_err'] for r in wp]):.4f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
