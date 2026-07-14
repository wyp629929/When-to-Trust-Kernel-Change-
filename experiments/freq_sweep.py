"""
Frequency sweep: sin(k*x) for k=1,2,3,5.
Tests whether high-frequency failure is just capacity or reflects
spectral alignment dynamics of kernel evolution.

Predictions:
- Low freq (k=1,2): kernel can adapt, gain > 0
- High freq (k=3,5): kernel struggles, gain ≈ 0 or negative
But crucially: check whether CKA/SVCCA still fail to predict this.
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

def run_one(k, w, seed):
    rng = np.random.RandomState(42 + seed)
    X = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    y = np.sin(k * X[:, 0])  # 1D frequency, rest is noise dims
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

    cka_val = cka_from_features(H0, H1)
    svcca_val = svcca(H0, H1)
    frob = np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100

    K0_te = (m.get_features(X_te) @ H1.T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {'k': int(k), 'width': int(w), 'seed': int(seed),
            'cka': float(cka_val), 'svcca': float(svcca_val),
            'frob': float(frob), 'delta_err': float(delta_err)}

if __name__ == '__main__':
    FREQS = [1, 2, 3, 5]
    all_results = []
    total = len(FREQS) * len(WIDTHS) * SEEDS
    done = 0
    t0 = time.time()

    for k in FREQS:
        for w in WIDTHS:
            for s in range(SEEDS):
                r = run_one(k, w, s)
                all_results.append(r)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total}] k={k} w={w} seed={s}: "
                      f"CKA={r['cka']:.4f} gain={r['delta_err']:.4f} {elapsed:.0f}s")

    out = '/Users/wangyaoping/Desktop/ml_paper/figures/freq_sweep_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"\n{total} runs complete. Saved to {out}  [{time.time()-t0:.0f}s]")

    # Summary
    for k in FREQS:
        pts = [r for r in all_results if r['k'] == k]
        gains = [r['delta_err'] for r in pts]
        ckas = [r['cka'] for r in pts]
        print(f"k={k}: mean gain={np.mean(gains):.4f}±{np.std(gains):.4f}, "
              f"mean CKA={np.mean(ckas):.4f}±{np.std(ckas):.4f}")
