"""
Extended width sweep: adds intermediate widths for tighter CIs.
3 datasets x 11 widths x 10 seeds = 330 runs.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import svcca, pwcca, cka_from_features

SEEDS = 10
WIDTHS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
N, EPOCHS, LR = 300, 500, 0.05

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
        X[:n//2] = rng.randn(n//2, 10)*0.5 + 1.0
        X[n//2:] = rng.randn(n-n//2, 10)*0.5 - 1.0
        y = np.array([1]*(n//2) + [0]*(n-n//2))
    else:
        raise ValueError(name)
    return X, y

def run_one(ds, w, seed):
    rng = np.random.RandomState(42 + seed)
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    Xall, yall = gen_data(ds, N, rng)
    idx = rng.choice(N, N, replace=False)
    Xi, yi = Xall[idx], yall[idx]
    yi = (yi - yi.mean()) / yi.std()
    n_tr = N - 60
    X_tr, X_te = Xi[:n_tr], Xi[n_tr:]
    y_tr, y_te = yi[:n_tr], yi[n_tr:]

    m = Net(10, w)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/w))
    opt = optim.SGD(m.parameters(), lr=LR, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * rng.randn(n_tr))

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / w
    reg = 0.01 * n_tr

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / w
    K0_te = (m.get_features(X_te) @ H1.T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + reg * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + reg * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    cka = float(cka_from_features(H0, H1))
    sv = float(svcca(H0, H1))
    pw = float(pwcca(H0, H1, y_tr))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)

    return {'ds': ds, 'width': int(w), 'seed': seed,
            'cka': cka, 'svcca': sv, 'pwcca': pw, 'frob': frob, 'delta_err': delta_err}

if __name__ == '__main__':
    t0 = time.time()
    results = []
    for ds in ['poly', 'highfreq', 'gmm']:
        for w in WIDTHS:
            for s in range(SEEDS):
                r = run_one(ds, w, s)
                results.append(r)
    print(f"Total: {len(results)} runs in {time.time()-t0:.0f}s")
    out = os.path.expanduser('~/width_extended_results.json')
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {out}")

    # Spearman correlations
    from scipy.stats import spearmanr
    print("\n=== Config-mean Spearman vs DeltaError ===")
    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [r for r in results if r['ds'] == ds]
        means = {}
        for r in pts:
            w = r['width']
            if w not in means:
                means[w] = {'frob': []}
            means[w]['frob'].append(r['frob'])
        ws = sorted(means.keys())
        fv = [np.mean(means[w]['frob']) for w in ws]
        dv = []
        for w in ws:
            dw = np.mean([r['delta_err'] for r in pts if r['width'] == w])
            dv.append(dw)
        rho, p = spearmanr(fv, dv)
        # Bootstrap CI
        rng_b = np.random.RandomState(42)
        rhos = []
        for _ in range(10000):
            idx = rng_b.choice(len(ws), len(ws), replace=True)
            r, _ = spearmanr([fv[i] for i in idx], [dv[i] for i in idx])
            rhos.append(r)
        ci = (np.percentile(rhos, 2.5), np.percentile(rhos, 97.5))
        print(f"  {ds}: n={len(ws)}, rho={rho:.3f}, 95%CI=[{ci[0]:.3f},{ci[1]:.3f}], p={p:.4f}")
