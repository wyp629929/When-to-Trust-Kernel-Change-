"""
μP/NTK parameterization comparison.
Tests whether the dissociation (ΔK vs Gain) depends on parameterization.
NTK param: init std = 1/sqrt(fan_in), same LR.
Compares Standard (He init) vs NTK param on poly task.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import cka_from_features

SEEDS = 5
WIDTHS = [32, 64, 128]
N, EPOCHS = 300, 500

class Net(nn.Module):
    def __init__(self, d, w, ntk=False):
        super().__init__()
        self.fc1 = nn.Linear(d, w, bias=False)
        self.fc2 = nn.Linear(w, 1, bias=False)
        self.ntk = ntk
        self._init_weights(d, w)

    def _init_weights(self, d, w):
        if self.ntk:
            nn.init.normal_(self.fc1.weight, std=1.0 / np.sqrt(d))
            nn.init.normal_(self.fc2.weight, std=1.0 / np.sqrt(w))
        else:
            nn.init.normal_(self.fc1.weight, std=np.sqrt(2.0 / d))
            nn.init.normal_(self.fc2.weight, std=np.sqrt(2.0 / w))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).flatten()

    def get_features(self, x):
        with torch.no_grad():
            return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def run_one(param, seed):
    rng = np.random.RandomState(42 + seed)
    X = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    y = X[:, 0]**2 + X[:, 1]
    y = (y - y.mean()) / y.std()
    n_tr = N - 60
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    w = 64  # single width for quick comparison
    m = Net(10, w, ntk=(param == 'ntk'))
    torch.manual_seed(42 + seed)
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

    return {'param': param, 'width': w, 'seed': seed,
            'cka': cka_val, 'frob': frob, 'gain': delta_err}

def run_width_sweep(param, seed):
    """Run multiple widths for one seed."""
    rng = np.random.RandomState(42 + seed)
    X = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    y = X[:, 0]**2 + X[:, 1]
    y = (y - y.mean()) / y.std()
    n_tr = N - 60
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    results = []
    for w in WIDTHS:
        m = Net(10, w, ntk=(param == 'ntk'))
        torch.manual_seed(42 + seed)
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

        results.append({'param': param, 'width': w, 'seed': seed,
                        'cka': cka_val, 'frob': frob, 'gain': delta_err})
    return results

if __name__ == '__main__':
    all_results = []
    for param in ['standard', 'ntk']:
        for s in range(SEEDS):
            rs = run_width_sweep(param, s)
            all_results.extend(rs)
            for r in rs:
                print(f"[{param}] w={r['width']} seed={s}: CKA={r['cka']:.4f} Frob={r['frob']:.1f}% Gain={r['gain']:.4f}")

    out = './figures/mup_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f)

    # Summary
    for param in ['standard', 'ntk']:
        pts = [r for r in all_results if r['param'] == param]
        for w in WIDTHS:
            wp = [r for r in pts if r['width'] == w]
            cka_m = np.mean([r['cka'] for r in wp])
            frob_m = np.mean([r['frob'] for r in wp])
            gain_m = np.mean([r['gain'] for r in wp])
            print(f"\n{param} w={w}: CKA={cka_m:.4f}±{np.std([r['cka'] for r in wp]):.4f}, "
                  f"Frob={frob_m:.1f}±{np.std([r['frob'] for r in wp]):.1f}, "
                  f"Gain={gain_m:.4f}±{np.std([r['gain'] for r in wp]):.4f}")
