"""
Low-frequency control: sin(x1) + cos(x2) at width=32.
Shows that highfreq result is NOT just capacity — at low freq,
kernel CAN adapt and gain is positive.
Compares: y = sin(x1) + cos(x2) vs y = sin(5*x1) + cos(7*x2)
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import svcca, pwcca, cka_from_features

SEEDS = 10
N, EPOCHS = 300, 500
WIDTH = 32

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

def run_one(name, seed):
    rng = np.random.RandomState(42 + seed)
    X = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
    if name == 'lowfreq':
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    elif name == 'highfreq':
        y = np.sin(5*X[:, 0]) + np.cos(7*X[:, 1])
    y = (y - y.mean()) / y.std()
    n_tr = N - 60
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    m = Net(10, WIDTH)
    torch.manual_seed(42 + seed)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/WIDTH))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * np.random.RandomState(seed).randn(n_tr))

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T / WIDTH

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T / WIDTH

    cka_val = float(cka_from_features(H0, H1))
    svcca_val = float(svcca(H0, H1))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)

    K0_te = (m.get_features(X_te) @ H1.T) / WIDTH
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

    return {'name': name, 'width': WIDTH, 'seed': seed,
            'cka': cka_val, 'svcca': svcca_val, 'frob': frob, 'delta_err': delta_err}

if __name__ == '__main__':
    all_results = []
    for name in ['lowfreq', 'highfreq']:
        for s in range(SEEDS):
            r = run_one(name, s)
            all_results.append(r)
            print(f"[{name}] seed={s}: CKA={r['cka']:.4f} gain={r['delta_err']:.4f} frob={r['frob']:.1f}%")

    out = './figures/lowfreq_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f)

    for name in ['lowfreq', 'highfreq']:
        pts = [r for r in all_results if r['name'] == name]
        gains = [r['delta_err'] for r in pts]
        ckas = [r['cka'] for r in pts]
        frobs = [r['frob'] for r in pts]
        print(f"\n{name} (w={WIDTH}, n={SEEDS}): "
              f"gain={np.mean(gains):.4f}±{np.std(gains):.4f}, "
              f"CKA={np.mean(ckas):.4f}±{np.std(ckas):.4f}, "
              f"Frob={np.mean(frobs):.1f}%±{np.std(frobs):.1f}")
