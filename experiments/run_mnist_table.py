"""
Generate MNIST binary classification (0 vs 1) table data.
Saves CKA, Frobenius change, and delta_err per width.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import cka_from_features

SEEDS = 5
WIDTHS = [32, 64, 128, 256, 512, 1024]
EPOCHS = 200
LR = 0.05

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

print("Loading MNIST...")
mnist = datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transforms.ToTensor())
Xall = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
yall = mnist.targets.numpy()

# Binary: 0 vs 1
mask = (yall == 0) | (yall == 1)
Xbin, ybin = Xall[mask], yall[mask].astype(np.float32)
ybin = (ybin - ybin.mean()) / ybin.std()

# Fixed split
n_tr = 1000
rng_global = np.random.RandomState(42)
idx = rng_global.permutation(len(Xbin))
X, y = Xbin[idx], ybin[idx]
X_tr, X_te = X[:n_tr], X[n_tr:n_tr+500]
y_tr, y_te = y[:n_tr], y[n_tr:n_tr+500]

results = []
for w in WIDTHS:
    for s in range(SEEDS):
        t0 = time.time()
        m = Net(784, w)
        torch.manual_seed(42 + s)
        nn.init.normal_(m.fc1.weight, std=np.sqrt(2/784))
        nn.init.normal_(m.fc2.weight, std=np.sqrt(2/w))
        opt = optim.SGD(m.parameters(), lr=LR, momentum=0.9)
        xt = torch.FloatTensor(X_tr)
        yt = torch.FloatTensor(y_tr + 0.01 * np.random.RandomState(s).randn(n_tr))

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

        t = time.time() - t0
        results.append({'width': w, 'seed': s, 'cka': cka_val, 'frob': frob, 'delta_err': delta_err})
        print(f"[MNIST] w={w} seed={s}: CKA={cka_val:.4f} Frob={frob:.1f}% dErr={delta_err:.4f} {t:.0f}s")

# Save
out = './figures/mnist_full_results.json'
with open(out, 'w') as f:
    json.dump(results, f)

# Summary
print("\nWidth  CKA         Frob%        ΔError")
for w in WIDTHS:
    pts = [r for r in results if r['width'] == w]
    cka_m = np.mean([r['cka'] for r in pts])
    cka_s = np.std([r['cka'] for r in pts])
    frob_m = np.mean([r['frob'] for r in pts])
    frob_s = np.std([r['frob'] for r in pts])
    de_m = np.mean([r['delta_err'] for r in pts])
    de_s = np.std([r['delta_err'] for r in pts])
    print(f"{w:<6} {cka_m:.4f}±{cka_s:.4f}  {frob_m:.1f}±{frob_s:.1f}  {de_m:.4f}±{de_s:.4f}")
