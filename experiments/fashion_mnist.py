"""FashionMNIST feature-kernel analysis + 5-seed extended width sweep"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, warnings, os
from torchvision import datasets, transforms
from scipy.linalg import eigh
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self,d,w):
        super().__init__(); self.fc1=nn.Linear(d,w,bias=False); self.fc2=nn.Linear(w,1,bias=False)
        nn.init.normal_(self.fc1.weight,std=np.sqrt(2/d))
        nn.init.normal_(self.fc2.weight,std=np.sqrt(2/w))
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self,x):
        with torch.no_grad(): return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

def cka(K,L):
    n=K.shape[0]; H=np.eye(n)-np.ones((n,n))/n; Kc=H@K@H; Lc=H@L@H
    return np.sum(Kc*Lc)/(np.sqrt(np.sum(Kc**2)*np.sum(Lc**2))+1e-12)

print("Loading FashionMNIST...")
dataset = datasets.FashionMNIST(root='/tmp/data', train=True, download=True, transform=transforms.ToTensor())
X = dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y = dataset.targets.numpy()
# Binary: class 0 (T-shirt/top) vs class 1 (Trouser) - well-separated categories
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask].astype(float)
y[y==0] = -1
print(f"Binary class 0/1: {len(y)} samples")

WIDTHS = [32, 64, 128, 256, 512, 1024]
n, epochs, n_seeds = 300, 500, 5
all_results = []

for w in WIDTHS:
    for s in range(n_seeds):
        rng = np.random.RandomState(42 + s)
        idx = rng.choice(len(y), n, replace=False)
        Xi, yi = X[idx], y[idx]
        yi = (yi - yi.mean()) / yi.std()
        n_tr = n - 60
        X_tr, X_te = Xi[:n_tr], Xi[n_tr:]
        y_tr, y_te = yi[:n_tr], yi[n_tr:]

        m = Net(784, w)
        opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
        xt = torch.FloatTensor(X_tr)
        yt = torch.FloatTensor(y_tr + 0.1 * np.random.randn(n_tr))
        H0 = m.get_features(X_tr)
        K0 = (H0 @ H0.T) / w

        for ep in range(epochs):
            opt.zero_grad()
            nn.MSELoss()(m(xt), yt).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        H1 = m.get_features(X_tr)
        K1 = (H1 @ H1.T) / w
        frob = np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100

        # KRR test error
        K0_te = (m.get_features(X_te) @ m.get_features(X_tr).T) / w
        K1_te = (m.get_features(X_te) @ m.get_features(X_tr).T) / w
        pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
        pred1 = K1_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
        err0 = np.mean((pred0 - y_te) ** 2)
        err1 = np.mean((pred1 - y_te) ** 2)

        all_results.append({'w': int(w), 'seed': int(s), 'frob': float(frob), 'delta_err': float(err0 - err1),
                           'cka': float(cka(K0, K1)), 'err0': float(err0), 'err1': float(err1)})
        del m

# Save
with open('/Users/wangyaoping/Desktop/ml_paper/figures/fashion_results.json', 'w') as f:
    json.dump(all_results, f)

# Print summary
print(f"\nResults ({n_seeds} seeds each):")
print(f"{'Width':<8} {'CKA':<12} {'Frob%':<12} {'ΔErr':<12}")
for w in WIDTHS:
    pts = [r for r in all_results if r['w'] == w]
    cka_m = np.mean([r['cka'] for r in pts])
    frob_m = np.mean([r['frob'] for r in pts])
    de_m = np.mean([r['delta_err'] for r in pts])
    print(f"{w:<8} {cka_m:<12.4f} {frob_m:<12.1f} {de_m:<+12.4f}")

# Correlation across all
frobs = [r['frob'] for r in all_results]
ders = [r['delta_err'] for r in all_results]
rho, p = spearmanr(frobs, ders)
print(f"\nSpearman ρ = {rho:.3f} (p = {p:.4f})")
