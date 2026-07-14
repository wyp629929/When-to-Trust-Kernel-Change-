"""Random-label baseline: how much kernel change occurs under null (no signal)?"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json
from scipy.stats import spearmanr
import warnings
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

def run_one(w, seed, X, y):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), 300, replace=False)
    Xi, yi = X[idx], y[idx].copy()
    rng.shuffle(yi)  # shuffle labels — no signal
    yi = (yi - yi.mean()) / yi.std()
    n_tr = 240
    X_tr, X_te = Xi[:n_tr], Xi[n_tr:]
    y_tr, y_te = yi[:n_tr], yi[n_tr:]

    m = Net(10, w)
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor(y_tr + 0.1 * np.random.randn(n_tr))

    H0 = m.get_features(X_tr)
    K0 = (H0 @ H0.T) / w

    for ep in range(500):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = (H1 @ H1.T) / w
    frob = np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100

    # KRR test
    K0_te = (m.get_features(X_te) @ m.get_features(X_tr).T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
    err0 = np.mean((pred0 - y_te) ** 2)
    err1 = np.mean((pred1 - y_te) ** 2)

    return {'w': int(w), 'seed': int(seed), 'frob': float(frob), 'delta_err': float(err0 - err1),
            'cka': float(cka(K0, K1)), 'err0': float(err0), 'err1': float(err1)}

print("Generating synthetic data (poly target, d=10)...")
rng = np.random.RandomState(0)
Xall = rng.uniform(-1, 1, (3000, 10)).astype(np.float32)
yall = (Xall[:, 0]**2 + Xall[:, 1]).astype(np.float32)

WIDTHS = [32, 64, 128, 256, 512, 1024]
results = []
for w in WIDTHS:
    for s in range(5):
        r = run_one(w, 42 + s, Xall, yall)
        results.append(r)
        print(f"  width {w} seed {s}: frob={r['frob']:.1f}%, delta_err={r['delta_err']:.4f}, cka={r['cka']:.4f}")

# Save
with open('/Users/wangyaoping/Desktop/ml_paper/figures/random_label_results.json', 'w') as f:
    json.dump(results, f)

# Summary
print(f"\nRandom-label baseline ({5} seeds each):")
print(f"{'Width':<8} {'CKA':<12} {'Frob%':<12} {'ΔErr':<12}")
for w in WIDTHS:
    pts = [r for r in results if r['w'] == w]
    cka_m = np.mean([r['cka'] for r in pts])
    frob_m = np.mean([r['frob'] for r in pts])
    de_m = np.mean([r['delta_err'] for r in pts])
    print(f"{w:<8} {cka_m:<12.4f} {frob_m:<12.1f} {de_m:<+12.4f}")

# Compare with original (signal) results
print("\nComparison with signal (poly, original labels):")
frobs_null = [r['frob'] for r in results]
ders_null = [r['delta_err'] for r in results]
print(f"Null: mean frob={np.mean(frobs_null):.1f}%, mean delta_err={np.mean(ders_null):.4f}")
rho_null, p_null = spearmanr(frobs_null, ders_null)
print(f"Null Spearman ρ = {rho_null:.3f} (p = {p_null:.4f})")
