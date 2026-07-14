"""
Spectral target sweep using NTK kernel for clean spectral decomposition.
Constructs targets with controlled alignment to the initial NTK eigenspace,
then trains MLPs and measures diagnostic reliability.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os
warnings.filterwarnings('ignore')

SEEDS, WIDTHS = 5, [32, 64, 128, 256, 512, 1024]
N, D, EPOCHS = 300, 10, 500
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]

class Net(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.fc1 = nn.Linear(d, w, bias=False)
        self.fc2 = nn.Linear(w, 1, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x, device):
        with torch.no_grad():
            return torch.relu(self.fc1(x.to(device))).cpu().numpy()

def ntk_kernel(X):
    """Two-layer ReLU NTK on sphere."""
    dot = X @ X.T
    cos_theta = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    k_nngp = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    return k_nngp + ((np.pi - theta) / np.pi) * dot

def build_targets(X, betas):
    """Precompute targets for all betas from NTK eigendecomposition."""
    # Spherical data for clean NTK spectrum
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = ntk_kernel(X_norm)
    evals, evecs = np.linalg.eigh(K)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    rng = np.random.RandomState(42)
    targets = {}
    for beta in betas:
        weights = (evals ** (beta / 2)) * rng.randn(len(evals))
        y = evecs @ weights
        y = (y - y.mean()) / y.std()
        # Predicted alignment: fraction of v_i in top 3 directions
        v = (evecs.T @ y) ** 2 / len(y)
        v = v / (v.sum() + 1e-12)
        targets[beta] = (y, float(v[:3].sum()))
    return targets, X_norm

def run_one(X, y_tr, y_te, w, seed, device, n_tr):
    torch.manual_seed(42 + seed)
    rng = np.random.RandomState(42 + seed)
    m = Net(D, w).to(device)
    nn.init.normal_(m.fc1.weight, std=np.sqrt(2/D))
    nn.init.normal_(m.fc2.weight, std=np.sqrt(2/w))
    opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    xt = torch.from_numpy(X[:n_tr]).float().to(device)
    yt = torch.from_numpy(y_tr + 0.1 * rng.randn(n_tr)).float().to(device)

    H0 = m.get_features(xt, device)
    K0 = H0 @ H0.T / w
    reg = 0.01 * n_tr
    X_te_t = torch.from_numpy(X[-len(y_te):]).float().to(device)

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(xt, device)
    K1 = H1 @ H1.T / w
    K0_te = (m.get_features(X_te_t, device) @ H1.T) / w
    pred0 = K0_te @ np.linalg.solve(K0 + reg * np.eye(n_tr), y_tr)
    pred1 = K0_te @ np.linalg.solve(K1 + reg * np.eye(n_tr), y_tr)
    delta_err = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    return frob, delta_err

if __name__ == '__main__':
    from scipy.stats import spearmanr
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    rng_data = np.random.RandomState(42)
    X = rng_data.uniform(-1, 1, (N, D)).astype(np.float32)
    targets, X_norm = build_targets(X, BETAS)
    n_tr = N - 60

    all_results = []
    for beta in BETAS:
        y, pred_align = targets[beta]
        y_tr, y_te = y[:n_tr], y[n_tr:]
        # Store per-width per-seed results
        config_data = []
        for w in WIDTHS:
            for s in range(SEEDS):
                f, de = run_one(X_norm, y_tr, y_te, w, s, device, n_tr)
                config_data.append({'beta': float(beta), 'width': w, 'seed': s, 'frob': f, 'delta_err': de})
        # Compute config-mean Spearman rho across widths
        means = {}
        for cd in config_data:
            w = cd['width']
            if w not in means:
                means[w] = {'frob': [], 'de': []}
            means[w]['frob'].append(cd['frob'])
            means[w]['de'].append(cd['delta_err'])
        ws = sorted(means.keys())
        fv = [np.mean(means[w]['frob']) for w in ws]
        dv = [np.mean(means[w]['de']) for w in ws]
        actual_rho = float(spearmanr(fv, dv)[0]) if len(set(fv)) > 1 and len(set(dv)) > 1 else float('nan')
        all_results.append({'beta': float(beta), 'pred_align': float(pred_align), 'actual_rho': actual_rho})
        print(f"β={beta:.1f} align={pred_align:.3f} → ρ={actual_rho:.3f}", flush=True)

    # Overall Spearman
    valid = [r for r in all_results if not (np.isnan(r['pred_align']) or np.isnan(r['actual_rho']))]
    if len(valid) >= 4:
        preds = [r['pred_align'] for r in valid]
        actuals = [r['actual_rho'] for r in valid]
        rho_total, p_total = spearmanr(preds, actuals)
        print(f"\nPredicted alignment vs actual ρ: ρ={rho_total:.3f}, p={p_total:.4f}", flush=True)
    else:
        print(f"\nNot enough valid points ({len(valid)})", flush=True)

    out = os.path.expanduser('~/spectral_sweep_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"Saved to {out}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)
