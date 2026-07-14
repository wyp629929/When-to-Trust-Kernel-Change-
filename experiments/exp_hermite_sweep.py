"""
Hermite degree sweep: targets = Hermite polynomials of degree 1..10.
Spectral alignment decreases with degree; headroom measured by init R^2.
Predicts: reliability_rho ~ alignment + headroom + alignment x headroom.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from numpy.polynomial.hermite_e import hermeval
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import json, time, warnings, os
warnings.filterwarnings('ignore')

SEEDS = 5
WIDTHS = [32, 64, 128, 256, 512, 1024]
N, D, EPOCHS = 300, 10, 500
DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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

def hermite_target(X, degree, w_dir, rng):
    """Evaluate degree-k Hermite polynomial along random direction."""
    t = X @ w_dir
    # herm_e_coeffs[k] = 1 means coefficient for t^k is 1
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 1.0
    y = hermeval(t, coeffs)
    return (y - y.mean()) / y.std()

def compute_v_dist(y, H):
    """Compute target projection v_i from feature matrix H."""
    K = H @ H.T / H.shape[1]
    evals, evecs = np.linalg.eigh(K)
    evecs = evecs[:, ::-1]
    v = (evecs.T @ y) ** 2 / len(y)
    return v / (v.sum() + 1e-12)

def run_config(X, y_tr, y_te, w, seed, device, n_tr, init_H):
    """Single training run, returns frob, delta_err."""
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
    return frob, delta_err, H0

def run_degree(deg, X, w_dir, n_tr, init_H, init_H_te, device):
    """Run all widths and seeds for a single Hermite degree, returns result dict."""
    from scipy.stats import spearmanr as _sr

    y = hermite_target(X, deg, w_dir, np.random.RandomState(42))
    y_tr, y_te = y[:n_tr], y[n_tr:]

    # Headroom = 1 - init_R2 (linear probe on ref features)
    lr = LinearRegression().fit(init_H, y_tr)
    init_r2 = lr.score(init_H_te, y_te)
    headroom = 1.0 - init_r2

    # Alignment from v_i distribution (using training data only)
    v = compute_v_dist(y_tr, init_H)
    align = float(v[:3].sum())

    # Train
    frobs, des = [], []
    for w in WIDTHS:
        for s in range(SEEDS):
            f, de, _ = run_config(X, y_tr, y_te, w, s, device, n_tr, init_H)
            frobs.append(f); des.append(de)

    # Config-mean rho
    f_arr = np.array(frobs).reshape(len(WIDTHS), SEEDS)
    d_arr = np.array(des).reshape(len(WIDTHS), SEEDS)
    f_means = f_arr.mean(axis=1)
    d_means = d_arr.mean(axis=1)
    if len(set(f_means)) > 1 and len(set(d_means)) > 1:
        actual_rho = float(_sr(f_means, d_means)[0])
    else:
        actual_rho = float('nan')

    result = {
        'degree': deg, 'align': align, 'headroom': headroom,
        'init_r2': init_r2, 'actual_rho': actual_rho
    }
    print(f"deg={deg:2d} align={align:.3f} headroom={headroom:.3f} ρ={actual_rho:.3f}", flush=True)
    return result

if __name__ == '__main__':
    from scipy.stats import spearmanr
    from sklearn.model_selection import LeaveOneOut
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    rng_data = np.random.RandomState(42)
    X = rng_data.uniform(-1, 1, (N, D)).astype(np.float32)
    w_dir = rng_data.randn(D)
    w_dir = w_dir / np.linalg.norm(w_dir)
    n_tr = N - 60

    # Reference model for init R^2 headroom measure
    ref_net = Net(D, 64)
    with torch.no_grad():
        nn.init.normal_(ref_net.fc1.weight, std=np.sqrt(2/D))
        init_H = ref_net.get_features(torch.FloatTensor(X[:n_tr]), 'cpu')
        init_H_te = ref_net.get_features(torch.FloatTensor(X[n_tr:]), 'cpu')

    # Parallel execution over degrees (each degree is independent)
    n_jobs = min(len(DEGREES), 10)
    print(f"Running {len(DEGREES)} degrees with {n_jobs} parallel workers", flush=True)
    all_points = Parallel(n_jobs=n_jobs)(
        delayed(run_degree)(deg, X, w_dir, n_tr, init_H, init_H_te, device) for deg in DEGREES
    )

    # Regression: rho ~ align + headroom + align:headroom
    valid = [p for p in all_points if not np.isnan(p['actual_rho'])]
    print(f"\nValid points: {len(valid)}", flush=True)
    in_r2, loocv_r2 = None, None
    if len(valid) >= 6:
        X_reg = np.array([[p['align'], p['headroom'], p['align'] * p['headroom']] for p in valid])
        y_reg = np.array([p['actual_rho'] for p in valid])
        lr = LinearRegression().fit(X_reg, y_reg)
        in_r2 = lr.score(X_reg, y_reg)
        print(f"\nRegression: rho = {lr.intercept_:.3f} + {lr.coef_[0]:.3f}*align + {lr.coef_[1]:.3f}*headroom + {lr.coef_[2]:.3f}*align*headroom", flush=True)
        print(f"In-sample R^2 = {in_r2:.3f}", flush=True)

        # LOOCV R^2
        loo = LeaveOneOut()
        preds, actuals = [], []
        for train_idx, test_idx in loo.split(X_reg):
            lr_loo = LinearRegression().fit(X_reg[train_idx], y_reg[train_idx])
            preds.append(lr_loo.predict(X_reg[test_idx])[0])
            actuals.append(y_reg[test_idx][0])
        from sklearn.metrics import r2_score
        loocv_r2 = r2_score(actuals, preds)
        print(f"LOOCV R^2 = {loocv_r2:.3f}", flush=True)

        # Simple Spearman: align vs rho
        rho_a, p_a = spearmanr([p['align'] for p in valid], [p['actual_rho'] for p in valid])
        print(f"\nAlign vs rho: ρ={rho_a:.3f}, p={p_a:.4f}", flush=True)
        # headroom vs rho
        rho_h, p_h = spearmanr([p['headroom'] for p in valid], [p['actual_rho'] for p in valid])
        print(f"Headroom vs rho: ρ={rho_h:.3f}, p={p_h:.4f}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, 'key_results', 'hermite_sweep_results.json')
    with open(out, 'w') as f:
        json.dump({'results': all_points, 'in_sample_r2': in_r2, 'loocv_r2': loocv_r2}, f, indent=2)
    print(f"\nSaved to {out}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)
