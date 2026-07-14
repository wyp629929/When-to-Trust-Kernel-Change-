"""Highfreq extended sweep: 11 widths (original 6 + 5 intermediate), 10 seeds each."""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time
from scipy.stats import spearmanr

SEEDS = 10
WIDTHS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
N, D, EPOCHS = 300, 10, 500

class Net(nn.Module):
    def __init__(self, d, w):
        super().__init__()
        self.fc1 = nn.Linear(d, w, bias=False)
        self.fc2 = nn.Linear(w, 1, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).flatten()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, configs: {len(WIDTHS)} widths x {SEEDS} seeds', flush=True)

    rng_data = np.random.RandomState(42)
    X = rng_data.uniform(-1, 1, (N, D)).astype(np.float32)
    f = np.sin(5*X[:,0]) + np.cos(7*X[:,1])
    y = (f - f.mean()) / f.std()
    n_tr = N - 60

    results = []
    t0 = time.time()
    for w in WIDTHS:
        frobs, des = [], []
        for s in range(SEEDS):
            torch.manual_seed(42 + s)
            rng = np.random.RandomState(42 + s)
            model = Net(D, w).to(device)
            nn.init.normal_(model.fc1.weight, std=np.sqrt(2/D))
            nn.init.normal_(model.fc2.weight, std=np.sqrt(2/w))

            X_t = torch.from_numpy(X[:n_tr]).float().to(device)
            y_t = torch.from_numpy(y[:n_tr] + 0.1*rng.randn(n_tr)).float().to(device)

            with torch.no_grad():
                H0 = torch.relu(model.fc1(X_t)).cpu().numpy()
            K0 = H0 @ H0.T / w
            reg = 0.01 * n_tr

            opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
            for ep in range(EPOCHS):
                opt.zero_grad()
                nn.MSELoss()(model(X_t), y_t).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            X_te = torch.from_numpy(X[n_tr:]).float().to(device)
            with torch.no_grad():
                H1 = torch.relu(model.fc1(X_t)).cpu().numpy()
            K1 = H1 @ H1.T / w
            with torch.no_grad():
                H1_te = torch.relu(model.fc1(X_te)).cpu().numpy()
            K0_te = (H1_te @ H1.T) / w
            pred0 = K0_te @ np.linalg.solve(K0 + reg*np.eye(n_tr), y[:n_tr])
            pred1 = K0_te @ np.linalg.solve(K1 + reg*np.eye(n_tr), y[:n_tr])
            de = float(np.mean((pred0 - y[n_tr:])**2) - np.mean((pred1 - y[n_tr:])**2))
            frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
            frobs.append(frob); des.append(de)

        f_mean = float(np.mean(frobs)); d_mean = float(np.mean(des))
        results.append({'width': w, 'frob': f_mean, 'delta_err': d_mean,
                        'frob_se': float(np.std(frobs)/np.sqrt(SEEDS)),
                        'de_se': float(np.std(des)/np.sqrt(SEEDS))})
        print(f'  w={w:4d} Frob={f_mean:.1f}% dErr={d_mean:.4f} ({time.time()-t0:.0f}s)', flush=True)

    f_means = np.array([r['frob'] for r in results])
    d_means = np.array([r['delta_err'] for r in results])
    rho, p = spearmanr(f_means, d_means)
    print(f'\nHighfreq extended ({len(WIDTHS)} widths): rho={rho:.3f}, p={p:.4f}', flush=True)
    for r in results:
        print(f'  w={r["width"]:4d}: Frob={r["frob"]:.1f}+-{r["frob_se"]:.1f}, dErr={r["delta_err"]:.4f}+-{r["de_se"]:.4f}', flush=True)

    with open('highfreq_extended_results.json', 'w') as f:
        json.dump({'widths': WIDTHS, 'results': results, 'rho': rho, 'p': p}, f, indent=2)
    print(f'Total: {time.time()-t0:.0f}s', flush=True)
