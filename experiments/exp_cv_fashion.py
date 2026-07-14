"""
Combined GPU experiment:
1. Cross-validated Δalign for poly width=64 (5 seeds × 5 folds)
2. FashionMNIST train/test error curves for width=64 (5 seeds)
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os, sys
from torchvision import datasets, transforms
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N, EPOCHS = 300, 500
WIDTH = 64

# ====== Shared Model ======
class Net(nn.Module):
    def __init__(self, d_in, width):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x))).flatten()
    def get_features(self, x):
        with torch.no_grad(): return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()

# ====== Part 1: Cross-validated Δalign ======
def run_cv():
    print("=== Cross-validated Δalign (poly w=64) ===")
    SEEDS, K = 5, 5
    all_folds = []
    t0 = time.time()

    for s in range(SEEDS):
        rng = np.random.RandomState(42 + s)
        Xall = rng.uniform(-1, 1, (N, 10)).astype(np.float32)
        yall = Xall[:, 0]**2 + Xall[:, 1]
        yall = (yall - yall.mean()) / yall.std()
        n_total = N - 60  # 240

        idx = rng.permutation(n_total)
        fold_size = n_total // K

        for fold in range(K):
            test_idx = idx[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(idx, test_idx)
            n_tr = len(train_idx)

            X_tr = Xall[train_idx]; y_tr = yall[train_idx]
            X_te = Xall[test_idx]; y_te = yall[test_idx]
            y_tr_noisy = y_tr + 0.1 * rng.randn(n_tr)

            m = Net(10, WIDTH)
            torch.manual_seed(42 + s)
            nn.init.normal_(m.fc1.weight, std=np.sqrt(2/10))
            nn.init.normal_(m.fc2.weight, std=np.sqrt(2/WIDTH))
            opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
            xt = torch.FloatTensor(X_tr)
            yt = torch.FloatTensor(y_tr_noisy)

            H0 = m.get_features(X_tr)
            K0 = H0 @ H0.T / WIDTH
            evals0, evecs0 = np.linalg.eigh(K0)
            evals0 = evals0[::-1]; evecs0 = evecs0[:, ::-1]

            for ep in range(EPOCHS):
                opt.zero_grad()
                nn.MSELoss()(m(xt), yt).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

            H1 = m.get_features(X_tr)
            K1 = H1 @ H1.T / WIDTH
            evalsT, _ = np.linalg.eigh(K1)
            evalsT = evalsT[::-1]

            proj = evecs0.T @ y_tr
            vi = (proj ** 2) / n_tr
            dalign = np.sum(vi * (evalsT - evals0)) / (np.sum(vi * evals0) + 1e-12)

            K0_te = (m.get_features(X_te) @ H1.T) / WIDTH
            pred0 = K0_te @ np.linalg.solve(K0 + 0.01 * n_tr * np.eye(n_tr), y_tr)
            pred1 = K0_te @ np.linalg.solve(K1 + 0.01 * n_tr * np.eye(n_tr), y_tr)
            gain = float(np.mean((pred0 - y_te)**2) - np.mean((pred1 - y_te)**2))

            elapsed = time.time() - t0
            all_folds.append({'seed': s, 'fold': fold, 'dalign': float(dalign), 'gain': gain})
            print(f"  CV seed={s} fold={fold}: Δalign={dalign:.4f} gain={gain:.4f} [{elapsed:.0f}s]")

    from scipy.stats import spearmanr
    daligns = [r['dalign'] for r in all_folds]
    gains = [r['gain'] for r in all_folds]
    rho, p = spearmanr(daligns, gains)
    print(f"CV results: ρ(Δalign, Gain) = {rho:.4f}, p={p:.4e} (n={len(all_folds)})")
    return {'rho': float(rho), 'p': float(p), 'n': len(all_folds), 'folds': all_folds}

# ====== Part 2: FashionMNIST train/test curves ======
def run_fashion():
    print("\n=== FashionMNIST train/test curves (w=64) ===")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_set = datasets.FashionMNIST(root='/tmp/data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='/tmp/data', train=False, download=True, transform=transform)

    # Binary: top vs sandal (classes 0 and 5)
    def filter_binary(dataset, cls1=0, cls2=5):
        targets = np.array(dataset.targets)
        mask = (targets == cls1) | (targets == cls2)
        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask].tolist()
        return dataset

    train_set = filter_binary(train_set)
    test_set = filter_binary(test_set)

    SEEDS = 5
    all_curves = []

    for s in range(SEEDS):
        rng = np.random.RandomState(42 + s)
        n_tr, n_te = 200, 100
        idx_tr = rng.choice(len(train_set), n_tr, replace=False)
        idx_te = rng.choice(len(test_set), n_te, replace=False)

        X_tr = np.stack([train_set[i][0].numpy() for i in idx_tr]).astype(np.float32)
        y_tr_raw = np.array([train_set[i][1] for i in idx_tr], dtype=np.float32)
        X_te = np.stack([test_set[i][0].numpy() for i in idx_te]).astype(np.float32)
        y_te_raw = np.array([test_set[i][1] for i in idx_te], dtype=np.float32)

        y_tr_all = (y_tr_raw - y_tr_raw.mean()) / y_tr_raw.std()
        y_te_all = (y_te_raw - y_te_raw.mean()) / y_te_raw.std()

        m = Net(784, WIDTH).to(DEVICE)
        torch.manual_seed(42 + s)
        nn.init.normal_(m.fc1.weight, std=np.sqrt(2/784))
        nn.init.normal_(m.fc2.weight, std=np.sqrt(2/WIDTH))
        opt = optim.SGD(m.parameters(), lr=0.05, momentum=0.9)

        train_errors, test_errors, epoches = [], [], []
        for ep in range(EPOCHS):
            m.train()
            opt.zero_grad()
            batch_idx = rng.choice(n_tr, min(64, n_tr), replace=False)
            xb = torch.FloatTensor(X_tr[batch_idx]).to(DEVICE)
            yb = torch.FloatTensor(y_tr_all[batch_idx]).to(DEVICE)
            loss = nn.MSELoss()(m(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

            if ep % 50 == 0 or ep == EPOCHS - 1:
                m.eval()
                with torch.no_grad():
                    pred_train = m(torch.FloatTensor(X_tr).to(DEVICE)).cpu().numpy()
                    pred_test = m(torch.FloatTensor(X_te).to(DEVICE)).cpu().numpy()
                train_mse = float(np.mean((pred_train - y_tr_all)**2))
                test_mse = float(np.mean((pred_test - y_te_all)**2))
                train_errors.append(train_mse); test_errors.append(test_mse); epoches.append(ep)

        all_curves.append({'seed': s, 'epochs': epoches,
                           'train_mse': train_errors, 'test_mse': test_errors})
        print(f"  Fashion seed={s}: final train MSE={train_errors[-1]:.4f}, test MSE={test_errors[-1]:.4f}")

    return {'curves': all_curves}

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    cv_results = run_cv()
    fashion_results = run_fashion()
    out = {'cv': cv_results, 'fashion': fashion_results}
    save_path = os.path.expanduser('~/cv_fashion_results.json')
    with open(save_path, 'w') as f:
        json.dump(out, f)
    print(f"\nAll done. Saved to {save_path}")
