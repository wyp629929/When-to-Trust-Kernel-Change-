"""
CNN + CIFAR-10 binary classification.
Tests whether the dissociation holds for deeper architectures on real images.

Architecture: 2-conv-layer CNN (mimics 2-layer MLP structure)
- Conv(3→W, 5) + ReLU + MaxPool(2)
- Conv(W→2W, 5) + ReLU + MaxPool(2)
- FC(2W*5*5 → 1)

Widths: [16, 32, 64, 128, 256] — scaled for CNN parameter count
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from representational_metrics import cka_from_features

SEEDS = 5
WIDTHS = [16, 32, 64, 128, 256]
EPOCHS = 100
LR = 0.01

class CNN(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.conv1 = nn.Conv2d(3, W, 5, bias=False)
        self.conv2 = nn.Conv2d(W, 2*W, 5, bias=False)
        self.fc = nn.Linear(2*W * 5 * 5, 1, bias=False)
        nn.init.normal_(self.conv1.weight, std=np.sqrt(2.0 / (3 * 5 * 5)))
        nn.init.normal_(self.conv2.weight, std=np.sqrt(2.0 / (W * 5 * 5)))
        nn.init.normal_(self.fc.weight, std=np.sqrt(2.0 / (2*W * 5 * 5)))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x).flatten()

    def get_features(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            return x.view(x.size(0), -1).cpu().numpy()

def run_one(W, seed, device):
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)

    # CIFAR-10 binary: classes 0 (airplane) and 1 (automobile)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)

    # Binary: classes 0 and 1
    def filter_binary(dataset, cls1=0, cls2=1):
        targets = np.array(dataset.targets)
        mask = (targets == cls1) | (targets == cls2)
        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask].tolist()
        # Convert to ±1
        dataset.targets = [1 if t == cls2 else -1 for t in dataset.targets]
        return dataset

    train_set = filter_binary(train_set)
    test_set = filter_binary(test_set)

    # Subset for speed (1000 train, 500 test)
    rng = np.random.RandomState(42 + seed)
    n_tr = 1000
    n_te = 500

    idx_tr = rng.choice(len(train_set), n_tr, replace=False)
    idx_te = rng.choice(len(test_set), n_te, replace=False)

    X_tr = torch.stack([train_set[i][0] for i in idx_tr]).to(device)
    y_tr = torch.FloatTensor([train_set[i][1] for i in idx_tr]).to(device)
    X_te = torch.stack([test_set[i][0] for i in idx_te]).to(device)
    y_te = torch.FloatTensor([test_set[i][1] for i in idx_te]).to(device)

    # Standardize
    y_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-8)
    y_te = (y_te - y_te.mean()) / (y_te.std() + 1e-8)
    y_tr_np = y_tr.cpu().numpy()

    m = CNN(W).to(device)
    opt = optim.SGD(m.parameters(), lr=LR, momentum=0.9)

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(X_tr), y_tr).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T

    cka_val = float(cka_from_features(H0, H1))
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)

    n_tr_actual = len(y_tr_np)
    # Linear probe: train ridge regression on features (separate train/test)
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    n_tr_lp = min(500, n_tr_actual)
    n_te_lp = min(500, len(X_te))
    H0_np = H0[:n_tr_lp]
    H1_np = H1[:n_tr_lp]
    y_tr_lp = y_tr_np[:n_tr_lp]
    H_te_np = m.get_features(X_te[:n_te_lp])
    y_te_lp = y_te.cpu().numpy()[:n_te_lp]

    # R² using initial features
    reg_lp = Ridge(alpha=1.0, fit_intercept=True)
    reg_lp.fit(H0_np, y_tr_lp)
    r2_init = float(r2_score(y_te_lp, reg_lp.predict(H_te_np)))

    # R² using final features
    reg_lp.fit(H1_np, y_tr_lp)
    r2_final = float(r2_score(y_te_lp, reg_lp.predict(H_te_np)))

    delta_r2 = r2_final - r2_init

    return {'width': int(W), 'seed': int(seed),
            'cka': cka_val, 'frob': frob, 'delta_r2': delta_r2}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = []
    total = len(WIDTHS) * SEEDS
    done = 0
    t0 = time.time()

    for W in WIDTHS:
        for s in range(SEEDS):
            r = run_one(W, s, device)
            all_results.append(r)
            done += 1
            elapsed = time.time() - t0
            print(f"[{done}/{total}] W={W} seed={s}: CKA={r['cka']:.4f} Frob={r['frob']:.1f}% ΔR²={r['delta_r2']:+.4f} {elapsed:.0f}s")

    out = os.path.expanduser('~/cnn_cifar_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f)
    print(f"\n{total} runs complete. Saved to {out} [{time.time()-t0:.0f}s]")

    for W in WIDTHS:
        pts = [r for r in all_results if r['width'] == W]
        if pts:
            print(f"W={W}: CKA={np.mean([r['cka'] for r in pts]):.4f} Frob={np.mean([r['frob'] for r in pts]):.1f}% ΔR²={np.mean([r['delta_r2'] for r in pts]):+.4f}")
