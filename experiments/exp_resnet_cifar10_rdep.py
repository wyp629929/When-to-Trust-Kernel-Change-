"""
ResNet-18 on CIFAR-10 binary: full RDEP calibration for medium-scale vision.
4 channel widths x 5 seeds + null baseline + spectral analysis + headroom.
10 parallel workers. Directly addresses JMLR 雷区二.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time, os, warnings
from torchvision import datasets, transforms
from sklearn.linear_model import Ridge, LinearRegression
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

SEEDS = 5
BASE_WIDTHS = [32, 48, 64, 96]
EPOCHS = 50
LR = 0.01
BATCH_SIZE = 128

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, base_width, num_classes=1):
        super().__init__()
        w = base_width
        self.in_planes = w
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = self._make_layer(w, 2, stride=1)
        self.layer2 = self._make_layer(w*2, 2, stride=2)
        self.layer3 = self._make_layer(w*4, 2, stride=2)
        self.layer4 = self._make_layer(w*8, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w*8, 1, bias=False)
        self.apply(lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   if isinstance(m, nn.Conv2d) else None)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x).flatten()
    def get_features(self, x):
        with torch.no_grad():
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x); x = self.layer2(x)
            x = self.layer3(x); x = self.layer4(x)
            return self.avgpool(x).view(x.size(0), -1).cpu().numpy()

def centered_kernel(H):
    K = H @ H.T; n = K.shape[0]; H_ = np.eye(n) - np.ones((n, n)) / n
    return H_ @ K @ H_

def cka(H0, H1):
    K0c, K1c = centered_kernel(H0), centered_kernel(H1)
    return float(np.sum(K0c * K1c) / (np.sqrt(np.sum(K0c**2) * np.sum(K1c**2)) + 1e-12))

def spectral_analysis(H, y):
    K = H @ H.T
    evals = np.linalg.eigvalsh(K)[::-1]
    evals = np.maximum(evals, 1e-12)
    p_eff = float(evals.sum() ** 2 / (evals ** 2).sum())
    _, evecs = np.linalg.eigh(K)
    evecs = evecs[:, ::-1]
    v = (evecs.T @ y) ** 2 / len(y)
    v = v / (v.sum() + 1e-12)
    return {'p_eff': p_eff, 'v_top3': float(v[:3].sum()), 'v_top5': float(v[:5].sum())}

def run(base_width, seed, device, shuffle=False):
    torch.manual_seed(42 + seed); np.random.seed(42 + seed)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_all = datasets.CIFAR10('/tmp/data', train=True, download=False, transform=transform)
    test_all = datasets.CIFAR10('/tmp/data', train=False, download=False, transform=transform)

    # Binary: class 0 vs 1
    for ds in [train_all, test_all]:
        targets = np.array(ds.targets)
        mask = (targets == 0) | (targets == 1)
        ds.data, ds.targets = ds.data[mask], targets[mask].tolist()

    rng = np.random.RandomState(42 + seed)
    train_idx = rng.choice(len(train_all), 3000, replace=False)
    test_idx = rng.choice(len(test_all), 1000, replace=False)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_all, train_idx), BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_all, test_idx), BATCH_SIZE)

    model = ResNet(base_width).to(device)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # Labels
    y_all_list = []
    for _, y in test_loader:
        y_all_list.append(y.numpy())
    y_all = np.concatenate(y_all_list).astype(float)
    y_all = (y_all - y_all.mean()) / y_all.std()

    if shuffle:
        np.random.RandomState(42 + seed).shuffle(y_all)

    # Initial features + linear probe
    model.eval()
    H0_list = []
    for x, _ in test_loader:
        H0_list.append(model.get_features(x.to(device)))
    H0 = np.concatenate(H0_list)
    lr0 = Ridge(alpha=1.0).fit(H0, y_all)
    err0 = float(np.mean((lr0.predict(H0) - y_all)**2))
    y_pred0 = lr0.predict(H0)
    init_r2 = float(1 - np.sum((y_pred0 - y_all)**2) / np.sum((y_all - y_all.mean())**2))

    # Train
    model.train()
    for ep in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            nn.MSELoss()(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # Post features
    model.eval()
    H1_list = []
    for x, _ in test_loader:
        H1_list.append(model.get_features(x.to(device)))
    H1 = np.concatenate(H1_list)
    lr1 = Ridge(alpha=1.0).fit(H1, y_all)
    err1 = float(np.mean((lr1.predict(H1) - y_all)**2))
    y_pred1 = lr1.predict(H1)
    post_r2 = float(1 - np.sum((y_pred1 - y_all)**2) / np.sum((y_all - y_all.mean())**2))

    cka_val = cka(H0, H1)
    K0, K1 = H0 @ H0.T / base_width, H1 @ H1.T / base_width
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    headroom = 1.0 - init_r2
    lpg = post_r2 - init_r2
    spec0 = spectral_analysis(H0, y_all)

    return {'base_width': base_width, 'seed': seed, 'shuffled': shuffle,
            'cka': cka_val, 'frob': frob, 'delta_err': err0 - err1,
            'init_r2': init_r2, 'post_r2': post_r2, 'headroom': headroom,
            'lpg': lpg, 'p_eff_0': spec0['p_eff'], 'v_top3_0': spec0['v_top3']}

if __name__ == '__main__':
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Pre-download to avoid thread contention
    print("Downloading CIFAR-10...", flush=True)
    datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transforms.ToTensor())
    datasets.CIFAR10('/tmp/data', train=False, download=True, transform=transforms.ToTensor())
    print("Done.", flush=True)

    configs = [(w, s, device, False) for w in BASE_WIDTHS for s in range(SEEDS)]
    configs += [(w, s, device, True) for w in BASE_WIDTHS for s in range(3)]
    print(f"Configs: {len(configs)}", flush=True)

    results = []
    n_jobs = min(10, len(configs))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        fut = {pool.submit(run, *c): c for c in configs}
        for f in as_completed(fut):
            results.append(f.result()); c = fut[f]
            tag = " [NULL]" if c[3] else ""
            print(f"  w={c[0]} seed={c[1]}{tag} ({time.time()-t0:.0f}s)", flush=True)

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'key_results', 'resnet_cifar10_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}", flush=True)

    # Analysis
    normal = [r for r in results if not r['shuffled']]
    null = [r for r in results if r['shuffled']]
    widths = sorted(set(r['base_width'] for r in normal))
    means = {}
    for w in widths:
        sub = [r for r in normal if r['base_width'] == w]
        means[w] = {k: float(np.mean([r[k] for r in sub]))
                    for k in ['cka', 'frob', 'delta_err', 'init_r2', 'headroom', 'p_eff_0', 'v_top3_0', 'lpg']}

    print(f"\n{'='*70}", flush=True)
    print("ResNet-18 on CIFAR-10 binary — RDEP Calibration", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'width':>6} {'CKA':>8} {'Frob%':>8} {'ΔErr':>8} {'initR2':>8} {'hdrm':>8} {'p_eff':>8} {'v_top3':>8}",
          flush=True)
    for w in widths:
        m = means[w]
        print(f"{w:6d} {m['cka']:8.4f} {m['frob']:8.1f} {m['delta_err']:8.4f} "
              f"{m['init_r2']:8.4f} {m['headroom']:8.4f} {m['p_eff_0']:8.1f} {m['v_top3_0']:8.3f}",
              flush=True)

    f_means = np.array([means[w]['frob'] for w in widths])
    d_means = np.array([means[w]['delta_err'] for w in widths])
    c_means = np.array([means[w]['cka'] for w in widths])
    h_means = np.array([means[w]['headroom'] for w in widths])
    p_means = np.array([means[w]['p_eff_0'] for w in widths])

    print(f"\n=== Config-mean Spearman (n={len(widths)} widths) ===", flush=True)
    rf, pf = spearmanr(f_means, d_means)
    print(f"Frob vs ΔErr:   ρ={rf:.3f}, p={pf:.4f}", flush=True)
    rc, pc = spearmanr(c_means, d_means)
    print(f"CKA vs ΔErr:    ρ={rc:.3f}, p={pc:.4f}", flush=True)
    rh, ph = spearmanr(h_means, d_means)
    print(f"Headroom vs ΔErr: ρ={rh:.3f}, p={ph:.4f}", flush=True)
    rp, pp = spearmanr(p_means, d_means)
    print(f"p_eff vs ΔErr:  ρ={rp:.3f}, p={pp:.4f}", flush=True)

    # Null
    null_frob = [r['frob'] for r in null]
    null_err = [r['delta_err'] for r in null]
    print(f"\n=== Null baseline (shuffled, n={len(null)}) ===", flush=True)
    print(f"Null Frob: mean={np.mean(null_frob):.1f}%  range=[{min(null_frob):.1f}, {max(null_frob):.1f}]", flush=True)
    print(f"Null ΔErr: mean={np.mean(null_err):.4f}  range=[{min(null_err):.4f}, {max(null_err):.4f}]", flush=True)
    for w in widths:
        nf = [r['frob'] for r in null if r['base_width'] == w]
        nd = [r['delta_err'] for r in null if r['base_width'] == w]
        norm_f = np.mean([r['frob'] for r in normal if r['base_width'] == w])
        norm_d = np.mean([r['delta_err'] for r in normal if r['base_width'] == w])
        print(f"  w={w:3d}: normal Frob={norm_f:.1f}% null Frob={np.mean(nf):.1f}%  "
              f"normal ΔErr={norm_d:.4f} null ΔErr={np.mean(nd):.4f}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
