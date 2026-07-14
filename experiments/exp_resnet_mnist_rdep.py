"""
ResNet-18 on MNIST binary: full RDEP calibration.
4 channel widths x 5 seeds + null baseline + spectral analysis.
10 parallel workers. Tests RDEP on modern architecture (BN + skip connections).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time, os, warnings
from torchvision import datasets, transforms
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

SEEDS = 5
BASE_WIDTHS = [16, 24, 32, 48]
EPOCHS = 30; LR = 0.01; BATCH_SIZE = 128

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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)); out += self.shortcut(x)
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, base_width):
        super().__init__(); w = base_width; self.in_planes = w
        self.conv1 = nn.Conv2d(1, w, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(w)
        self.layer1 = self._make_layer(w, 2, 1); self.layer2 = self._make_layer(w*2, 2, 2)
        self.layer3 = self._make_layer(w*4, 2, 2); self.layer4 = self._make_layer(w*8, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(w*8, 1, bias=False)
        self.apply(lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') if isinstance(m, nn.Conv2d) else None)
    def _make_layer(self, planes, blocks, stride):
        s = [stride] + [1]*(blocks-1); layers = []
        for st in s: layers.append(BasicBlock(self.in_planes, planes, st)); self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]: x = l(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1)).flatten()
    def get_features(self, x):
        with torch.no_grad():
            x = torch.relu(self.bn1(self.conv1(x)))
            for l in [self.layer1, self.layer2, self.layer3, self.layer4]: x = l(x)
            return self.avgpool(x).view(x.size(0), -1).cpu().numpy()

def centered_kernel(H):
    K = H @ H.T; n = K.shape[0]; H_ = np.eye(n) - np.ones((n, n)) / n
    return H_ @ K @ H_

def cka(H0, H1):
    K0c, K1c = centered_kernel(H0), centered_kernel(H1)
    return float(np.sum(K0c * K1c) / (np.sqrt(np.sum(K0c**2) * np.sum(K1c**2)) + 1e-12))

def spectral_analysis(H, y):
    K = H @ H.T; evals = np.linalg.eigvalsh(K)[::-1]; evals = np.maximum(evals, 1e-12)
    p_eff = float(evals.sum() ** 2 / (evals ** 2).sum())
    _, evecs = np.linalg.eigh(K); evecs = evecs[:, ::-1]
    v = (evecs.T @ y) ** 2 / len(y); v = v / (v.sum() + 1e-12)
    return {'p_eff': p_eff, 'v_top3': float(v[:3].sum()), 'v_top5': float(v[:5].sum())}

def run(bw, seed, device, shuffle=False):
    torch.manual_seed(42 + seed); np.random.seed(42 + seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_all = datasets.MNIST('/tmp/data/MNIST', train=True, download=False, transform=transform)
    test_all = datasets.MNIST('/tmp/data/MNIST', train=False, download=False, transform=transform)
    # Filter 0/1
    for ds in [train_all, test_all]:
        t = ds.targets.numpy(); m = (t == 0) | (t == 1)
        ds.data, ds.targets = ds.data[m], t[m].tolist()
    rng = np.random.RandomState(42 + seed)
    train_idx = rng.choice(len(train_all), 2000, replace=False)
    test_idx = rng.choice(len(test_all), 500, replace=False)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_all, train_idx), BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_all, test_idx), BATCH_SIZE)

    model = ResNet(bw).to(device)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    y_list = []; [y_list.append(y.numpy()) for _, y in test_loader]
    y_all = np.concatenate(y_list).astype(float)
    y_all = (y_all - y_all.mean()) / y_all.std()
    if shuffle: np.random.RandomState(42 + seed).shuffle(y_all)

    model.eval()
    H0_list = [model.get_features(x.to(device)) for x, _ in test_loader]
    H0 = np.concatenate(H0_list)
    lr0 = Ridge(alpha=1.0).fit(H0, y_all); err0 = float(np.mean((lr0.predict(H0) - y_all)**2))
    y_pred0 = lr0.predict(H0)
    init_r2 = float(1 - np.sum((y_pred0 - y_all)**2) / np.sum((y_all - y_all.mean())**2))

    model.train()
    for ep in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad(); nn.MSELoss()(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    model.eval()
    H1_list = [model.get_features(x.to(device)) for x, _ in test_loader]
    H1 = np.concatenate(H1_list)
    lr1 = Ridge(alpha=1.0).fit(H1, y_all); err1 = float(np.mean((lr1.predict(H1) - y_all)**2))
    y_pred1 = lr1.predict(H1)
    post_r2 = float(1 - np.sum((y_pred1 - y_all)**2) / np.sum((y_all - y_all.mean())**2))

    cka_val = cka(H0, H1)
    K0, K1 = H0 @ H0.T / bw, H1 @ H1.T / bw
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    spec0 = spectral_analysis(H0, y_all)
    return {'base_width': bw, 'seed': seed, 'shuffled': shuffle, 'cka': cka_val, 'frob': frob,
            'delta_err': err0 - err1, 'init_r2': init_r2, 'post_r2': post_r2,
            'headroom': 1.0 - init_r2, 'lpg': post_r2 - init_r2,
            'p_eff_0': spec0['p_eff'], 'v_top3_0': spec0['v_top3']}

if __name__ == '__main__':
    t0 = time.time(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    # Pre-download to avoid thread contention
    datasets.MNIST('/tmp/data/MNIST', train=True, download=False, transform=transforms.ToTensor())
    datasets.MNIST('/tmp/data/MNIST', train=False, download=False, transform=transforms.ToTensor())
    print("MNIST ready", flush=True)
    configs = [(w, s, device, False) for w in BASE_WIDTHS for s in range(SEEDS)]
    configs += [(w, s, device, True) for w in BASE_WIDTHS for s in range(3)]
    print(f"Configs: {len(configs)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        fut = {pool.submit(run, *c): c for c in configs}
        for f in as_completed(fut):
            results.append(f.result()); c = fut[f]
            print(f"  w={c[0]} seed={c[1]} {'[NULL]' if c[3] else ''} ({time.time()-t0:.0f}s)", flush=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'key_results', 'resnet_mnist_results.json')
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"Saved to {out_path}", flush=True)

    normal = [r for r in results if not r['shuffled']]; widths = sorted(set(r['base_width'] for r in normal))
    means = {w: {k: float(np.mean([r[k] for r in normal if r['base_width'] == w])) for k in ['cka','frob','delta_err','init_r2','headroom','p_eff_0','v_top3_0']} for w in widths}
    print(f"\n{'='*60}\nResNet-18 on MNIST (0/1) — RDEP\n{'='*60}")
    print(f"{'w':>4} {'CKA':>8} {'Frob':>8} {'ΔErr':>8} {'hdrm':>8} {'p_eff':>8}")
    for w in widths: m = means[w]; print(f"{w:4d} {m['cka']:8.4f} {m['frob']:8.1f} {m['delta_err']:8.4f} {m['headroom']:8.4f} {m['p_eff_0']:8.1f}")
    f_m, d_m = np.array([means[w]['frob'] for w in widths]), np.array([means[w]['delta_err'] for w in widths])
    h_m = np.array([means[w]['headroom'] for w in widths])
    print(f"\nFrob vs ΔErr:  ρ={spearmanr(f_m, d_m)[0]:.3f}, p={spearmanr(f_m, d_m)[1]:.4f}")
    print(f"Headroom ΔErr: ρ={spearmanr(h_m, d_m)[0]:.3f}, p={spearmanr(h_m, d_m)[1]:.4f}")
    null = [r for r in results if r['shuffled']]
    if null:
        nf = np.mean([r['frob'] for r in null]); ne = np.mean([r['delta_err'] for r in null])
        print(f"\nNull: Frob={nf:.1f}% ΔErr={ne:.4f}")
    print(f"Total: {time.time()-t0:.0f}s")
