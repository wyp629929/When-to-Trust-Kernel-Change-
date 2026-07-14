"""
Pre-trained model RDEP: fine-tune ImageNet pre-trained models on CIFAR-100.
3 architectures x 2 fine-tune configs + null baseline.
Tests RDEP on the pre-training regime (different from random init).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time, os, warnings
from torchvision import datasets, transforms, models
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

ARCHS = {
    'resnet18': (models.resnet18, 512),
    'resnet34': (models.resnet34, 512),
    'vit_b_16': (models.vit_b_16, 768),
}
EPOCHS = 30; LR = 0.001; BATCH_SIZE = 256
torch.backends.cudnn.benchmark = True
N_TRAIN = 10000
N_TEST = 1000

IMG_SIZE = 64
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
# ViT needs 224x224
IMG_SIZE_VIT = 224
transform_train_vit = transforms.Compose([
    transforms.Resize(IMG_SIZE_VIT), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
transform_test_vit = transforms.Compose([
    transforms.Resize(IMG_SIZE_VIT),
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])

def cka(H0, H1):
    K0, K1 = H0 @ H0.T, H1 @ H1.T; n = K0.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K0c, K1c = H @ K0 @ H, H @ K1 @ H
    return float(np.sum(K0c * K1c) / (np.sqrt(np.sum(K0c**2) * np.sum(K1c**2)) + 1e-12))

def spectral_analysis(H, y_onehot):
    K = H @ H.T
    evals = np.linalg.eigvalsh(K)[::-1]
    evals = np.maximum(evals, 1e-12)
    p_eff = float(evals.sum() ** 2 / (evals ** 2).sum())
    _, evecs = np.linalg.eigh(K); evecs = evecs[:, ::-1]
    y_flat = y_onehot.mean(axis=1)
    v = (evecs.T @ y_flat) ** 2 / len(y_flat)
    v = v / (v.sum() + 1e-12)
    return {'p_eff': p_eff, 'v_top3': float(v[:3].sum())}

def get_features(model, loader, device, arch_name):
    """Extract penultimate-layer features (before final FC)."""
    model.eval(); feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if arch_name.startswith('vit'):
                f = model._process_input(x)
                n = x.shape[0]
                b = model.class_token.expand(n, -1, -1)
                f = torch.cat([b, f], dim=1)
                f = model.encoder(f)
                f = f[:, 0]  # CLS token embedding
            elif arch_name.startswith('resnet'):
                # Forward through conv layers (skip FC)
                f = model.conv1(x)
                f = model.bn1(f); f = model.relu(f); f = model.maxpool(f)
                f = model.layer1(f); f = model.layer2(f)
                f = model.layer3(f); f = model.layer4(f)
                f = model.avgpool(f); f = torch.flatten(f, 1)
            else:
                f = model(x)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)

def run(arch_name, lr, seed, device, shuffle=False):
    torch.manual_seed(42 + seed); np.random.seed(42 + seed)

    is_vit = arch_name.startswith('vit')
    tr_transform = transform_train_vit if is_vit else transform_train
    te_transform = transform_test_vit if is_vit else transform_test
    train_all = datasets.CIFAR100('/tmp/data', train=True, download=False, transform=tr_transform)
    test_all = datasets.CIFAR100('/tmp/data', train=False, download=False, transform=te_transform)

    rng = np.random.RandomState(42 + seed)
    n_tr_avail = min(N_TRAIN, len(train_all))
    n_te_avail = min(N_TEST, len(test_all))
    train_idx = rng.choice(len(train_all), n_tr_avail, replace=False)
    test_idx = rng.choice(len(test_all), n_te_avail, replace=False)

    train_subset = torch.utils.data.Subset(train_all, train_idx)
    test_subset = torch.utils.data.Subset(test_all, test_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset, BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_subset, BATCH_SIZE,
        num_workers=8, pin_memory=True)

    # Load pre-trained model
    builder, feat_dim = ARCHS[arch_name]
    if arch_name.startswith('vit'):
        model = builder(weights='IMAGENET1K_V1')
        model.head = nn.Linear(feat_dim, 100)
    else:
        model = builder(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(feat_dim, 100)

    model = model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    # Initial features
    H0, y_all = get_features(model, test_loader, device, arch_name)
    # For multi-class, we use the features before the final classification layer
    y_onehot = np.zeros((len(y_all), 100))
    y_onehot[np.arange(len(y_all)), y_all] = 1

    # Multi-output Ridge: fit all 100 classes at once
    lr0 = Ridge(alpha=1.0).fit(H0, y_onehot)
    y_pred0 = lr0.predict(H0)
    err0 = float(np.mean((y_pred0 - y_onehot) ** 2))
    init_r2 = float(1 - np.sum((y_pred0 - y_onehot)**2) / np.sum((y_onehot - y_onehot.mean())**2))

    if shuffle:
        np.random.RandomState(42 + seed).shuffle(y_all)
        y_onehot_shuf = np.zeros((len(y_all), 100))
        y_onehot_shuf[np.arange(len(y_all)), y_all] = 1
        y_onehot = y_onehot_shuf
        # Recreate train loader with shuffled labels
        X_tr = torch.stack([train_all[i][0] for i in train_idx.tolist()])
        y_tr_shuf = torch.from_numpy(y_all.astype(np.int64))
        train_shuf_ds = torch.utils.data.TensorDataset(X_tr, y_tr_shuf)
        train_loader = torch.utils.data.DataLoader(
            train_shuf_ds, BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True)

    # Fine-tune
    model.train()
    for ep in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

    # Post fine-tune features
    H1, _ = get_features(model, test_loader, device, arch_name)
    lr1 = Ridge(alpha=1.0).fit(H1, y_onehot)
    y_pred1 = lr1.predict(H1)
    err1 = float(np.mean((y_pred1 - y_onehot) ** 2))
    post_r2 = float(1 - np.sum((y_pred1 - y_onehot)**2) / np.sum((y_onehot - y_onehot.mean())**2))

    cka_val = cka(H0, H1)
    K0, K1 = H0 @ H0.T / feat_dim, H1 @ H1.T / feat_dim
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    spec0 = spectral_analysis(H0, y_onehot)

    return {'arch': arch_name, 'seed': seed, 'shuffled': shuffle, 'lr': lr,
            'cka': cka_val, 'frob': frob, 'delta_err': err0 - err1,
            'init_r2': init_r2, 'post_r2': post_r2,
            'headroom': 1.0 - init_r2, 'lpg': post_r2 - init_r2,
            'p_eff_0': spec0['p_eff'], 'v_top3_0': spec0['v_top3']}

if __name__ == '__main__':
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Pre-download CIFAR-100
    datasets.CIFAR100('/tmp/data', train=True, download=True, transform=transforms.ToTensor())
    datasets.CIFAR100('/tmp/data', train=False, download=True, transform=transforms.ToTensor())

    configs = []
    for arch in ['resnet18', 'resnet34']:
        configs += [(arch, 0.001, s, device, False) for s in range(3)]
        configs += [(arch, 0.01, s, device, False) for s in range(2)]
    configs += [('vit_b_16', 1e-5, s, device, False) for s in range(1)]  # ViT needs smaller LR + 224x224

    print(f"Configs: {len(configs)}", flush=True)
    results = []
    # Sequential execution avoids CUDA context issues
    for c in configs:
        try:
            r = run(*c)
            results.append(r)
            tag = " [NULL]" if c[4] else ""
            print(f"  {c[0]} lr={c[1]} seed={c[2]}{tag} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {c[0]} lr={c[1]} seed={c[2]} FAILED: {e} ({time.time()-t0:.0f}s)", flush=True)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'key_results', 'pretrained_rdep_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # Analysis
    normal = [r for r in results if not r['shuffled']]
    print(f"\nResults ({len(normal)} normal runs):")
    for r in normal:
        print(f"  {r['arch']:<12} lr={r['lr']:.4f} CKA={r['cka']:.3f} "
              f"Frob={r['frob']:.1f}% ΔErr={r['delta_err']:.3f} hdrm={r['headroom']:.3f}")

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
