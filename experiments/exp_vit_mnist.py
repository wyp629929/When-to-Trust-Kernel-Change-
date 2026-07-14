"""
ViT-tiny on binary MNIST (0 vs 1): check dissociation extends to attention-based archs.
Extends from 3 to 5 seeds (5 embed dims x 5 seeds = 25 runs).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import json, time, warnings, os
warnings.filterwarnings('ignore')

SEEDS = 5
EMBED_DIMS = [64, 96, 128, 192, 256]
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

class ViTTiny(nn.Module):
    """Minimal ViT for small-scale experiments (no pos embed learned, fixed)."""
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64,
                 depth=4, num_heads=4, mlp_ratio=2.0, num_classes=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        n_patches = (img_size // patch_size) ** 2  # 16 for 28/7

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Fixed sinusoidal pos embedding
        pos = torch.zeros(1, n_patches, embed_dim)
        for p in range(n_patches):
            for d in range(0, embed_dim, 2):
                pos[0, p, d] = np.sin(p / (10000 ** (d / embed_dim)))
                if d+1 < embed_dim:
                    pos[0, p, d+1] = np.cos(p / (10000 ** ((d+1) / embed_dim)))
        self.register_buffer('pos_embed', pos)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, E, Hp, Wp
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)  # global avg pool
        if return_features:
            return x
        return self.head(x).flatten()

@torch.no_grad()
def compute_metrics(model, loader, device):
    """Run model on entire dataset, return features and predictions."""
    all_feats, all_preds, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        feats = model(x, return_features=True)
        preds = model.head(feats).flatten()
        all_feats.append(feats.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    return (torch.cat(all_feats).numpy(), torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy())

def compute_cka(K0, K1):
    """Centered Kernel Alignment."""
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    K0_c, K1_c = center(K0), center(K1)
    return (K0_c * K1_c).sum() / np.sqrt((K0_c**2).sum() * (K1_c**2).sum() + 1e-12)

def compute_frob(K0, K1):
    return float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)

if __name__ == '__main__':
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.datasets import MNIST

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Binary MNIST: digits 0 vs 1
    train_mnist = MNIST(root='/tmp/mnist', train=True, download=True)
    test_mnist = MNIST(root='/tmp/mnist', train=False, download=True)

    # Filter 0/1 and convert to float tensors
    idx_tr = (train_mnist.targets == 0) | (train_mnist.targets == 1)
    idx_te = (test_mnist.targets == 0) | (test_mnist.targets == 1)
    X_tr = train_mnist.data[idx_tr].unsqueeze(1).float() / 255.0
    y_tr = (train_mnist.targets[idx_tr] == 1).float()
    X_te = test_mnist.data[idx_te].unsqueeze(1).float() / 255.0
    y_te = (test_mnist.targets[idx_te] == 1).float()

    # Downsample train to ~3000 for speed
    rng_state = np.random.RandomState(42)
    n_tr = min(3000, len(X_tr))
    idx = rng_state.choice(len(X_tr), n_tr, replace=False)
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    train_dataset = TensorDataset(X_tr, y_tr)
    test_dataset = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}", flush=True)

    all_results = []
    for embed_dim in EMBED_DIMS:
        for seed in range(SEEDS):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            model = ViTTiny(embed_dim=embed_dim).to(device)
            opt = optim.AdamW(model.parameters(), lr=LR)

            # Initial features
            init_feats, _, _ = compute_metrics(model, test_loader, device)
            K0 = init_feats @ init_feats.T / embed_dim

            # Train
            for ep in range(N_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    nn.BCEWithLogitsLoss()(model(x), y).backward()
                    opt.step()

            # Trained features
            train_feats, preds, labels = compute_metrics(model, test_loader, device)
            K1 = train_feats @ train_feats.T / embed_dim

            cka = compute_cka(K0, K1)
            frob = compute_frob(K0, K1)
            err = nn.BCEWithLogitsLoss()(torch.from_numpy(preds), torch.from_numpy(labels)).item()
            # Baseline: always predict 0.5
            baseline_err = nn.BCEWithLogitsLoss()(torch.zeros_like(torch.from_numpy(labels)), torch.from_numpy(labels)).item()
            delta_err = baseline_err - err

            all_results.append({
                'embed_dim': embed_dim, 'seed': seed,
                'cka': cka, 'frob': frob, 'delta_err': delta_err
            })
            print(f"dim={embed_dim:3d} seed={seed} CKA={cka:.3f} Frob={frob:.1f}% ΔErr={delta_err:.3f}",
                  flush=True)

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, 'key_results', 'vit_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {out}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)

    # Compute correlations
    from scipy.stats import spearmanr
    frobs = np.array([r['frob'] for r in all_results])
    des = np.array([r['delta_err'] for r in all_results])
    ckas = np.array([r['cka'] for r in all_results])
    rho_f, p_f = spearmanr(frobs, des)
    rho_c, p_c = spearmanr(ckas, des)
    print(f"\nFrobenius vs ΔError: ρ={rho_f:.3f}, p={p_f:.4f}", flush=True)
    print(f"CKA vs ΔError: ρ={rho_c:.3f}, p={p_c:.4f}", flush=True)

    # Per-embed-dim means
    print("\nPer-dimension means:", flush=True)
    for dim in EMBED_DIMS:
        mask = np.array([r['embed_dim'] == dim for r in all_results])
        print(f"  dim={dim:3d}: CKA={ckas[mask].mean():.3f} Frob={frobs[mask].mean():.1f}% ΔErr={des[mask].mean():.3f}",
              flush=True)
