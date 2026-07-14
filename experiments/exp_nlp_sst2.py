"""
Enhanced Tiny Transformer on SST-2.
Adds: 5 seeds, null baseline per dim, spectral analysis (p_eff, v_i concentration).
Frames as cross-regime test of the headroom framework (Section 5.1.1).
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time, os, csv, warnings
from collections import Counter
from sklearn.linear_model import Ridge, LinearRegression
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

SEEDS = 5; HIDDEN_DIMS = [64, 96, 128, 192, 256]
NULL_SEEDS = 3  # shuffled-label null baseline per dim
EPOCHS = 20; LR = 0.001; BATCH_SIZE = 64; MAX_LEN = 48; VOCAB_SIZE = 5000; N_LAYERS = 4; N_HEADS = 4

def load_sst2(data_dir='/tmp/SST-2', max_train=3000):
    def read_tsv(path):
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            texts, labels = [], []
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0]); labels.append(int(row[1]))
        return texts, labels
    train_texts, train_labels = read_tsv(f'{data_dir}/train.tsv')
    train_texts = train_texts[:max_train]; train_labels = train_labels[:max_train]
    test_texts, test_labels = read_tsv(f'{data_dir}/dev.tsv')
    counter = Counter()
    for t in train_texts:
        counter.update(t.lower().split())
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (w, _) in enumerate(counter.most_common(VOCAB_SIZE - 2)):
        vocab[w] = i + 2
    def encode(text):
        tokens = text.lower().split()[:MAX_LEN]
        return [vocab.get(t, vocab['<unk>']) for t in tokens] + [vocab['<pad>']] * (MAX_LEN - len(tokens))
    X_tr = torch.LongTensor([encode(t) for t in train_texts])
    X_te = torch.LongTensor([encode(t) for t in test_texts])
    y_tr = torch.FloatTensor([(l - 0.5) * 2 for l in train_labels])
    y_te = torch.FloatTensor([(l - 0.5) * 2 for l in test_labels])
    return X_tr, X_te, y_tr, y_te

class TinyTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_LEN, d_model) * 0.02)
        self.ln_pre = nn.LayerNorm(d_model)
        el = nn.TransformerEncoderLayer(d_model, N_HEADS, d_model*4, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, N_LAYERS)
        self.ln_post = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1, bias=False)
        self.apply(lambda m: nn.init.normal_(m.weight, std=0.02) if hasattr(m, 'weight') and m.weight.dim() > 1 and not isinstance(m, nn.LayerNorm) else None)
    def forward(self, x):
        x = self.encoder(self.ln_pre(self.embed(x) + self.pos_embed))
        return self.head(self.ln_post(x[:, 0])).flatten()
    def get_features(self, x):
        with torch.no_grad():
            return self.ln_post(self.encoder(self.ln_pre(self.embed(x) + self.pos_embed)))[:, 0].cpu().numpy()

def centered_kernel(H):
    K = H @ H.T
    n = K.shape[0]
    H_ = np.eye(n) - np.ones((n, n)) / n
    return H_ @ K @ H_

def cka(H0, H1):
    K0c, K1c = centered_kernel(H0), centered_kernel(H1)
    return float(np.sum(K0c * K1c) / (np.sqrt(np.sum(K0c**2) * np.sum(K1c**2)) + 1e-12))

def spectral_analysis(H, y):
    """Compute effective rank and target projection concentration."""
    K = H @ H.T
    evals = np.linalg.eigvalsh(K)[::-1]
    evals = np.maximum(evals, 1e-12)
    p_eff = float(evals.sum() ** 2 / (evals ** 2).sum())
    # Target projections
    _, evecs = np.linalg.eigh(K)
    evecs = evecs[:, ::-1]
    v = (evecs.T @ y) ** 2 / len(y)
    v = v / (v.sum() + 1e-12)
    return {'p_eff': p_eff, 'v_top3': float(v[:3].sum()), 'v_top5': float(v[:5].sum()),
            'eigengap': float(evals[0] - evals[1]) if len(evals) > 1 else float('nan')}

def run(d_model, seed, device, X_tr, X_te, y_tr, y_te, shuffle=False):
    torch.manual_seed(42 + seed); np.random.seed(42 + seed)
    y_tr_np = y_tr.numpy().copy()
    y_te_np = y_te.numpy().copy()
    if shuffle:
        np.random.RandomState(42 + seed).shuffle(y_tr_np)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, torch.FloatTensor(y_tr_np)), BATCH_SIZE, shuffle=True)
    model = TinyTransformer(d_model).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    model.eval()
    H0 = model.get_features(X_te.to(device))
    lr0 = Ridge(alpha=1.0).fit(H0, y_te_np)
    err0 = float(np.mean((lr0.predict(H0) - y_te_np)**2))
    y_pred0 = lr0.predict(H0)
    init_r2 = float(1 - np.sum((y_pred0 - y_te_np)**2) / np.sum((y_te_np - y_te_np.mean())**2))

    model.train()
    for ep in range(EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); nn.MSELoss()(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    H1 = model.get_features(X_te.to(device))
    lr1 = Ridge(alpha=1.0).fit(H1, y_te_np)
    err1 = float(np.mean((lr1.predict(H1) - y_te_np)**2))
    y_pred1 = lr1.predict(H1)
    post_r2 = float(1 - np.sum((y_pred1 - y_te_np)**2) / np.sum((y_te_np - y_te_np.mean())**2))

    K0, K1 = H0 @ H0.T, H1 @ H1.T
    frob = float(np.linalg.norm(K1 - K0, 'fro') / (np.linalg.norm(K0, 'fro') + 1e-12) * 100)
    headroom = 1.0 - init_r2
    delta_r2 = post_r2 - init_r2
    lpg = delta_r2  # Linear Probe Gain

    spec0 = spectral_analysis(H0, y_te_np)
    spec1 = spectral_analysis(H1, y_te_np)

    return {
        'd_model': d_model, 'seed': seed, 'shuffled': shuffle,
        'cka': cka(H0, H1), 'frob': frob, 'delta_err': err0 - err1,
        'init_r2': init_r2, 'post_r2': post_r2, 'headroom': headroom,
        'lpg': lpg,
        'p_eff_0': spec0['p_eff'], 'p_eff_1': spec1['p_eff'],
        'v_top3_0': spec0['v_top3'], 'v_top5_0': spec0['v_top5'],
        'eigengap_0': spec0['eigengap'],
    }

def compute_dim_means(results):
    """Compute dimension-means and Spearman correlations (matching paper methodology)."""
    normal = [r for r in results if not r['shuffled']]
    dims = sorted(set(r['d_model'] for r in normal))
    means = {d: {} for d in dims}
    for d in dims:
        subset = [r for r in normal if r['d_model'] == d]
        for k in ['cka', 'frob', 'delta_err', 'init_r2', 'post_r2', 'headroom', 'p_eff_0', 'v_top3_0', 'lpg']:
            means[d][k] = float(np.mean([r[k] for r in subset]))
    return dims, means

if __name__ == '__main__':
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    all_data = load_sst2()

    configs = [(d, s, device, *all_data, False) for d in HIDDEN_DIMS for s in range(SEEDS)]
    configs += [(d, s, device, *all_data, True) for d in HIDDEN_DIMS for s in range(NULL_SEEDS)]
    print(f"Configs: {len(configs)} ({len(HIDDEN_DIMS)} dims x {SEEDS} seeds + {len(HIDDEN_DIMS)} dims x {NULL_SEEDS} null)",
          flush=True)

    results = []
    n_jobs = min(10, len(configs))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        fut = {pool.submit(run, *c): c for c in configs}
        for f in as_completed(fut):
            results.append(f.result()); c = fut[f]
            tag = " [NULL]" if c[-1] else ""
            print(f"  dim={c[0]} seed={c[1]}{tag} ({time.time()-t0:.0f}s)", flush=True)

    # Save raw results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'key_results', 'nlp_sst2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}", flush=True)

    # --- Analysis ---
    normal = [r for r in results if not r['shuffled']]
    null = [r for r in results if r['shuffled']]
    dims, means = compute_dim_means(results)
    f_means = np.array([means[d]['frob'] for d in dims])
    d_means = np.array([means[d]['delta_err'] for d in dims])
    c_means = np.array([means[d]['cka'] for d in dims])
    l_means = np.array([means[d]['lpg'] for d in dims])
    h_means = np.array([means[d]['headroom'] for d in dims])

    print(f"\n{'='*70}", flush=True)
    print("Tiny Transformer on SST-2 — Cross-Regime Test of Headroom Framework", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'dim':>5} {'CKA':>8} {'Frob%':>8} {'ΔErr':>8} {'initR2':>8} {'hdrm':>8} {'p_eff':>8} {'v_top3':>8}",
          flush=True)
    for d in dims:
        m = means[d]
        print(f"{d:5d} {m['cka']:8.4f} {m['frob']:8.1f} {m['delta_err']:8.4f} "
              f"{m['init_r2']:8.4f} {m['headroom']:8.4f} {m['p_eff_0']:8.1f} {m['v_top3_0']:8.3f}",
              flush=True)

    print(f"\n=== Config-mean Spearman (n={len(dims)} dims) ===", flush=True)
    # Frob vs ΔErr
    if len(set(f_means)) > 1 and len(set(d_means)) > 1:
        rf, pf = spearmanr(f_means, d_means)
        print(f"Frob vs ΔErr:   ρ={rf:.3f}, p={pf:.4f}", flush=True)
    # CKA vs ΔErr
    if len(set(c_means)) > 1 and len(set(d_means)) > 1:
        rc, pc = spearmanr(c_means, d_means)
        print(f"CKA vs ΔErr:    ρ={rc:.3f}, p={pc:.4f}", flush=True)
    # LPG vs ΔErr
    if len(set(l_means)) > 1 and len(set(d_means)) > 1:
        rl, pl = spearmanr(l_means, d_means)
        print(f"LPG vs ΔErr:    ρ={rl:.3f}, p={pl:.4f}", flush=True)
    # Headroom vs ΔErr (prediction: positive — cross-regime test of Section 5.1.1)
    if len(set(h_means)) > 1 and len(set(d_means)) > 1:
        rh, ph = spearmanr(h_means, d_means)
        print(f"Headroom vs ΔErr: ρ={rh:.3f}, p={ph:.4f}  ← cross-regime headroom test", flush=True)
    # Effective rank vs ΔErr
    p_means = np.array([means[d]['p_eff_0'] for d in dims])
    if len(set(p_means)) > 1 and len(set(d_means)) > 1:
        rp, pp = spearmanr(p_means, d_means)
        print(f"p_eff vs ΔErr:  ρ={rp:.3f}, p={pp:.4f}  ← diagnostic validation gap test", flush=True)

    # Null baseline
    null_frob = [r['frob'] for r in null]
    null_err = [r['delta_err'] for r in null]
    print(f"\n=== Null baseline (shuffled labels, n={len(null)}) ===", flush=True)
    print(f"Null Frob:  mean={np.mean(null_frob):.1f}%  range=[{min(null_frob):.1f}, {max(null_frob):.1f}]",
          flush=True)
    print(f"Null ΔErr:  mean={np.mean(null_err):.4f}  range=[{min(null_err):.4f}, {max(null_err):.4f}]",
          flush=True)
    # Compare null vs normal at each dim
    for d in HIDDEN_DIMS:
        normal_frob = [r['frob'] for r in normal if r['d_model'] == d]
        null_frob_d = [r['frob'] for r in null if r['d_model'] == d]
        normal_err = [r['delta_err'] for r in normal if r['d_model'] == d]
        null_err_d = [r['delta_err'] for r in null if r['d_model'] == d]
        print(f"  dim={d:3d}: normal Frob={np.mean(normal_frob):.1f}% null Frob={np.mean(null_frob_d):.1f}%  "
              f"normal ΔErr={np.mean(normal_err):.4f} null ΔErr={np.mean(null_err_d):.4f}", flush=True)

    # Per-dataset CKA blind spot check
    print(f"\n=== CKA blind spot (CKA vs Frobenius dissociation) ===", flush=True)
    for d in dims:
        m = means[d]
        print(f"  dim={d:3d}: CKA={m['cka']:.4f}  Frob={m['frob']:.1f}%  "
              f"p_eff={m['p_eff_0']:.0f}  v_top3={m['v_top3_0']:.3f}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
