"""
Width sweep: kernel stability vs generalization across width

W = [32, 64, 128, 256, 512, 1024]
每个宽度的动态跟踪：test error, CKA, 特征值, v_i
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings, os, time
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = 'cpu'


class TwoLayerNet(nn.Module):
    def __init__(self, d_in, width):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=np.sqrt(2.0 / d_in))
        nn.init.normal_(self.fc2.weight, std=np.sqrt(2.0 / width))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).flatten()

    def get_features(self, x):
        with torch.no_grad():
            return torch.relu(self.fc1(torch.FloatTensor(x))).numpy()


def cka(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return np.sum(Kc * Lc) / (np.sqrt(np.sum(Kc**2) * np.sum(Lc**2)) + 1e-12)

# ============================================================
# 三个数据条件
# ============================================================

def make_data(name, n, d, rng):
    X = rng.uniform(-1, 1, (n, d)).astype(np.float32)
    if name == 'poly':
        y = X[:, 0]**2 + X[:, 1]
    elif name == 'highfreq':
        y = np.sin(5*X[:, 0]) + np.cos(7*X[:, 1])
    elif name == 'gmm':
        X[:n//2] = rng.randn(n//2, d)*0.5 + 1.0
        X[n//2:] = rng.randn(n-n//2, d)*0.5 - 1.0
        y = np.array([1]*(n//2) + [0]*(n-n//2))
    else:
        raise ValueError(name)
    y = (y - y.mean()) / y.std()
    return X, y


# ============================================================
# 主实验：一个 (width, data) 组合
# ============================================================

def run(width, data_name, n=300, d=10, noise=0.1, lr=0.05, epochs=500,
         val_frac=0.2):
    """返回每个 checkpoints 的指标"""
    # 数据
    rng = np.random.RandomState(SEED + hash(data_name) % 100000)
    X, y = make_data(data_name, n, d, rng)
    y_train = y + noise * np.random.randn(n)

    # 分割 validation
    n_val = int(n * val_frac)
    n_tr = n - n_val
    X_tr, X_val = X[:n_tr], X[n_tr:]
    y_tr, y_val = y[:n_tr], y[n_tr:]
    y_train_tr = y_train[:n_tr]
    y_train_val = y_train[n_tr:]

    # 模型
    model = TwoLayerNet(d, width)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    x_t = torch.FloatTensor(X_tr)
    y_t = torch.FloatTensor(y_train_tr)

    # checkpoints
    cps = [0, 10, 20, 50, 100, 200, 500]

    results = {'epochs': [], 'test_err': [], 'cka_kk': [], 'cka_kl': [],
               'frob_diff': [], 'top_eig_ratio': [], 'align_total': []}

    def measure(ep):
        H = model.get_features(X)
        K = (H @ H.T) / width
        ev, evec = eigh(K)
        ev = ev[::-1]; evec = evec[:, ::-1]
        v = (evec.T @ y)**2 / n
        L = np.outer(y, y)

        # test error on validation clean
        pred_val = K[n_tr:, :n_tr] @ np.linalg.solve(
            K[:n_tr, :n_tr] + width * 1e-2 * np.eye(n_tr), y_train_tr)
        test_err = np.mean((pred_val - y_val)**2)

        return {'epoch': ep, 'test_err': test_err,
                'K': K, 'ev': ev, 'v': v, 'label_align': np.sum(v * ev)}

    # 初始
    m0 = measure(0)

    # 训练
    for ep in range(1, max(cps) + 1):
        opt.zero_grad()
        loss = nn.MSELoss()(model(x_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep in cps:
            m = measure(ep)
            results['epochs'].append(ep)
            results['test_err'].append(m['test_err'])
            results['cka_kk'].append(cka(m0['K'], m['K']))
            results['cka_kl'].append(cka(m['K'], np.outer(y, y)))
            diff_norm = np.linalg.norm(m['K'] - m0['K'], 'fro')
            results['frob_diff'].append(diff_norm / (np.linalg.norm(m0['K'], 'fro') + 1e-12))
            results['top_eig_ratio'].append(m['ev'][0] / (m0['ev'][0] + 1e-12))
            results['align_total'].append(m['label_align'])

    # 把 epoch 0 也加入
    results['epochs'] = [0] + results['epochs']
    results['test_err'] = [m0['test_err']] + results['test_err']
    results['cka_kk'] = [1.0] + results['cka_kk']
    results['cka_kl'] = [cka(m0['K'], np.outer(y, y))] + results['cka_kl']
    results['frob_diff'] = [0.0] + results['frob_diff']
    results['top_eig_ratio'] = [1.0] + results['top_eig_ratio']
    results['align_total'] = [m0['label_align']] + results['align_total']
    results['ev_init'] = m0['ev'][:10]
    results['ev_final'] = measure(max(cps))['ev'][:10]

    return results


# ============================================================
# 扫描
# ============================================================

WIDTHS = [32, 64, 128, 256, 512, 1024]
DATASETS = ['poly', 'highfreq', 'gmm']

all_data = {}
for dn in DATASETS:
    print(f"\n=== {dn} ===", flush=True)
    all_data[dn] = {}
    for w in WIDTHS:
        t0 = time.time()
        r = run(w, dn)
        t = time.time() - t0
        last_cka = r['cka_kk'][-1]
        last_frob = r['frob_diff'][-1] * 100
        print(f"  width={w:4d}  CKA={last_cka:.4f}  FrobΔ={last_frob:.1f}%  "
              f"test_err={r['test_err'][-1]:.4f}  {t:.0f}s", flush=True)
        all_data[dn][w] = r


# ============================================================
# 绘图
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
colors_w = {32: 'tab:blue', 64: 'tab:orange', 128: 'tab:green',
            256: 'tab:red', 512: 'tab:purple', 1024: 'tab:brown'}

W_LABEL = {32: 32, 64: 64, 128: 128, 256: 256, 512: 512, 1024: 1024}

for row, dn in enumerate(DATASETS):
    data = all_data[dn]

    # 列0: test error 动态
    ax = axes[row, 0]
    for w in WIDTHS:
        r = data[w]
        ax.plot(r['epochs'], r['test_err'], 'o-', color=colors_w[w],
                label=f'w={w}', linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Error')
    ax.set_title(f'{dn}: Test Error')
    ax.legend(fontsize=7, ncol=2)

    # 列1: CKA 动态
    ax = axes[row, 1]
    for w in WIDTHS:
        r = data[w]
        ax.plot(r['epochs'], r['cka_kk'], 'o-', color=colors_w[w],
                linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('CKA(K₀, Kₜ)')
    ax.set_title(f'{dn}: Kernel Stability (CKA)')
    ax.set_ylim(0.995, 1.001)

    # 列2: Frobenius 变化
    ax = axes[row, 2]
    for w in WIDTHS:
        r = data[w]
        ax.plot(r['epochs'], [x*100 for x in r['frob_diff']], 'o-',
                color=colors_w[w], linewidth=1.5, markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('‖ΔK‖/‖K‖ (%)')
    ax.set_title(f'{dn}: Frobenius Change')

    # 列3: width 扫描的最终 CKA
    ax = axes[row, 3]
    final_cka = [data[w]['cka_kk'][-1] for w in WIDTHS]
    final_frob = [data[w]['frob_diff'][-1] * 100 for w in WIDTHS]
    final_err = [data[w]['test_err'][0] / data[w]['test_err'][-1] for w in WIDTHS]  # error reduction ratio

    ax2 = ax.twinx()
    ax.plot(WIDTHS, final_cka, 'o-', color='C0', linewidth=2, label='CKA')
    ax2.plot(WIDTHS, final_frob, 's--', color='C1', linewidth=2, label='Frob%')
    ax.set_xlabel('Width')
    ax.set_ylabel('Final CKA', color='C0')
    ax2.set_ylabel('Final ‖ΔK‖/‖K‖ (%)', color='C1')
    ax.set_title(f'{dn}: Width → Stability')
    ax.set_xscale('log', base=2)
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])

plt.tight_layout()
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/width_sweep.png', dpi=150)
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/width_sweep.pdf')
print(f"\nFigure saved.", flush=True)

# ---- 打印汇总表 ----
print("\n\n" + "="*80)
print(f"{'Dataset':<12} {'Width':<8} {'CKA_final':<12} {'Frob%':<12} {'Err_init':<12} {'Err_final':<12} {'Err_ratio':<12}")
print("="*80)
for dn in DATASETS:
    for w in WIDTHS:
        r = all_data[dn][w]
        ei = r['test_err'][0]
        ef = r['test_err'][-1]
        print(f"{dn:<12} {w:<8} {r['cka_kk'][-1]:<12.6f} {r['frob_diff'][-1]*100:<12.2f} {ei:<12.4f} {ef:<12.4f} {ei/ef:<12.2f}")
    print("-"*80)
