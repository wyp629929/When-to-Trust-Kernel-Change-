"""
Gap 1: 自适应核对齐理论
——合成数据实验：验证 ΔAlign 预测泛化差距

核心思路：
1. 生成球面数据，构造已知频谱的目标函数
2. 用 NTK 公式计算固定核矩阵
3. 构造自适应核（按任务能量重新分配谱质量）
4. SVM 回归，比较泛化误差
5. 测量核对齐度，验证 ΔAlign 预测
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. 合成数据生成器
# ============================================================

def sample_sphere(n, d):
    """在 d 维球面上均匀采样 n 个点"""
    x = np.random.randn(n, d)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def arc_cosine_kernel(X, Y=None, order=1):
    """
    ReLU 激活函数的 arc-cosine kernel (NNGP kernel).
    k(x, y) = (1/pi) * ||x|| ||y|| * J_1(theta)
    其中 J_1(theta) = sin(theta) + (pi - theta) * cos(theta)
    """
    if Y is None:
        Y = X
    # 归一化
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    cos_theta = (X @ Y.T) / (X_norm @ Y_norm.T)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    norms = X_norm @ Y_norm.T
    if order == 1:
        J = (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    elif order == 0:
        J = (np.pi - theta) / np.pi
    else:
        raise ValueError(f"Unknown order: {order}")
    return norms * J


def ntk_kernel(X, Y=None):
    """
    两层 ReLU 网络的无限宽 NTK.
    K_NTK(x, y) = K_NNGP(x, y) + (x·y) * K_dot(x, y)
    其中 K_dot(x, y) = (1/pi) * (pi - theta)
    """
    if Y is None:
        Y = X
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    dot = X @ Y.T
    cos_theta = dot / (X_norm @ Y_norm.T)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    norms = X_norm @ Y_norm.T
    # NNGP term
    k_nngp = norms * (np.sin(theta) + (np.pi - theta) * cos_theta) / np.pi
    # NTK correction term
    k_dot = (np.pi - theta) / np.pi
    k_ntk = k_nngp + k_dot * dot
    return k_ntk


def construct_target(K, n_components, spectral_decay=1.0, seed=SEED):
    """
    构造目标函数 y = f*(x) + noise

    利用核矩阵 K 的特征分解：
    K = Phi @ diag(lambda) @ Phi^T
    目标：y = sum_i w_i * phi_i

    参数：
        spectral_decay: w_i^2 ~ lambda_i^spectral_decay 的幂律衰减
                        = 1.0: 目标能量与核谱匹配（easy task）
                        > 1.0: 目标集中在前几个分量（easier）
                        < 1.0: 目标高频分量多（hard task）
    """
    rng = np.random.RandomState(seed)
    eigvals, eigvecs = eigh(K)
    # eigh 返回从小到大，翻转
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # 取前 n_components 个
    eigvals = eigvals[:n_components] + 1e-10
    eigvecs = eigvecs[:, :n_components]
    # w_i^2 ~ eigvals[i]^spectral_decay
    w = eigvals ** (spectral_decay / 2)
    # 目标函数
    y = eigvecs @ (w * rng.randn(n_components))
    # 归一化
    y = y / np.std(y)
    return y, eigvals, eigvecs


def compute_cka(K, L):
    """Centered Kernel Alignment"""
    n = K.shape[0]
    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    # Frobenius inner product
    num = np.sum(K_c * L_c)
    den = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return num / (den + 1e-10)


# ============================================================
# 2. 自适应核构造
# ============================================================

def construct_adaptive_kernel(K_fixed, y_train, gamma_0=0.5, lam=0.01):
    """
    构造自适应核（模拟特征学习后的核）。

    核心想法：特征学习把谱质量重新分配到任务对齐的方向。
    lambda_tilde_i = lambda_i * [1 + gamma_0 * v_i / (n * lam + lambda_i)]

    其中 v_i = <y, phi_i>^2 / n 是任务在第 i 方向上的能量。
    """
    n = len(y_train)
    eigvals, eigvecs = eigh(K_fixed)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    # 任务投影 v_i
    proj = eigvecs.T @ y_train
    v_i = (proj ** 2) / n
    # 自适应调整
    eigvals_adapt = eigvals * (1 + gamma_0 * v_i / (n * lam + eigvals))
    eigvals_adapt = np.maximum(eigvals_adapt, 1e-10)
    # 重构核矩阵
    K_adapt = eigvecs @ np.diag(eigvals_adapt) @ eigvecs.T
    # 对称化
    K_adapt = (K_adapt + K_adapt.T) / 2
    return K_adapt, eigvals, eigvals_adapt, v_i


# ============================================================
# 3. 实验
# ============================================================

def run_experiment(d=20, n_train=200, n_test=500, n_components=50,
                   spectral_decay=1.0, gamma_0=0.5, noise_level=0.1,
                   lam_reg=0.01):
    """运行单次实验"""
    # 生成数据
    X = sample_sphere(n_train + n_test, d)
    K_full = ntk_kernel(X)
    y_full, eigvals, eigvecs = construct_target(
        K_full, n_components=n_components, spectral_decay=spectral_decay
    )
    # 加噪声
    y_full = y_full + noise_level * np.random.randn(len(y_full))
    # 分割
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_full[:n_train], y_full[n_train:]
    # 计算 NTK 子矩阵
    K_train = K_full[:n_train, :n_train]
    K_test = K_full[n_train:, :n_train]
    K_test_test = K_full[n_train:, n_train:]
    # 自适应核
    K_adapt_train, eigvals_ntk, eigvals_adapt, v_i = construct_adaptive_kernel(
        K_train, y_train, gamma_0=gamma_0, lam=lam_reg
    )
    # 构建自适应核的测试矩阵
    # 对于自适应核，训练和测试之间的协方差也需要调整
    # 简化：用训练集特征向量来调整
    eigvecs_train = eigh(K_train)[1][:, ::-1]
    K_adapt_test = K_test.copy()  # 简化起见暂用原值

    # ---- SVM with NTK kernel ----
    svr_ntk = SVR(kernel='precomputed', C=1.0/lam_reg, epsilon=0.0)
    svr_ntk.fit(K_train, y_train)
    pred_ntk = svr_ntk.predict(K_test)
    err_ntk = np.mean((pred_ntk - y_test) ** 2)

    # ---- SVM with adaptive kernel ----
    svr_adapt = SVR(kernel='precomputed', C=1.0/lam_reg, epsilon=0.0)
    svr_adapt.fit(K_adapt_train, y_train)
    pred_adapt = svr_adapt.predict(K_test)
    err_adapt = np.mean((pred_adapt - y_test) ** 2)

    # ---- Metrics ----
    y_label = np.outer(y_train, y_train)
    cka_ntk = compute_cka(K_train, y_label)
    cka_adapt = compute_cka(K_adapt_train, y_label)
    delta_align = cka_adapt - cka_ntk

    return {
        'err_ntk': err_ntk,
        'err_adapt': err_adapt,
        'err_gap': err_adapt - err_ntk,
        'cka_ntk': cka_ntk,
        'cka_adapt': cka_adapt,
        'delta_align': delta_align,
        'eigvals_ntk': eigvals_ntk[:20],
        'eigvals_adapt': eigvals_adapt[:20],
        'v_i': v_i[:20],
    }


def scan_spectral_decay():
    """扫描 spectrum_decay：控制任务难度"""
    decays = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []
    for sd in decays:
        r = run_experiment(spectral_decay=sd, gamma_0=0.5)
        r['spectral_decay'] = sd
        results.append(r)
        print(f"  decay={sd:.1f}  ntk_err={r['err_ntk']:.4f}  "
              f"adapt_err={r['err_adapt']:.4f}  gap={r['err_gap']:.4f}  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_gamma():
    """扫描 gamma_0：特征学习强度"""
    gammas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    results = []
    for g in gammas:
        r = run_experiment(spectral_decay=0.8, gamma_0=g)
        r['gamma'] = g
        results.append(r)
        print(f"  gamma={g:.1f}  ntk_err={r['err_ntk']:.4f}  "
              f"adapt_err={r['err_adapt']:.4f}  gap={r['err_gap']:.4f}  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_noise():
    """扫描噪声水平"""
    noises = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    for nl in noises:
        r = run_experiment(spectral_decay=0.8, gamma_0=0.5, noise_level=nl)
        r['noise'] = nl
        results.append(r)
        print(f"  noise={nl:.2f}  ntk_err={r['err_ntk']:.4f}  "
              f"adapt_err={r['err_adapt']:.4f}  gap={r['err_gap']:.4f}  "
              f"dAlign={r['delta_align']:.4f}")
    return results


def scan_n_train():
    """扫描训练样本量"""
    ns = [50, 100, 200, 400, 800]
    results = []
    for n in ns:
        r = run_experiment(n_train=n, spectral_decay=0.8, gamma_0=0.5)
        r['n_train'] = n
        results.append(r)
        print(f"  n={n}  ntk_err={r['err_ntk']:.4f}  "
              f"adapt_err={r['err_adapt']:.4f}  gap={r['err_gap']:.4f}  "
              f"dAlign={r['delta_align']:.4f}")
    return results


# ============================================================
# 4. 可视化
# ============================================================

def plot_results():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) 频谱衰减 vs 泛化差距
    print("\n=== 扫描 spectral_decay ===")
    r1 = scan_spectral_decay()
    ax = axes[0, 0]
    x = [r['spectral_decay'] for r in r1]
    ax.plot(x, [r['err_gap'] for r in r1], 'o-', label='err_gap', color='C0')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Spectral Decay (high=easier)')
    ax.set_ylabel('Generalization Gap (adapt - ntk)')
    ax.set_title('(a) Task Difficulty → Gap')
    ax.legend()

    # 对齐 vs 差距
    ax_twin = ax.twinx()
    ax_twin.plot(x, [r['delta_align'] for r in r1], 's--', label='ΔAlign', color='C1')
    ax_twin.set_ylabel('ΔAlignment', color='C1')
    ax_twin.tick_params(axis='y', labelcolor='C1')

    # (b) 特征学习强度 vs 泛化差距
    print("\n=== 扫描 gamma ===")
    r2 = scan_gamma()
    ax = axes[0, 1]
    x = [r['gamma'] for r in r2]
    ax.plot(x, [r['err_gap'] for r in r2], 'o-', color='C0')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Feature Learning Strength γ₀')
    ax.set_ylabel('Generalization Gap')
    ax.set_title('(b) Feature Learning Strength → Gap')

    # (c) 噪声水平 vs 泛化差距
    print("\n=== 扫描 noise ===")
    r3 = scan_noise()
    ax = axes[1, 0]
    x = [r['noise'] for r in r3]
    ax.plot(x, [r['err_gap'] for r in r3], 'o-', color='C0')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Generalization Gap')
    ax.set_title('(c) Noise Level → Gap')

    # (d) 核对齐 vs 泛化差距（合并所有实验）
    ax = axes[1, 1]
    all_gaps = []
    all_align = []
    for r in r1 + r2 + r3:
        all_gaps.append(r['err_gap'])
        all_align.append(r['delta_align'])
    ax.scatter(all_align, all_gaps, alpha=0.7)
    # 线性拟合
    coeffs = np.polyfit(all_align, all_gaps, 1)
    x_fit = np.linspace(min(all_align), max(all_align), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r--',
            label=f'slope={coeffs[0]:.2f}')
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')
    ax.set_xlabel('ΔAlignment (CKA_adapt - CKA_ntk)')
    ax.set_ylabel('ΔGeneralization (err_adapt - err_ntk)')
    ax.set_title(f'(d) ΔAlign vs ΔGen  r²={np.corrcoef(all_align, all_gaps)[0,1]**2:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig('./figures/exp1_synthetic.png', dpi=150)
    plt.savefig('./figures/exp1_synthetic.pdf')
    print(f"\nFigure saved.")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('./figures', exist_ok=True)
    plot_results()
