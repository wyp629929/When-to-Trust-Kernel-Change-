"""CNN CIFAR-10: save eigenvalue spectra for spectrum evolution plot.
Just 1 seed, W=16 and W=128, to show spectral reorganization.
"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import json, warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

EPOCHS = 100; LR = 0.01

class CNN(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.conv1 = nn.Conv2d(3, W, 5, bias=False)
        self.conv2 = nn.Conv2d(W, 2*W, 5, bias=False)
        self.fc = nn.Linear(2*W * 5 * 5, 1, bias=False)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x).flatten()
    def get_features(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
            return x.view(x.size(0), -1).cpu().numpy()

def run(W, seed):
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_set = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)
    def filter_binary(ds, c1=0, c2=1):
        targets = np.array(ds.targets)
        mask = (targets == c1) | (targets == c2)
        ds.data = ds.data[mask]; ds.targets = targets[mask].tolist()
        ds.targets = [1 if t == c2 else -1 for t in ds.targets]
        return ds
    train_set = filter_binary(train_set); test_set = filter_binary(test_set)
    rng = np.random.RandomState(42+seed)
    n_tr, n_te = 1000, 500
    idx_tr = rng.choice(len(train_set), n_tr, replace=False)
    idx_te = rng.choice(len(test_set), n_te, replace=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr = torch.stack([train_set[i][0] for i in idx_tr]).to(device)
    y_tr = torch.FloatTensor([train_set[i][1] for i in idx_tr]).to(device)
    y_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-8)

    m = CNN(W).to(device)
    opt = optim.SGD(m.parameters(), lr=LR, momentum=0.9)

    H0 = m.get_features(X_tr)
    K0 = H0 @ H0.T
    evals0, evecs0 = np.linalg.eigh(K0)
    evals0 = evals0[::-1]

    for ep in range(EPOCHS):
        opt.zero_grad()
        nn.MSELoss()(m(X_tr), y_tr).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    H1 = m.get_features(X_tr)
    K1 = H1 @ H1.T
    evalsT, evecsT = np.linalg.eigh(K1)
    evalsT = evalsT[::-1]

    return {'W': W, 'seed': seed,
            'evals0_init': evals0[:20].tolist(),
            'evalsT_init': evalsT[:20].tolist()}

if __name__ == '__main__':
    results = []
    for W in [16, 128]:
        r = run(W, 0)
        results.append(r)
        print(f"W={W}: top-5 init evals = {r['evals0_init'][:5]}")
        print(f"W={W}: top-5 final evals = {r['evalsT_init'][:5]}")
    out = os.path.expanduser('~/cnn_spectrum_results.json')
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {out}")
