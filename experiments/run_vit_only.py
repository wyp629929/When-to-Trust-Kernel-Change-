"""Run just the ViT experiment once with smaller batch."""
import json, sys
sys.path.insert(0, '.')
import exp_pretrained_rdep
exp_pretrained_rdep.BATCH_SIZE = 64  # smaller batch for 224x224 ViT
from exp_pretrained_rdep import run
import torch
device = torch.device('cuda')
exp_pretrained_rdep.EPOCHS = 15
exp_pretrained_rdep.N_TEST = 500
result = run('vit_b_16', 1e-5, 0, device, False)
print("ViT result:", flush=True)
print(json.dumps(result, indent=2))
with open('key_results/vit_pretrained_result.json', 'w') as f:
    json.dump(result, f)
print('ViT done')
