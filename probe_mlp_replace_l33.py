"""THE TEST: 134K MLP replaces cooperative zone. L34-L35 proofread.
Teacher-forced: capture h_L18 from baseline, predict h_L33 direction,
inject into model, let L34-L35 run, compare output tokens.
"""
import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings; warnings.filterwarnings('ignore')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', dtype=torch.float16, device_map='cuda', trust_remote_code=True)
model.eval()

SYS = {'en': 'You are a careful mathematical reasoner. Think step by step.',
       'zh': '\u4f60\u662f\u4e00\u4e2a\u4e25\u8c28\u7684\u6570\u5b66\u63a8\u7406\u8005\u3002'}
problems_en = [
    "Solve: 3x + 7 = 22", "Calculate: 347 + 658", "Hypotenuse legs 5 and 12",
    "GCD of 84 and 120", "Choose 3 from 7", "23 times 17", "1000 - 387",
    "Area circle radius 7", "Volume cube side 6", "Perimeter rectangle 15x8",
    "Sum primes under 20", "2^10 mod 7", "5 factorial", "8!/(5!*3!)",
    "All roses red. Red flower. Must it be rose?",
    "Every frumble transparent. Transparent creature. Must be frumble?",
    "If all cats mammals and some mammals swim, can cats swim?",
    "Solve: x^2 - 5x + 6 = 0",
]
problems_zh = ["\u89e3: 3x+7=22","\u8ba1\u7b97: 347+658","\u659c\u8fb9 5\u548c12","84\u548c120\u6700\u5927\u516c\u7ea6\u6570",
    "\u4ece7\u9009\u62e93","23\u4e5817","1000-387","\u534a\u5f847\u5706\u9762\u79ef","\u8fb96\u7acb\u65b9\u4f53\u4f53\u79ef",
    "15x8\u77e9\u5f62\u5468\u957f","20\u4ee5\u5185\u8d28\u6570\u548c","2^10 mod 7","5!","8!/(5!*3!)",
    "\u73ab\u7470\u7ea2\u3002\u7ea2\u82b1\u3002\u5fc5\u987b\u73ab\u7470\uff1f","frumble\u900f\u660e\u3002\u900f\u660e\u751f\u7269\u3002\u5fc5\u987bfrumble\uff1f",
    "\u732b\u54fa\u4e73\u3002\u67d0\u4e9b\u54fa\u4e73\u6e38\u6cf3\u3002\u732b\u80fd\u6e38\uff1f","\u89e3: x^2-5x+6=0"]

class HCap:
    def __init__(self): self.out = None
    def __call__(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        self.out = h[0, -1].detach().float().cpu()

# Collect training data
cap18 = HCap(); cap33 = HCap()
hook18 = model.model.layers[18].register_forward_hook(cap18)
hook33 = model.model.layers[33].register_forward_hook(cap33)
X_tr, Y_tr = [], []
for lang, probs in [('en', problems_en), ('zh', problems_zh)]:
    for p in probs:
        msgs = [{'role':'system','content':SYS[lang]},{'role':'user','content':p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').input_ids.to('cuda')
        with torch.inference_mode(): model(ids)
        X_tr.append(cap18.out.clone()); Y_tr.append(cap33.out.clone())
hook18.remove(); hook33.remove()
X_tr = torch.stack(X_tr).float(); Y_tr = torch.stack(Y_tr).float()
mean_norm = Y_tr.norm(dim=1).mean().item()
print(f'train: {X_tr.shape}, mean_norm: {mean_norm:.1f}', flush=True)

# Train direction MLP
class DirMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, 2048)
        )
    def forward(self, x):
        return self.net(x)

dir_mlp = DirMLP().float()
opt = torch.optim.Adam(dir_mlp.parameters(), lr=1e-3)
Y_dirs = Y_tr / (Y_tr.norm(dim=1, keepdim=True) + 1e-6)
for epoch in range(2000):
    opt.zero_grad()
    p = dir_mlp(X_tr)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-6)
    loss = (1 - nn.functional.cosine_similarity(p, Y_dirs, dim=1)).mean()
    loss.backward(); opt.step()
    if epoch == 1000:
        for g in opt.param_groups: g['lr'] = 1e-4

with torch.no_grad():
    p = dir_mlp(X_tr); p = p/(p.norm(dim=1,keepdim=True)+1e-6)
    cos_train = nn.functional.cosine_similarity(p, Y_dirs, dim=1).mean()
print(f'direction MLP cos_train: {cos_train:.4f}', flush=True)

prompt = "Every frumble in a glasshouse is transparent. Every transparent creature can pass through walls. I found a creature in a glasshouse that can pass through walls. Must it be a frumble?\n"

# Pass 1: baseline + capture h_L18 per step
cap18b = HCap()
hook18b = model.model.layers[18].register_forward_hook(cap18b)
ids = tok(prompt, return_tensors='pt').input_ids.to('cuda')
baseline_tokens = []; h18_per_step = []
for step in range(80):
    with torch.inference_mode(): out = model(ids)
    h18_per_step.append(cap18b.out.clone())
    nid = out.logits[0, -1].argmax().item()
    baseline_tokens.append(tok.decode(nid))
    if nid == tok.eos_token_id: break
    ids = torch.cat([ids, torch.tensor([[nid]], device='cuda')], dim=1)
hook18b.remove()

print(f'\nBASELINE ({len(baseline_tokens)} tokens):')
print(''.join(baseline_tokens))

# Pass 2: teacher-forced with MLP replacement at L33
step_idx = [0]
def replace_hook(module, inp, output):
    h = output[0] if isinstance(output, tuple) else output
    idx = step_idx[0]
    if idx < len(h18_per_step):
        with torch.no_grad():
            h18 = h18_per_step[idx].unsqueeze(0).unsqueeze(0).float()
            pred = dir_mlp(h18)
            pred_dir = pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)
            pred_scaled = (pred_dir * mean_norm).half().to(h.device)
            h[0, -1:, :] = pred_scaled
    return (h,) + output[1:] if isinstance(output, tuple) else h

hook_rep = model.model.layers[33].register_forward_hook(replace_hook)
ids2 = tok(prompt, return_tensors='pt').input_ids.to('cuda')
mlp_tokens = []
for step in range(len(baseline_tokens)):
    step_idx[0] = step
    with torch.inference_mode(): out = model(ids2)
    nid = out.logits[0, -1].argmax().item()
    mlp_tokens.append(tok.decode(nid))
    # Teacher force: append the BASELINE token to keep sequences aligned
    bt_ids = tok.encode(baseline_tokens[step], add_special_tokens=False)
    if bt_ids:
        ids2 = torch.cat([ids2, torch.tensor([bt_ids], device='cuda')], dim=1)
    else:
        break
hook_rep.remove()

print(f'\nMLP REPLACEMENT ({sum(p.numel() for p in dir_mlp.parameters())} params, L34-L35 proofread):')
print(''.join(mlp_tokens))

n_match = sum(1 for a, b in zip(baseline_tokens, mlp_tokens) if a == b)
print(f'\n{n_match}/{min(len(baseline_tokens), len(mlp_tokens))} tokens match')

del model; torch.cuda.empty_cache()
