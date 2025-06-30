
---

### 2. `diffusion_model.py`
```python
import torch, torch.nn as nn

class MLPScoreNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
    def forward(self, x, t):
        t_emb = t.float().unsqueeze(-1) / 1000
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

def get_beta_schedule(T):
    return torch.linspace(1e-4, 0.02, T)

def q_sample(x0, t, betas, noise):
    sqrt_alpha_cum = torch.sqrt(torch.cumprod(1 - betas, dim=0))
    return sqrt_alpha_cum[t].unsqueeze(-1) * x0 + torch.sqrt(1 - sqrt_alpha_cum[t]**2).unsqueeze(-1) * noise
