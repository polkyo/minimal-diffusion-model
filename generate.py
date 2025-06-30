import torch, argparse
import matplotlib.pyplot as plt
from diffusion_model import MLPScoreNet, get_beta_schedule

def generate(args):
    model = MLPScoreNet(28*28).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()
    betas = get_beta_schedule(args.timesteps).to(args.device)
    x = torch.randn(args.n_samples, 28*28).to(args.device)
    with torch.no_grad():
        for t in reversed(range(args.timesteps)):
            t_batch = torch.full((args.n_samples,), t, device=args.device, dtype=torch.long)
            noise_pred = model(x, t_batch)
            alpha = 1 - betas[t]
            x = (x - noise_pred * (1 - alpha).sqrt()) / alpha.sqrt()
            if t > 0:
                x += torch.sqrt(betas[t]) * torch.randn_like(x)
    grid = x.view(-1, 1, 28, 28)
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(torch.cat([g for g in grid[:16]], dim=2).cpu().numpy(), cmap='gray')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    generate(args)
