import torch, argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion_model import MLPScoreNet, get_beta_schedule, q_sample
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(args):
    ds = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = MLPScoreNet(28*28).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    betas = get_beta_schedule(args.timesteps).to(args.device)

    for epoch in range(args.epochs):
        total = 0
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(args.device)
            t = torch.randint(0, args.timesteps, (x.size(0),), device=args.device)
            noise = torch.randn_like(x)
            xt = q_sample(x, t, betas, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1} – loss = {total / len(loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
