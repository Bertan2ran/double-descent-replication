import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from models.cnn import CNN5
from utils.data import get_dataloaders

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_err, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(dim=1)
            total_loss += loss.item() * len(y)
            total_err += (preds != y).sum().item()
            total_samples += len(y)
    return total_loss / total_samples, total_err / total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=24, help="Width parameter k")
    parser.add_argument("--samples", type=int, default=50000, help="Train samples n")
    parser.add_argument("--noise", type=float, default=0.20, help="Symmetric label noise rate")
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Executing run: width={args.width}, n={args.samples}, noise={args.noise} on {device}")

    train_loader, test_loader = get_dataloaders(
        n_samples=args.samples, noise_rate=args.noise, batch_size=args.batch_size, seed=args.seed
    )

    model = CNN5(k=args.width, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    history = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_err, total_samples = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_err += (out.argmax(dim=1) != y).sum().item()
            total_samples += len(y)

        train_loss /= total_samples
        train_err /= total_samples

        if epoch % 25 == 0 or epoch == args.epochs:
            test_loss, test_err = evaluate(model, test_loader, criterion, device)
            print(f"Epoch [{epoch:3d}/{args.epochs:3d}] | Train Loss: {train_loss:.4f} | "
                  f"Train Err: {train_err:.4f} | Test Err: {test_err:.4f}")
        else:
            test_loss, test_err = None, None


        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_err": train_err,
            "test_loss": test_loss,
            "test_err": test_err,
            "width": args.width,
            "samples": args.samples,
            "noise": args.noise
        })
    

    df = pd.DataFrame(history)
    df.to_csv(args.out, index=False)
    print(f"Logs successfully written to: {args.out}")

if __name__ == "__main__":
    main()
