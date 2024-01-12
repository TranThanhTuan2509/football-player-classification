import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from dataset import FootballDataset, collate_fn
from models import CNN
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
import argparse
import shutil



def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    parser.add_argument("--log_path", "-p", type=str, default="football_tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="football_checkpoints")
    parser.add_argument("--checkpoint_model", "-m", type=str, default=None)
    parser.add_argument("--lr", "-l", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        ToTensor(),
        RandomAffine(degrees=(-5, 5), translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        Resize((224, 224)),
        ColorJitter(brightness=0.125, contrast=0.25, saturation=0.5, hue=0.05)
    ])
    train_dataset = FootballDataset(root="./football-dataset/football", train=True, transform=train_transform)
    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": True,
        "collate_fn": collate_fn
    }
    train_dataloader = DataLoader(dataset=train_dataset, **train_params)
    val_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    val_dataset = FootballDataset(root="./football-dataset/football", train=False, transform=val_transform)
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "drop_last": False,
        "collate_fn": collate_fn
    }
    val_dataloader = DataLoader(dataset=val_dataset, **val_params)
    model = CNN(num_classes=20).to(device)
    # summary(model, input_size=(1, 3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.checkpoint_model and os.path.isfile(args.checkpoint_model):
        checkpoint = torch.load(args.checkpoint_model)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["best_loss"]
    else:
        start_epoch = 0
        best_loss = 1000

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if os.path.isdir(args.checkpoint_path):
        shutil.rmtree(args.checkpoint_path)
    os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(start_epoch, args.epochs):
        # MODEL TRAINING
        model.train()
        progress_bar = tqdm(train_dataloader, colour="blue")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device).long()
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward pass + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, loss.item()))
            writer.add_scalar("Train/loss", loss.item(), iter + epoch * len(train_dataloader))

        # MODEL VALIDATION
        all_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="yellow")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device).long()
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())

            acc = accuracy_score(all_labels, all_predictions)
            loss = sum(all_losses) / len(all_losses)
            print("Epoch {}. Validation loss: {}. Validation accuracy: {}".format(epoch + 1, loss, acc))
            writer.add_scalar("Valid/loss", loss, epoch)
            writer.add_scalar("Valid/acc", acc, epoch)

        # save model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if loss < best_loss:
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
            best_loss = loss


if __name__ == '__main__':
    args = get_args()
    train(args)
