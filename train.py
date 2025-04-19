#!/usr/bin/env python

import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms
from networks.letters import LettersNetwork
import os

def get_data_loaders(data_dir='./data/letters', batch_size=64):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
    ])
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train(model, train_loader, criterion, optimizer, device, epochs=100, print_every=100):
    model.train()
    model.to(device)
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            if steps % print_every == 0:
                print(f"Epoch: {e+1}/{epochs}... Loss: {running_loss/print_every:.4f}")
                running_loss = 0

def save_model(model, path='./models/letters.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders()
    model = LettersNetwork()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_loader, criterion, optimizer, device)
    save_model(model)
    print("Training complete. Model saved to ./models/letters.pth")

if __name__ == "__main__":
    main()
