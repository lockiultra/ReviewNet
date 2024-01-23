from models import ReviewNet, ReviewsDataset

import sys

import numpy as np
import pandas as pd
import evaluate

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

def get_train_test_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    len_of_dataset = len(y)
    permutation = np.random.permutation(len_of_dataset)
    train_permutation = permutation[:int(len_of_dataset * 0.85)]
    test_permutation = permutation[int(len_of_dataset * 0.85):]
    return X.iloc[train_permutation], X.iloc[test_permutation], y.iloc[train_permutation], y.iloc[test_permutation]

def get_train_test_datasets(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[ReviewsDataset, ReviewsDataset]:
    train_dataset = ReviewsDataset(X_train, y_train)
    test_dataset = ReviewsDataset(X_test, y_test)
    return train_dataset, test_dataset

def get_metric_score(model: ReviewNet, metric: str, test_loader: DataLoader) -> float:
    metric_fn = evaluate.load(metric)
    scores = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.type(torch.float32)
            y_batch = y_batch.type(torch.int64)
            y_pred = model(X_batch)
            y_pred = torch.argmax(y_pred, dim=1)
            metric_score = metric_fn.compute(predictions=y_pred, references=y_batch, average='weighted')
            scores.append(metric_score[metric])
    return np.sum(scores) / len(scores)

def train(train_loader: DataLoader, test_loader: DataLoader, metric: str='f1', num_epochs: int=3) -> list:
    net = ReviewNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    f1_scores = []

    for epoch in range(num_epochs):
        print(f'\n========== epoch {epoch + 1} ==========\n')
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            net.train()
            optimizer.zero_grad()
            X_batch = X_batch.type(torch.float32)
            y_batch = y_batch.type(torch.int64)
            y_pred = net(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            print(f'Batch {batch_idx}/{len(train_loader)}, batch loss - {loss.data:.4f}')
            optimizer.step()

            if batch_idx % 1000 == 0:
                f1_scores.append(get_metric_score(net, metric, test_loader))

    print(f'\nModel was trained, loss - {loss.data:.4f}')
    net.metric_scores = f1_scores
    torch.save(net.state_dict(), './ReviewNet.model')
    return f1_scores

def main(path_to_data: str):
    data = pd.read_csv(path_to_data)
    X = data.drop(['rating'], sxis=1)
    y= data['rating']
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    train_dataset, test_dataset = get_train_test_datasets(X_train, X_test, y_train, y_test)
    train_loader = DataLoader(train_dataset, batch_size=)
    test_loader = DataLoader(test_dataset)
    train(train_loader, test_loader)

if __name__ == '__main__':
    path_to_data = sys.argv[1]
    main(path_to_data)