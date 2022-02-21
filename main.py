import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, miners, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from dataloaders import BalancedBatchDataLoader

from datasets import AIC2020Track2
import sys
import datetime

from utils import set_deterministic, set_seed

SEED = 3698
set_deterministic()

loss = sys.argv[1]
margin = float(sys.argv[2])
gamma = float(sys.argv[3])
pos_level = sys.argv[4]
neg_level = sys.argv[5]

date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

outdir = f'runs/{loss}_{margin}_{gamma}_{pos_level}_{neg_level}_{date}'
writer = SummaryWriter(outdir)


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training Loss', loss.detach().cpu(),
                          epoch * len(train_loader) + batch_idx)


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print(
        "Test set accuracy (Precision@1) = {}".format(
            accuracies["precision_at_1"])
    )
    print(
        "Test set MAP = {}".format(accuracies["mean_average_precision_at_r"])
    )
    return accuracies["precision_at_1"], accuracies['mean_average_precision_at_r']


device = torch.device("cuda")

print('Loading data... ', end='', flush=True)
set_seed(SEED)
train_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_train.csv', True)
train_dl = BalancedBatchDataLoader(train_ds, 128, 4)

set_seed(SEED)
gallery_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_gallery_val.csv', False)
gallery_dl = torch.utils.data.DataLoader(gallery_ds, batch_size=256)

set_seed(SEED)
query_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_query_val.csv', False)
query_dl = torch.utils.data.DataLoader(query_ds, batch_size=256)
print('Done!', flush=True)

print('Loading model... ', end='', flush=True)
set_seed(SEED)
model = torchvision.models.resnet18(pretrained=True)
model.emb_dim = model.fc.in_features
model.fc = nn.Identity()
model = model.to(device)
print('Done!', flush=True)

print('Loading optimizer... ', end='', flush=True)
set_seed(SEED)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
print('Done!', flush=True)

num_epochs = 25

set_seed(SEED)
if loss == 'triplet':
    loss_func = losses.TripletMarginLoss(margin=margin)
elif loss == 'circle':
    loss_func = losses.CircleLoss(m=margin, gamma=gamma)
elif loss == 'am':
    loss_func = losses.CosFaceLoss(
        num_classes=train_ds.nclasses, embedding_size=model.emb_dim, margin=margin, scale=gamma)

set_seed(SEED)
mining_func = miners.BatchEasyHardMiner(
    pos_strategy=pos_level,
    neg_strategy=neg_level,
)

set_seed(SEED)
accuracy_calculator = AccuracyCalculator(
    include=('precision_at_1', 'mean_average_precision_at_r'))
### pytorch-metric-learning stuff ###

best_acc = 0.0
best_map = 0.0
for epoch in range(num_epochs):
    print(f'Epoch {epoch}', flush=True)

    print(f'Training...', flush=True)
    set_seed(SEED)
    train(model, loss_func, mining_func, device,
          train_dl, optimizer, epoch, writer)

    print(f'Evaluating...', flush=True)
    set_seed(SEED)
    acc, map_r = test(gallery_ds, query_ds, model, accuracy_calculator)

    writer.add_scalar('Validation Accuracy', acc, epoch)
    writer.add_scalar('Validation MAP', map_r, epoch)

    if acc > best_acc:
        torch.save(model.state_dict(), outdir + f'/best_acc.pth')
        best_acc = acc
    if map_r > best_map:
        torch.save(model.state_dict(), outdir + f'/best_map.pth')
        best_map = map_r

    print('===================', flush=True)
print('Best accuracy:', best_acc, flush=True)
print('Best MAP:', best_map, flush=True)
