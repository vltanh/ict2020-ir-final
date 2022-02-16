import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from datasets import AIC2020Track2
import os
import sys

loss = sys.argv[1]

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                ), flush=True
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
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
        "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"])
    )
    print(
        "Test set MAP = {}".format(accuracies["mean_average_precision"])
    )
    return accuracies["precision_at_1"], accuracies['mean_average_precision']


device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

batch_size = 256

print('Loading data... ', end='', flush=True)
train_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_train.csv', True)
gallery_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_gallery_val.csv', False)
query_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_query_val.csv', False)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
gallery_dl = torch.utils.data.DataLoader(gallery_ds, batch_size=256)
query_dl = torch.utils.data.DataLoader(query_ds, batch_size=256)
print('Done!', flush=True)

print('Loading model... ', end='', flush=True)
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Identity()
model = model.to(device)
print('Done!', flush=True)

print('Loading optimizer... ', end='', flush=True)
optimizer = optim.Adam(model.parameters(), lr=0.01)
print('Done!', flush=True)

num_epochs = 50

### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)

if loss == 'triplet':
    loss_func = losses.TripletMarginLoss(
        margin=0.2, distance=distance, reducer=reducer)
elif loss == 'circle':
    loss_func = losses.CircleLoss(m=0.2, distance=distance, reducer=reducer)

mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
accuracy_calculator = AccuracyCalculator(include=(), k=1)
### pytorch-metric-learning stuff ###

outdir = f'runs/{loss}_loss'
os.makedirs(outdir, exist_ok=True)

best_acc = 0.0
best_map = 0.0
for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}', flush=True)

    print(f'Training...', flush=True)
    train(model, loss_func, mining_func, device,
          train_dl, optimizer, epoch)

    print(f'Evaluating...', flush=True)
    acc, map = test(gallery_ds, query_ds, model, accuracy_calculator)
    if acc > best_acc:
        torch.save(model.state_dict(), outdir + '/best_acc.pth')
    if map > best_map:
        torch.save(model.state_dict(), outdir + '/best_map.pth')

    print('===================', flush=True)
