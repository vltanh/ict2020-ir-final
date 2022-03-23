import torchvision
import torch
import torch.nn as nn
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from datasets import AIC2020Track2

import sys

weight_path = sys.argv[1]


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
    return accuracies["precision_at_1"], accuracies['mean_average_precision_at_r']


device = torch.device("cuda")

print('Loading data... ', end='', flush=True)
gallery_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_gallery_test.csv', False)
query_ds = AIC2020Track2(
    'data/AIC21_Track2_ReID/image_train', 'list/reid_query_test.csv', False)

gallery_dl = torch.utils.data.DataLoader(gallery_ds, batch_size=256)
query_dl = torch.utils.data.DataLoader(query_ds, batch_size=256)
print('Done!', flush=True)

print('Loading model... ', end='', flush=True)
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Identity()
model.load_state_dict(torch.load(weight_path))
model = model.to(device)
print('Done!', flush=True)

accuracy_calculator = AccuracyCalculator(
    include=('precision_at_1', 'mean_average_precision_at_r'))

print(f'Evaluating...', flush=True)
acc, map_r = test(gallery_ds, query_ds, model, accuracy_calculator)

print(f'{weight_path} | Accuracy: {acc} | MAP@R: {map_r}', flush=True)
