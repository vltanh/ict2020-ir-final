import json
import csv
import random

random.seed(3698)

dataset_json_path = 'list/train_image_metadata.json'
train_ratio = 0.6
val_ratio = 0.2
query_ratio = 0.5

# Stratified split
splits = {
    'train': dict(),
    'gallery_val': dict(),
    'query_val': dict(),
    'gallery_test': dict(),
    'query_test': dict(),
}

# Load JSON
reference = json.load(open(dataset_json_path))

# Split
vehs = list(reference.keys())
random.shuffle(vehs)

train_sz = max(1, int(len(vehs) * train_ratio))
val_sz = max(1, int(len(vehs) * val_ratio))

train_vehs = vehs[:train_sz]
val_vehs = vehs[train_sz:train_sz+val_sz]
test_vehs = vehs[train_sz+val_sz:]

for veh_id in train_vehs:
    splits['train'][veh_id] = reference[veh_id]

for veh_id in val_vehs:
    cams = list(reference[veh_id].keys())
    random.shuffle(cams)

    query_sz = int(len(cams) * query_ratio)
    query_cams = cams[:query_sz]
    gallery_cams = cams[query_sz:]

    splits['gallery_val'][veh_id] = dict()
    splits['query_val'][veh_id] = dict()

    for cam_id in gallery_cams:
        splits['gallery_val'][veh_id][cam_id] = reference[veh_id][cam_id]

    for cam_id in query_cams:
        splits['query_val'][veh_id][cam_id] = reference[veh_id][cam_id]

for veh_id in test_vehs:
    cams = list(reference[veh_id].keys())
    random.shuffle(cams)

    query_sz = int(len(cams) * query_ratio)
    query_cams = cams[:query_sz]
    gallery_cams = cams[query_sz:]

    splits['gallery_test'][veh_id] = dict()
    splits['query_test'][veh_id] = dict()

    for cam_id in gallery_cams:
        splits['gallery_test'][veh_id][cam_id] = reference[veh_id][cam_id]

    for cam_id in query_cams:
        splits['query_test'][veh_id][cam_id] = reference[veh_id][cam_id]

# Output split to CSV
for split in splits.keys():
    rows = [['image_name', 'camera_id', 'vehicle_id']]
    for veh_id, v in splits[split].items():
        for cam_id, x in v.items():
            rows.extend([[xx, cam_id, veh_id] for xx in x])
    with open(f"list/reid_{split}.csv", 'w') as f:
        csv.writer(f).writerows(rows)
