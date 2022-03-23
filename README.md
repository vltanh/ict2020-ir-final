# **Experiments with Circle Loss on AIC 2021's Vehicle Retrieval Dataset**

## **Info**

|Name|Student ID|Mail|
|---|---|---|
|Vũ Lê Thế Anh *|20C13002|anh.vu2020@ict.jvn.edu.vn|
|Nguyễn Lê Hồng Hạnh|20C13005|hanh.nguyen2020@ict.jvn.edu.vn|
|Trần Ngọc Quốc|20C13009|quoc.tran2020@ict.jvn.edu.vn|

* corresponding

## **Usage**

### **Extract metadata**

The dataset provided metadata in the form of an XML file `train_label.xml` which can be hard to processed. We first convert this into a more accessible JSON file.

```
python extract_train.py
```

The result will be saved as `list/train_image_metadata.json`.

### **Split data**

Since we use the data above for training, evaluation, and testing, we split it into corresponding CSV files.

```
python split.py
```

The results are stored in the `list` folder as CSVs file of tuples of `(image_id, vehicle_id, cam_id)`:
- `reid_train.csv`: contains the training data 
- `reid_query_[val/test]`: contains the queries for evaluation
- `reid_gallery_[val/test]`: contains the gallery for evaluation

### **Train**

```
python train.py <method> <m> <gamma> <pos mining> <neg mining>
```
where:
- `method`: either `am`, `triplet`, or `circle`
- `m`: relaxation factor (for Circle loss) or margin (for AM-Softmax and Triplet)
- `gamma`: scale factor (only for Circle and AM-Softmax)
- `pos/neg mining`: method to mine triplets, either `hard`, `semihard` or `all` (cannot both be `semihard`)

Example:

```
python train.py circle 0.4 256 hard semihard
```

### **Test**

```
python test.py <path/to/weight>
```

Example:
```
python test.py runs/circle_0.4_256.0_hard_semihard_3698/best_map.pth
```