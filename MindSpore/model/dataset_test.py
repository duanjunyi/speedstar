# 测试dataset文件
from train_utils.yolo_dataset import create_dataloader
import sys
from pathlib import Path
import yaml
BASE_DIR = Path(__file__).resolve().parent
PROJ_DIR = BASE_DIR / '../../'
from train_utils.general import labels_to_class_weights
# config
data_path = PROJ_DIR / 'data/datav3/test.txt'
hpy_path = PROJ_DIR / 'data/hyps/hyp.speedstar.yaml'
with open(hpy_path) as f:
        hyp = yaml.safe_load(f)



data_loader, dataset = create_dataloader(path=data_path, imgsz=640, batch_size=3, stride=32, hyp=hyp, augment=True, cache=False, pad=0.0,
                    rect=False, workers=4, image_weights=False, prefix='', multi_process=False)
class_weights = labels_to_class_weights(dataset.labels, 6) * 6
print(f"class_weights: {class_weights}")
# da = dataset[0]
# print(da)
for i, data in enumerate(data_loader):
    images = data["labels"]
    print(data.keys())
    print(data["labels"].shape)
    print(type(data["labels"]))
    print(type(data["labels"][0]))
    print('***********\n', data["images"].shape)
    print(type(data["images"]))
    print(type(data["images"][0]))
    print(data["labels"])
    exit()