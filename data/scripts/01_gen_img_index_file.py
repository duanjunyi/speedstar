"""
Generate train.txt and test.txt in DATASET_PATH
"""
import numpy as np
from pathlib import Path
PROJ_DIR = Path(__file__).parent.parent.parent
DATA_YAML = PROJ_DIR.joinpath('data/dataset.yaml')    # 数据集配置文件
with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)
DATASET_PATH = PROJ_DIR.joinpath(data_cfg['path'])
from tqdm import tqdm
import yaml


if __name__ == "__main__":
    Img_path     = DATASET_PATH / 'images'
    train_index_file = DATASET_PATH / 'train.txt'
    test_index_file =  DATASET_PATH / 'test.txt'
    # 获取所有图片路径
    img_list = [str(img.resolve()) for img in Img_path.glob('*.jpg')]
    # shuffle and split
    index = np.random.permutation(len(img_list))
    split_index = int(len(index) * data_cfg['split_percent'])

    train_list  = np.array(img_list)[ index[:split_index] ]
    test_list   = np.array(img_list)[ index[split_index:] ]

    with open(train_index_file, "w") as f:
        for image_id in tqdm(train_list):
            f.write(image_id + "\n")

    with open(test_index_file, "w") as f:
        for image_id in tqdm(test_list):
            f.write(image_id + "\n")
