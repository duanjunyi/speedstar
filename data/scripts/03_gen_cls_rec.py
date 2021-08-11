"""
split test annotations according to class name
and save at data/class_records/[classname].txt
each line: 'img_idx x1,y1,x2,y2,difficult'
"""
from collections import defaultdict
from pathlib import Path
import yaml
import xml.etree.ElementTree as ET

PROJ_DIR = Path(__file__).parent.parent.parent
DATA_YAML = PROJ_DIR.joinpath('data/dataset.yaml')    # 数据集配置文件
with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)
DATASET_PATH = PROJ_DIR.joinpath(data_cfg['path'])
ANNO_PATH = DATASET_PATH / 'annotations'

def parse_anno(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text.split('.')[0]),
            int(bbox.find("ymin").text.split('.')[0]),
            int(bbox.find("xmax").text.split('.')[0]),
            int(bbox.find("ymax").text.split('.')[0]),
        ]
        objects.append(obj_struct)
    return objects



if __name__ == "__main__":
    # 类记录路径
    cls_record_path = DATASET_PATH / "class_records"
    cls_record_path.mkdir(exist_ok=True)

    with (DATASET_PATH/'test.txt').open('r') as f:
        img_list = f.readlines()

    Cls_Record = defaultdict(list)
    for img in img_list:
        img_name = Path(img).stem
        objs = parse_anno(str(ANNO_PATH.joinpath(img_name+'.xml')))
        line = img_name + ' {:d},{:d},{:d},{:d},{:d}\n'
        for obj in objs:
            cls_name = obj["name"]
            difficult = obj["difficult"]
            bbox = obj["bbox"]
            Cls_Record[cls_name].append(line.format(*bbox, difficult))

    for cls_name, lines in Cls_Record.items():
        print("class: {:<20s}, total num: {:d}".format(cls_name, len(lines)))
        with cls_record_path.joinpath(cls_name+'.txt').open('w') as f:
            f.write(''.join(lines))





