# coding=utf-8
import os.path as osp


# 服务配置
WEIGHT = 'yolov5m6_datav3_1280.pt'                              # 使用的模型
MODEL_PATH = osp.abspath(osp.join(osp.dirname(__file__)))      # model文件夹路径
WEIGHT_PATH = osp.join(MODEL_PATH, 'weights', WEIGHT)
try:
    IMG_SIZE = int(WEIGHT.split('.')[0].split('_')[-1])       # 图片尺寸
except:
    IMG_SIZE = 640
print(f'Image size: {IMG_SIZE}')

# 本地测试配置
EVAL_WEIGHT_PATH = WEIGHT_PATH
EVAL_DATASET_PATH = 'D:\\Huawei\\ModelArts_Yolov4\\data\\datav3'   # 测试用的数据集: 必须按我放置的格式

# 推理配置
NMS_THRESH =0.1
CONF_THRESH = 0.3

Customer_DATA = {
    "NUM": 6,  # your dataset number
    "CLASSES": [
        "red_stop",
        "green_go",
        "yellow_back",
        "speed_limited",
        "speed_unlimited",
        "pedestrian_crossing",
        ],  # your dataset class
}
