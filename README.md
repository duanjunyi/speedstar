# YOLOv5-Pytorch for ModelArts

本仓库是基于[ultralytics/yolov5](https://github.com/ultralytics/yolov5)实现的YOLOv5。你可以方便地在自己的数据集上训练模型，并将模型部署为ModelArts服务。

## 1. 环境配置
* Windows
* python 3.7
* torch>=1.7.0

运行如下命令安装所有依赖：
```bash
pip install -r requirements.txt
```

## 2. 准备工作
### 2.1 数据集

以华为云人工智能大赛数据集为例，首先下载[数据集](https://marketplace.huaweicloud.com/markets/aihub/usercenter/?activeTab=MyDownload&id=b3879ae9-dc2a-4b8c-bfaf-428ecf2f1f9e#dataset)到OBS中，然后使用obsutil将数据集下载至本地。或者可以从百度云下载[Speedstar数据集（验证码：yolo）](https://pan.baidu.com/s/1rEd5VljAcuTPE7E9QOGsjA)。
下载完成后，将数据集放置于`data/`文件夹下，并修改文件夹名称，至此，你的目录结构应该为：
```
yolov5
├───data
│   └───Your_dataset
│       ├───annotations
│       │   ├──1.xml
│       │   └──...
│       ├───images
│       │   ├──1.jpg
│       │   └──...
```

接着，进行如下步骤，完成数据集的配置：
* 更新`data/dataset.yaml`配置文件中的`"path"`为数据集相对于train.py的路径，更新 `"nc"`，`"names"` 为数据集中类别总数以及类别名称。你也可以修改`"split_percent"`来修改训练集和测试集的比例。

* 划分数据集为测试集和训练集：运行`data/scripts/01_gen_img_index_file.py`，之后你会得到按照`"split_percent"`划分的数据集和测试集列表文件`data/Your_dataset/train.txt(test.txt)`。

* 将 `VOC - *.xml` 格式标签转换为 `YOLO - *.txt` 格式 (img x y w h)：运行 `data/scripts/02_voc_to_yolo.py`，生成的YOLO格式标签文件保存在`data/Your_dataset/labels/`下。

* 为每个类别生成标签文件：运行 `data/scripts/03_gen_cls_rec.py`，生成的类别标签保存于 `your_dataset/class_records/` 中，便于计算APs时读取。


### 2.2 下载权重文件
下载最新的模型： https://github.com/ultralytics/yolov5/releases

或者运行如下命令下载模型
```
$ bash path/to/download_weights.sh
```
将权重放置于`train.py`同级目录下。

## 3. 训练
运行如下指令开始训练:
```
python -u train.py
```
查看可选参数：
```
python train.py -h
```
在训练过程中，模型的副本将会默认保存在`runs/train/exp/weights/`下，支持断点继续训练：

```
python -u train.py --weights  your_backup.pth  --resume
```

## 4. 模型评估

### 4.1 评估设置
训练完成后，将`best.pt`复制到`modelarts/model/weights/`文件夹下，并修改`yolov5_config.py`文件中的`"WEIGHT"`，`"IMG_SIZE"`和
`"EVAL_DATASET_PATH"`。

### 4.2 计算mAP
运行：
```
cd ./modelarts/model/
python evaluate.py
```
运行结束后会输出模型对于测试集的`mAP@0.5 / mAP@0.5:0.95`。

### 4.3 可视化预测效果

运行`predict.py`，可以一张一张查看预测效果。

![demo](https://gitee.com/junyi-duan/speedstar/raw/master/docs/predict.gif)


## 5. ModelArts部署
本地运行`customize_service.py`，如果可以正确输出`test.jpg`中的边框，说明配置正确。接着将`modelarts`文件夹整体上传至OBS中，并选择从该文件夹构建模型。


