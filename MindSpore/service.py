import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent

import cv2
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, context, Tensor, dtype, ops
from service_utils.tools import Resize,  batch_nms
from service_utils.general import colorprint
from service_utils.visualize import visualize_boxes

import yolov5_config as cfg
from models.yolov5s_datav3_1024_2 import Model


class Yolov5Service():

    def __init__(self):
        super(Yolov5Service, self).__init__()
        # 加载模型
        self.yolo = Model(bs=1)  # batch size 默认为 1
        param_dict = load_checkpoint(str(cfg.WEIGHT_PATH))
        not_load_params = load_param_into_net(self.yolo, param_dict)

        self.test_size = cfg.IMG_SIZE
        self.org_shape = (0, 0)
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        self.Classes = np.array(cfg.Customer_DATA["CLASSES"])
        self.resize = Resize((self.test_size, self.test_size), correct_box=False)

    def _preprocess(self, img):
        self.org_shape = img.shape[:2]
        img = self.resize(img).transpose(2, 0, 1) # c,w,h
        return Tensor(img[np.newaxis, ...], dtype=dtype.float32) # bs,c,w,h

    def _inference(self, data):
        # infer
        preds = self.yolo(data)  # list of Tensor
        pred=preds[3][0:1,...].asnumpy()  # Tensor [1, N, 11(xywh)]
        # nms
        pred = batch_nms(pred, self.conf_thresh, self.nms_thresh)  # numpy [1, n, 6(xyxy)]
        pred = pred[0]
        # resize to original size
        bbox = self.resize_pb( pred, self.test_size, self.org_shape)
        return bbox

    def run(self, data):
        return self._inference(self._preprocess(data))

    def resize_pb(self, pred_bbox, test_input_size, org_img_shape):
        """
        input: pred_bbox - [n, x+y+x+y+conf+cls(6)]
        Resize to origin size
        output: pred_bbox [n1, x+y+x+y+conf+cls (6)]
        """
        pred_coor = pred_bbox[:, :4] #  xyxy
        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0*test_input_size/org_w, 1.0*test_input_size/org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)Crop off the portion of the predicted Bbox that is beyond the original image
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )

        return np.concatenate([pred_coor, pred_bbox[:, 4:]], axis=-1)


    def __imread(self, img_byteio):
        """
        read image from byteio
        """
        return cv2.imdecode(
            np.frombuffer(img_byteio.getbuffer(), np.uint8),
            cv2.IMREAD_COLOR
        )

    def visualize(self, img, pred, img_name=None, save=False, score_thresh=0, save_path=None):
        if len(pred) > 0:
            bboxes  = pred[:, :4]
            cls_ids = pred[:, 5].round().astype(np.int32)
            scores  = pred[:, 4]
            visualize_boxes(
                image=img, boxes=bboxes, labels=cls_ids, probs=scores,
                class_labels=self.Classes, min_score_thresh=score_thresh)
        if save:
            cv2.imwrite(save_path, img)
        return img



if __name__ == "__main__":
    # image list
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    IMG_PATH = Path(cfg.EVAL_DATASET_PATH).joinpath('JPEGImages')
    img_list = np.array([x.name for x in IMG_PATH.iterdir()])
    np.random.shuffle(img_list)
    img_list[0] = '836.jpg'
    # predict
    predictor = Yolov5Service()

    cv2.namedWindow('Predict', flags=cv2.WINDOW_NORMAL)
    for img_name in img_list:
        # read image
        img = cv2.imread(str(IMG_PATH/img_name))
        print('Show: ' + str(IMG_PATH/img_name))
        print('Continue? ([y]/n)? ')
        pred = predictor.run(img)
        print(pred)
        img = predictor.visualize(img, pred, score_thresh=0.)
        cv2.imshow('Predict', img)
        c = cv2.waitKey()
        if c in [ord('n'), ord('N')]:
            exit()

