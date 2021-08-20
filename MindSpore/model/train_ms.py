"""Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from mindspore.context import ParallelMode
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net
import val  # for end-of-epoch mAP
from models.yolov5s import Model
from train_utils.yolo_dataset import create_dataloader
from train_utils.general import labels_to_class_weights, check_img_size, one_cycle, colorstr, fitness
from train_utils.loss_ms import ComputeLoss

PROJ_DIR = Path(__file__).resolve()   # TODO 项目根目录，以下所有目录相对于此


def train(hyp,  opt):
    # opt
    epochs, batch_size, weights, data, noval, nosave= \
        opt.epochs, opt.batch_size, opt.weights, opt.data, opt.noval, opt.nosave
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device)  #change "GPU" when needed
    context.reset_auto_parallel_context()
    parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=1)

    # Directories
    save_dir = Path(PROJ_DIR / opt.save_dir)
    weight_dir = save_dir / 'weights'
    weight_dir.mkdir(parents=True, exist_ok=True)  # make dir
    last = weight_dir / 'last.pt'
    best = weight_dir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Image sizes
    gs = 32  # grid size (max stride)
    nl = 3  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_val = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # datasest config
    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
    train_path = data_dict['train']
    val_path = data_dict['val']

    # 训练集
    data_loader, dataset = create_dataloader(train_path, imgsz, batch_size, stride=gs, hyp=hyp, augment=True, cache=opt.cache_images,
                      rect=opt.rect, workers=opt.workers, image_weights=opt.image_weights, multi_process=opt.multi_process)

    nb = len(dataset) // batch_size # number of batches

    # 测试集 TODO
    # valloader = create_dataloader(val_path, imgsz_val, batch_size * 2, gs, single_cls,
    #                                   hyp=hyp, cache=opt.cache_images and not noval, rect=True, rank=-1,
    #                                   workers=workers,
    #                                   pad=0.5, prefix=colorstr('val: '))[0]

    # Model
    if weights.suffix=='.ckpt' and opt.resume:  # 如果给了预训练模型，并且resume，加载预训练模型
        # TODO 加载模型
        param_dict = load_checkpoint(str(weights))
        img_size = param_dict['module1_8.conv2d_0.weight'].shape[1]
        assert img_size==imgsz, 'image size of weights conflict!'
        model = Model(bs=batch_size, img_size=imgsz)  # batch size
        not_load_params = load_param_into_net(model, param_dict)
        model.set_train(True)
    else:   # 初始化模型
        model = Model(bs=batch_size, img_size=imgsz)  # create
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc) * nc  # attach class weights
    model.names = names

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 学习率
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    compute_loss = ComputeLoss(model)  # init loss class

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(dataloader)
        print(('\n' + '%10s' * 8) % ('Epoch', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # forward
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                f'{epoch}/{epochs - 1}', *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        final_epoch = epoch + 1 == epochs
        if not noval or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(data_dict,
                                        batch_size=batch_size * 2,
                                        imgsz=imgsz_val,
                                        model=ema.ema,
                                        single_cls=single_cls,
                                        dataloader=valloader,
                                        save_dir=save_dir,
                                        save_json=final_epoch,
                                        verbose=nc < 50 and final_epoch,
                                        plots=plots and final_epoch,
                                        compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),
                        'optimizer': optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    print(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # option file
    parser.add_argument('--option', type=str, default='', help='option.yaml path')
    # train config
    parser.add_argument('--device', type=str, default='CPU', help='CPU, GPU, ASCEND')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1024, 1024], help='[train, val] image sizes')
    parser.add_argument('--resume', default=False, help='resume most recent training')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    opt.multi_process = (opt.device == 'GPU')
    # read option file
    opt_file = PROJ_DIR / opt.option
    assert opt_file.exists(), "opt yaml does not exist"
    with opt_file.open('r') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace
    # check weights file
    opt.weights = PROJ_DIR / opt.weights
    assert opt.weights.exists(), 'ERROR: --resume weights checkpoint does not exist'
    print('Resuming training from %s' % str(opt.weights))
    # print
    print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    return opt


def main(opt):
    # Train
    train(opt.hyp, opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
