# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.core.evaluation import get_palette,mean_dice,mean_iou
from mmcv.runner import get_dist_info, init_dist

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import cv2
import os

classes = ('background', 'pedestrian')
palette = [[0, 0, 0], [256, 256, 256]]

@DATASETS.register_module()
class TinyObjectSegmentationDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                    split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    #setup_multi_processes(cfg)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if 'checkpoint' in args and osp.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')


    #rank, world_size = get_dist_info()

    #print(rank, world_size)

    #os.environ['RANK'] = '1'
    #os.environ['LOCAL_RANK'] = '1'
    #os.environ['WORLD_SIZE'] = '1'
    #os.environ['MASTER_ADDR'] = "127.0.0.1"
    #os.environ['MASTER_PORT'] = '29500'
    #init_dist('pytorch', **cfg.dist_params)

    model = MMDataParallel(model, device_ids=[0])
    
    model.eval()

    image_count = 0
    total_iou = 0
    total_dice = 0


    gt_list = []
    seg_list = []

    # the first several iterations may be very slow so skip them
    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        filename = data['img_metas'][0].data[0][0]['filename'] #data['img_metas'][0]#.item() #'/mnt/disk2/tos_group/dataset3/Original/part5_21.jpg'        

        filename = filename.replace('jpg', 'png')
        filename = filename.replace('Original', 'Label')

        mask = cv2.imread(filename, 0)


        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = np.array(result).squeeze(0)

        iou_score = mean_iou(result ,mask,2,ignore_index=None)['IoU'][1]
        dice_score = mean_dice(result,mask,2,ignore_index=None)['Dice'][1]

        gt_list.append(mask)
        seg_list.append(result)

        total_dice += dice_score
        total_iou += iou_score
        image_count += 1

    print(total_iou/image_count)
    print(total_dice/image_count)

    iou_score = mean_iou(seg_list ,gt_list,2,ignore_index=None)['IoU'][1]
    dice_score = mean_dice(seg_list,gt_list,2,ignore_index=None)['Dice'][1]

    print(iou_score)
    print(dice_score)



if __name__ == '__main__':
    main()
