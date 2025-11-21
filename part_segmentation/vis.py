import argparse
import os
import torch
import datetime
import time
import random
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
# import ipdb
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
Airplane	02691156
Bag	        02773838
Cap	        02954340
Car	        02958343
Chair	    03001627
Earphone	03261776
Guitar	    03467517
Knife	    03624134
Lamp	    03636649
Laptop	    03642806
Motorbike   03790512
Mug	        03797390
Pistol	    03948459
Rocket	    04099429
Skateboard  04225987
Table	    04379243'''

cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                 [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                 [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt_mamba3', help='model name')
    parser.add_argument('--batch_size', type=int, default=40, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='part_seg_visual', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--config', type=str, default='cfgs/config.yaml', help='config file')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default='output/part_seg/1014_1340/checkpoints/best_model.pth', help='ckpts')
    parser.add_argument('--root', type=str, default='../../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/',
                        help='data root')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%m%d-%H%M'))
    exp_dir = Path('../')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('output')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    shutil.copy(args.ckpts, str(exp_dir))
    shutil.copy(args.config, str(exp_dir))
    # shutil.copy('models/%s.py' % 'pt_mamba', str(exp_dir))


    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    # TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=10)
    # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    # shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    if args.config is not None:
        from utils.config import cfg_from_yaml_file
        from utils.logger import print_log
        if args.config[:13] == "segmentation/":
            args.config = args.config[13:]
        config = cfg_from_yaml_file(args.config)
        log_string(config)
        if hasattr(config, 'epoch'):
            args.epoch = config.epoch
        if hasattr(config, 'batch_size'):
            args.epoch = config.batch_size
        if hasattr(config, 'learning_rate'):
            args.learning_rate = config.learning_rate
        if hasattr(config, 'ckpt') and args.ckpts is None:
            args.ckpts = config.ckpts
        if hasattr(config, 'model'):
            MODEL = importlib.import_module(config.model) if hasattr(config, 'model') else importlib.import_module(
                args.model)
            classifier = MODEL.get_model(num_part,config).cuda()
        else:
            MODEL = importlib.import_module(args.model)
            classifier = MODEL.get_model(num_part).cuda()
    else:
        MODEL = importlib.import_module(args.model)
        shutil.copy('models/%s.py' % args.model, str(exp_dir))
        classifier = MODEL.get_model(num_part).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    classifier.load_state_dict(torch.load(args.ckpts)['model_state_dict'],strict=False)



    MODEL_MAE = importlib.import_module('pt_mamba3')
    # shutil.copy('models/%s.py' % 'pt_mamba', str(exp_dir))
    classifier_2 = MODEL_MAE.get_model(num_part, config).cuda()
    ckpt_path = args.ckpts
    classifier_2.load_state_dict(torch.load(ckpt_path)['model_state_dict'],strict=False)



    ## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        num_trainable_params = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
                num_trainable_params += param.numel()
            else:
                decay.append(param)
                num_trainable_params += param.numel()

        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        log_string('########################################################################')
        log_string('>> {:25s}\t{:.2f}\tM  {:.2f}\tK'.format(
            '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6), num_trainable_params / (1.0 * 10 ** 3)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
        log_string('>> {:25s}\t{:.2f}\t%'.format('# TuningRatio:', num_trainable_params / total_params * 100.))
        log_string('########################################################################')

        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  # t_mul=1,
                                  lr_min=1e-6,
                                  # decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    classifier.zero_grad()
    for epoch in range(0, 1):
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()
            classifier2 = classifier_2.eval()

            data_path = str(exp_dir) + '/vis/'
            data_path_gt = str(exp_dir) + '/vis/'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            if not os.path.exists(data_path_gt):
                os.makedirs(data_path_gt)
            selected_batch_id = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                                 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
            # selected_batch_id = np.linspace(100,2500,80).tolist()
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if batch_id in selected_batch_id:  ## randomly select some instance for visualization.
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    ### mamba prediction
                    seg_pred = classifier(points, to_categorical(label, num_classes))
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    ### masksurf prediction
                    seg_pred_masksurf = classifier2(points, to_categorical(label, num_classes))
                    cur_pred_val_masksurf = seg_pred_masksurf.cpu().data.numpy()
                    cur_pred_val_logits_masksurf = cur_pred_val_masksurf

                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()
                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        logits_masksurf = cur_pred_val_logits_masksurf[i, :, :]

                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                        label_in_cate = np.argmax(logits[:, seg_classes[cat]], 1) ## 0,1,2,3
                        label_in_cate_masksurf = np.argmax(logits_masksurf[:, seg_classes[cat]], 1)  ## 0,1,2,3
                        label2color = torch.from_numpy(cmap[label_in_cate])
                        label2color_masksurf = torch.from_numpy(cmap[label_in_cate_masksurf])
                        points = points.cpu()
                        point_color = torch.cat([points[0].transpose(0,1), label2color], dim=1)
                        point_color_masksurf = torch.cat([points[0].transpose(0, 1), label2color_masksurf], dim=1)

                        gt_label_in_cate = target - seg_classes[cat][0]
                        label2color_gt = torch.from_numpy(cmap[gt_label_in_cate])
                        point_color_gt = torch.cat([points[0].transpose(0, 1), label2color_gt[0]], dim=1)

                        fout = open(data_path + cat + str(batch_id) + 'mamba.obj', 'w')
                        fout_masksurf = open(data_path + cat + str(batch_id) + 'mae.obj', 'w')
                        fout_gt = open(data_path_gt + cat + str(batch_id) + 'gt.obj', 'w')
                        for i in range(point_color.size(0)):
                            fout.write('v %f %f %f %d %d %d\n' % (
                                point_color[i, 0], point_color[i, 1], point_color[i, 2], point_color[i, 3], point_color[i, 4],
                                point_color[i, 5]))
                            fout_masksurf.write('v %f %f %f %d %d %d\n' % (
                                point_color_masksurf[i, 0], point_color_masksurf[i, 1], point_color_masksurf[i, 2], point_color_masksurf[i, 3],
                                point_color_masksurf[i, 4],
                                point_color_masksurf[i, 5]))
                            fout_gt.write('v %f %f %f %d %d %d\n' % (
                                point_color_gt[i, 0], point_color_gt[i, 1], point_color_gt[i, 2], point_color_gt[i, 3], point_color_gt[i, 4],
                                point_color_gt[i, 5]))
                        fout.close()
                        fout_masksurf.close()
                        fout_gt.close()
                        # print((cur_pred_val == target).sum() / 2048)
                        # ipdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    main(args)