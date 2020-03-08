# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from config import cfg

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            
            
            # All to water
            water_id = 21
            sea_id = 26
            river_id = 60
            lake_id = 128
            
            pred[pred == sea_id] = water_id
            pred[pred == river_id] = water_id
            pred[pred == lake_id] = water_id
            
            # All to tree
            plant_id = 17
            grass_id = 9
            tree_id = 72
            fence_id = 32
            
            pred[pred == plant_id] = tree_id
            pred[pred == grass_id] = tree_id
            pred[pred == fence_id] = tree_id
            
            # All to building
            house_id = 25
            building_id = 1
            skyscraper_id = 48
            wall_id = 0
            hovel_id = 79
            tower_id = 84
            
            pred[pred == house_id] = building_id
            pred[pred == skyscraper_id] = building_id
            pred[pred == wall_id] = building_id
            pred[pred == hovel_id] = building_id
            pred[pred == tower_id] = building_id
            
            # All to road
            road_id = 6
            path_id = 52
            sidewalk_id = 11
            sand_id = 46
            hill_id = 68
            earth_id = 13
            field_id = 29
            land_id = 94
            grandstand_id = 51
            stage_id = 101
            dirttrack_id = 91
            sky_id = 2
            
            pred[pred == path_id] = road_id
            pred[pred == sidewalk_id] = road_id
            pred[pred == sand_id] = road_id
            pred[pred == hill_id] = road_id
            pred[pred == earth_id] = road_id
            pred[pred == field_id] = road_id
            pred[pred == land_id] = road_id
            pred[pred == grandstand_id] = road_id
            pred[pred == stage_id] = road_id
            pred[pred == dirttrack_id] = road_id
            
            # All to ceiling
            awning_id = 86
            ceiling_id = 5
            
            pred[pred == awning_id] = ceiling_id


            # all to mountain
            mountain_id = 16
            rock_id = 34
            
            pred[pred == rock_id] = mountain_id

            
            # all to stairs
            stairs_id = 53
            stairway_id = 59
            step_id = 121
            
            pred[pred == stairway_id] = stairs_id
            pred[pred == step_id] = stairs_id


            # all to chair
            chair_id = 19
            seat_id = 31
            armchair_id = 30
            sofa_id = 30
            bench_id = 69
            swivel_id = 76
            
            pred[pred == seat_id] = chair_id
            pred[pred == armchair_id] = chair_id
            pred[pred == sofa_id] = chair_id
            pred[pred == bench_id] = chair_id
            pred[pred == swivel_id] == chair_id

            # all to car
            car_id = 20
            bus_id = 80
            truck_id = 83
            van_id = 102
            
            pred[pred == bus_id] = car_id
            pred[pred == truck_id] = car_id
            pred[pred == van_id] = car_id
            

        # [1, 21, 2, 6, 5, 16, 53, 19, 20]
#         ids = [building_id, water_id, sky_id, road_id, ceiling_id, mountain_id, stairs_id, chair_id, car_id]
        ids = [sky_id, water_id]
        
       
        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )
        
#         for i, cid in enumerate(ids):
#             pred[pred == cid] = i
               
        np.save(batch_data['info'][:len(batch_data['info'])-4] + ".npy", pred)


        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image paths, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    print(args.imgs)
    if os.path.isdir(args.imgs):
        print("many")
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
        
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
