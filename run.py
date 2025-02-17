import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator # LCH: refer here for generator
from modules.discriminator import MultiScaleDiscriminator # LCH: refer here for discriminator
from modules.keypoint_detector import KPDetector # LCH: refer here for key point detector

import torch

from train import train # LCH: For training process, everything in this module
from reconstruction import reconstruction
from animate import animate

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    # LCH: Argument handler, 
    # The parameter in the command will be handled by this handler
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    # opt contains all parameters for this command
    opt = parser.parse_args()
    with open(opt.config) as f:
        # read in the config file
        config = yaml.load(f) # config file contains code directions, including training details

    # generate a log path (store running time details)
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    # Declare an image generator
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    # If GPU Available, adapt to it
    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    # Declare a discriminator
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    # Declare a key point detector
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    # Print network details if using --verbose flag
    if opt.verbose:
        print(kp_detector)

    # Read in dataset details, defined in *.yaml config file, "dataset_params" section
    # Refer to ./config/vox-256.yaml for details
    # 数据预处理在此步骤完成，并读取进 dataset 变量中
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    print("Dataset size: {}, repeat number: {}".format(len(dataset), config['train_params']['num_repeats']))

    # Create the logging direction
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Copy the config file (*.yaml) into the logging path
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        # Start training
        # Look into this part further
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
