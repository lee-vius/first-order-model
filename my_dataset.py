import os
from numpy.lib.utils import source
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from frames_dataset import read_video

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob


class MyDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    This Dataset will include both source and target data.
    """
    # 适应stylizer训练的 data 构建器
    # 整体与 frames_dataset 类似
    def __init__(self, source_dir, target_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        
        self.source_videos = os.listdir(source_dir)
        self.target_videos = os.listdir(target_dir)

        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        # load in source videos
        if os.path.exists(os.path.join(source_dir, 'train')):
            assert os.path.exists(os.path.join(source_dir, 'test'))
            print("Use predefined train-test split for source videos.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(source_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(source_dir, 'train'))
            test_videos = os.listdir(os.path.join(source_dir, 'test'))
            self.source_dir = os.path.join(self.source_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.source_videos, random_state=random_seed, test_size=0.2)
        # store the split videos
        if is_train:
            self.source_videos = train_videos
        else:
            self.source_videos = test_videos

        # load in target videos
        if os.path.exists(os.path.join(target_dir, 'train')):
            assert os.path.exists(os.path.join(target_dir, 'test'))
            print("Use predefined train-test split for target videos.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(target_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(target_dir, 'train'))
            test_videos = os.listdir(os.path.join(target_dir, 'test'))
            self.target_dir = os.path.join(self.target_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split")
            train_videos, test_videos = train_test_split(self.target_dir, random_state=random_seed, test_size=0.2)
        # store the split videos
        if is_train:
            self.target_videos = train_videos
        else:
            self.target_videos = test_videos

        self.is_train = is_train
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.target_videos)

    def __getitem__(self, idx):
        # will return index of source videos
        # and randomly return index of a target videos
        if self.is_train and self.id_sampling:
            name_source = self.source_videos[idx]
            path_source = np.random.choice(glob.glob(os.path.join(self.source_dir, name_source + '*.mp4')))
            name_target = self.target_videos[idx % len(self.target_videos)]
            path_target = np.random.choice(glob.glob(os.path.join(self.target_dir, name_target + '*.mp4')))
        else:
            name_source = self.source_videos[idx]
            path_source = os.path.join(self.source_dir, name_source)
            name_target = self.target_videos[idx % len(self.target_videos)]
            path_target = os.path.join(self.target_dir, name_target)

        video_src_name = os.path.basename(path_source)
        video_tar_name = os.path.basename(path_target)

        # handle source
        # 此情况是 path 是一个文件夹，里面装了每一帧的 png
        if self.is_train and os.path.isdir(path_source):
            frames = os.listdir(path_source)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            source_array = [img_as_float32(io.imread(os.path.join(path_source, frames[idx]))) for idx in frame_idx]
        else:
            # 读取视频
            source_array = read_video(path_source, frame_shape=self.frame_shape)
            num_frames = len(source_array)
            # 此处根据模式选项，打乱了视频顺序
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            source_array = source_array[frame_idx]
        # handle target
        # 此情况是 path 是一个文件夹，里面装了每一帧的 png
        if self.is_train and os.path.isdir(path_target):
            frames = os.listdir(path_target)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            target_array = [img_as_float32(io.imread(os.path.join(path_target, frames[idx]))) for idx in frame_idx]
        else:
            # 读取视频
            target_array = read_video(path_target, frame_shape=self.frame_shape)
            num_frames = len(target_array)
            # 此处根据模式选项，打乱了视频顺序
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            target_array = target_array[frame_idx]

        if self.transform is not None:
            source_array = self.transform(source_array)
            target_array = self.transform(target_array)

        out = {}
        # 构建输出字典
        if self.is_train:
            # 输出的时候只选取了前两帧作为源和驱动
            # 注：此处把 channel 作为第一个维度输出了
            s_source = np.array(source_array[0], dtype='float32')
            s_driving = np.array(source_array[1], dtype='float32')
            t_source = np.array(target_array[0], dtype='float32')
            t_driving = np.array(target_array[1], dtype='float32')

            out['driving'] = s_driving.transpose((2, 0, 1))
            out['source'] = s_source.transpose((2, 0, 1))
            out['t_driving'] = t_driving.transpose((2, 0, 1))
            out['t_source'] = t_source.transpose((2, 0, 1))
        else:
            video = np.array(source_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            video = np.array(target_array, dtype='float32')
            out['t_video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_src_name
        out['t_name'] = video_tar_name

        return out