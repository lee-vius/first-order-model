from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


# This function runs in the main file for training process.
def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    # Refer to *.yaml, "train_params" section.
    # This including epoch nums, etc ...
    train_params = config['train_params']

    # Define the optimizers for three sub-networks
    # Refer to Adam() document for details
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        # Load in pretrained-models if set so
        # Models passed in are empty-initialized, which will be loaded in the following function
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    # TODO: not sure what's this, it seems to define schedulers contronlling training details
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        # Augment the dataset according to "num_reapeat"
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    # Load in data with form that network can determine
    # Refer to pytorch DataLoader for details
    # 这里dataloader是一个FramesDataset类，它是 Dataset 的一个子类，所以可以有如下操作
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    # Initialize two models for training
    # TODO: 阅读 generator 和 discrimator 的构造，key point detector 的部分应包含在 generator 当中
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    # TODO: 阅读 discriminator，需注意的是上述 Generator 中也有 discriminator 存在，高清两者区别
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    # Transfer model to gpu type
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                # 此处为前向传播，第一个返回值为loss，第二个为生成器的输出图片
                losses_generator, generated = generator_full(x)

                # TODO: 猜测此处计算的loss是针对一个视频的重建结果，这里取了平均值进行计算
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                # 此处分别使用不同部分的优化器进行 step 更新
                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                # 此处判断是否使用 GAN 的训练思想
                if train_params['loss_weights']['generator_gan'] != 0:
                    # 增加判别器的使用
                    optimizer_discriminator.zero_grad()
                    # 用判别器判定生成数据和源数据
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    # 更新判别器
                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                # 注意此处的 update 是 python 中字典自带的更新方式
                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            # 此处为一个 epoch 的工作完成
            # TODO: 这是之前不确定是什么的数据结构，推断是对训练的schedule器的更新
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
