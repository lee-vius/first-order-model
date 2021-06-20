from posix import listdir
import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class MyLogger:
    # This class is specified especially for training of stylizer
    # The function including saving cpks, records losses, etc
    def __init__(self, log_dir, checkpoint_freq=5, log_file_name='log.txt'):

        self.loss_list = []
        self.loss_all = {}
        # store checkpoints
        self.cpk_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(self.cpk_dir):
            os.mkdir(self.cpk_dir)
        # store visualized motion fields
        self.visualizations_dir = os.path.join(log_dir, 'figs')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    # print out each loss score into log file
    def log_scores(self, loss_names):
        # get mean loss for batches per type of losses
        loss_mean = np.array(self.loss_list).mean(axis=0)
        print("epoch {}".format(self.epoch), file=self.log_file)
        for name, value in zip(loss_names, loss_mean):
            # store in loss dict for plotting
            if name not in self.loss_all:
                self.loss_all[name] = []
            loss_string = "%s - %.5f" % (name, value)
            self.loss_all[name].append(value)
            print(loss_string, file=self.log_file)
        # flush out the buffer into the file
        self.loss_list = []
        self.log_file.flush()

    # plot the motion fields based on inputs
    def plot_motions(self, shape, inp, out):
        # reshape the data for plot
        inp = inp.reshape(shape).detach().data.cpu().numpy()[0] # only take the first data
        out = out.reshape(shape).detach().data.cpu().numpy()[0] # only take the first data

        save_path = os.path.join(self.visualizations_dir, "motion_{}.png".format(self.epoch))
        shape = inp.shape # the shape should be (num_kp, 64, 64, 2)
        fig, axes = plt.subplots(4, shape[0], figsize=(4 * shape[0], 4 * 4))

        for kp, ax in enumerate(axes[0]):
            # plot the original motion field
            ax.set_xlim(0, shape[1])
            ax.set_ylim(0, shape[2])
            ax.set_title("kp{}".format(kp))
            for i in range(shape[1]):
                if i % 4 != 0:
                    continue
                for j in range(shape[2]):
                    if j % 4 != 0:
                        continue
                    ax.arrow(i, j, *inp[kp, i, j] * 2, color='b', linewidth=0.5, head_width=0.5, head_length=0.5)

        for kp, ax in enumerate(axes[1]):
            # plot the generated motion field
            ax.set_xlim(0, shape[1])
            ax.set_ylim(0, shape[2])
            ax.set_title("kp{}".format(kp))
            for i in range(shape[1]):
                if i % 4 != 0:
                    continue
                for j in range(shape[2]):
                    if j % 4 != 0:
                        continue
                    ax.arrow(i, j, *out[kp, i, j] * 2, color='r', linewidth=0.5, head_width=0.5, head_length=0.5)

        for kp, ax in enumerate(axes[2]):
            # plot both original and generated motion fields
            ax.set_xlim(0, shape[1])
            ax.set_ylim(0, shape[2])
            ax.set_title("kp{}".format(kp))
            for i in range(shape[1]):
                if i % 4 != 0:
                    continue
                for j in range(shape[2]):
                    if j % 4 != 0:
                        continue
                    ax.arrow(i, j, *inp[kp, i, j] * 2, color='b', linewidth=0.5, head_width=0.5, head_length=0.5)
                    ax.arrow(i, j, *out[kp, i, j] *2 , color='r', linewidth=0.5, head_width=0.5, head_length=0.5)

        for kp, ax in enumerate(axes[3]):
            # plot difference motion fields
            ax.set_xlim(0, shape[1])
            ax.set_ylim(0, shape[2])
            ax.set_title("kp{}".format(kp))
            for i in range(shape[1]):
                if i % 4 != 0:
                    continue
                for j in range(shape[2]):
                    if j % 4 != 0:
                        continue
                    ax.arrow(i, j, *(out[kp, i, j] - inp[kp, i, j]) * 2, color='g', linewidth=0.5, head_width=0.5, head_length=0.5)

        # save figs
        plt.savefig(save_path)
        plt.close()
        return

    def plot_scores(self):
        # plot loss history
        save_path = self.visualizations_dir
        for name, value in self.loss_all.items():
            x_axis = list(range(len(value)))
            plt.plot(x_axis, value)
            plt.legend(['{}-loss'.format(name)])
            plt.xlabel("epoch number")
            plt.ylabel("Loss " + name)
            plt.title("Loss for Stylizer Module Training")
            plt.savefig(os.path.join(save_path, 'Loss_{}.png'.format(name)))
            plt.close()

    # save the checkpoints
    def save_cpk(self, emergent=False):
        # record params of each model passed in
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        # record the epoch number
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(8)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, stylizer=None, styDiscrim=None,
                 optimizer_stylizer=None, optimizer_styDiscrim=None):
        checkpoint = torch.load(checkpoint_path)
        # load in networks
        if stylizer is not None:
            stylizer.load_state_dict(checkpoint['stylizer'])
        if styDiscrim is not None:
            styDiscrim.load_state_dict(checkpoint['styDiscrim'])
        
        # load in optimizers
        if optimizer_stylizer is not None:
            optimizer_stylizer.load_state_dict(checkpoint['optimizer_stylizer'])
        if optimizer_styDiscrim is not None:
            optimizer_styDiscrim.load_state_dict(checkpoint['optimizer_styDiscrim'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    # record losses for each batch
    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    # make logs for an epoch
    def log_epoch(self, epoch, models, shape=None, inp=None, out=None):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
            if shape != None:
                self.plot_motions(shape, inp, out)
        self.log_scores(self.names)
