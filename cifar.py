# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import ast
import numpy as np
import math
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
from models.resnet import ResNet
from grod import GrodOptimizer


IMAGENET1000 = 50000
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 32

def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument("-e", "--epoch", default=70, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")
    args = parser.parse_args()
    return args




def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        batch_len = len(data)
        dy_x_data = np.array([x[0].reshape(3, 32, 32)
                              for x in data]).astype('float32')
        y_data = np.array(
            # [x[1] for x in data]).astype('int64').reshape(batch_size, 1)
        # y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_len, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean





def optimizer_setting():

    total_images = IMAGENET1000

    step = int(math.ceil(float(total_images) / batch_size))

    epochs = [30, 60, 90]
    bd = [step * e for e in epochs]

    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay))


    return optimizer

def optimizer_setting2(ori_dict):

    total_images = IMAGENET1000

    step = int(math.ceil(float(total_images) / batch_size))

    epochs = [30, 60, 90]
    bd = [step * e for e in epochs]

    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    optimizer = GrodOptimizer(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        initial_point=ori_dict,
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer

def train(args):
    epoch_num = args.epoch
    BATCH_SIZE = batch_size

    # place = fluid.CUDAPlace(0)
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        # net_tgt = ResNet("resnet")
        ori_dict, _ = fluid.dygraph.load_persistables("resnet_params")
        # models, optimizers = fluid.dygraph.load_persistables("save_dir")
        # net_tgt.load_dict(ori_dict)

        net = ResNet("resnet", class_dim=10)
        net.load_dict(ori_dict)
        optimizer = optimizer_setting2(ori_dict)
        # optimizer = optimizer_setting()
        if args.use_data_parallel:
            net = fluid.dygraph.parallel.DataParallel(net, strategy)
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
            batch_size=BATCH_SIZE)

        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        # # Reader for testing. A separated data set for testing.
        test_reader = paddle.batch(
            paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(3, 32, 32)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost, acc = net(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                if args.use_data_parallel:
                    avg_loss = net.scale_loss(avg_loss)
                    avg_loss.backward()
                    net.apply_collective_grads()
                else:
                    avg_loss.backward()

                optimizer.minimize(avg_loss)
                net.clear_gradients()
                if batch_id % 1 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))

            net.eval()
            test_cost, test_acc = test_mnist(test_reader, net, BATCH_SIZE)
            net.train()
            if args.ce:
                print("kpis\ttest_acc\t%s" % test_acc)
                print("kpis\ttest_cost\t%s" % test_cost)
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                epoch, test_cost, test_acc))

        fluid.dygraph.save_persistables(net.state_dict(), "grod_checkpoint", optimizer)
        print("checkpoint saved")


if __name__ == '__main__':
    args = parse_args()
    train(args)
