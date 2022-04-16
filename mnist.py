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
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable
from models.simpleNet import MNIST
from grod import GrodOptimizer



def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")
    args = parser.parse_args()
    return args


def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)

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


def inference_mnist():
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        mnist_infer = MNIST("mnist")
        # load checkpoint
        model_dict, _ = fluid.dygraph.load_persistables("mnist_params")
        mnist_infer.load_dict(model_dict)
        print("checkpoint loaded")

        # start evaluate mode
        mnist_infer.eval()

        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/infer_3.png')

        results = mnist_infer(to_variable(tensor_img))
        lab = np.argsort(results.numpy())
        print("Inference result of image/infer_3.png is: %d" % lab[0][-1])
import math

IMAGENET1000 = 50000
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 32
def optimizer_setting(ori_dict):

    total_images = IMAGENET1000

    step = int(math.ceil(float(total_images) / batch_size))

    epochs = [30, 60, 90]
    bd = [step * e for e in epochs]

    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    # optimizer = fluid.optimizer.Momentum(
    #     learning_rate=fluid.layers.piecewise_decay(
    #         boundaries=bd, values=lr),
    #     # initial_point=ori_dict,
    #     momentum=momentum_rate,
    #     regularization=fluid.regularizer.L2Decay(l2_decay))
    optimizer = GrodOptimizer(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        initial_point=ori_dict,
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer

def train_mnist(args):
    epoch_num = args.epoch
    BATCH_SIZE = 64

    trainer_count = fluid.dygraph.parallel.Env().nranks
    # place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
    #     if args.use_data_parallel else fluid.CUDAPlace(0)
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

        # mnist_tgt = MNIST("mnist")
        # ori_dict, _ = fluid.dygraph.load_persistables("mnist/MINIST_0")
        ori_dict, _ = fluid.dygraph.load_persistables("mnist_params")
        # models, optimizers = fluid.dygraph.load_persistables("save_dir")

        # target_dict = quant_method.quantization(ori_dict2)
        # mnist_tgt.load_dict(ori_dict)


        mnist = MNIST("mnist")
        mnist.load_dict(ori_dict)
        # adam = AdamOptimizer(learning_rate=0.001)
        optimizer = optimizer_setting(ori_dict)
        if args.use_data_parallel:
            mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost, acc = mnist(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                if args.use_data_parallel:
                    avg_loss = mnist.scale_loss(avg_loss)
                    avg_loss.backward()
                    mnist.apply_collective_grads()
                else:
                    avg_loss.backward()

                # adam.minimize(avg_loss)
                optimizer.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))

            mnist.eval()
            test_cost, test_acc = test_mnist(test_reader, mnist, BATCH_SIZE)
            mnist.train()
            if args.ce:
                print("kpis\ttest_acc\t%s" % test_acc)
                print("kpis\ttest_cost\t%s" % test_cost)
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                epoch, test_cost, test_acc))

        # fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir", optimizer)
        # print("checkpoint saved")

        inference_mnist()


if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)
