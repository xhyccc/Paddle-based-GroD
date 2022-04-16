This is the official site for PaddlePaddle-based implementation of the paper "GrOD: Deep Learning with Gradients Orthogonal Decomposition for Knowledge Transfer, Distillation, and Adversarial Training" published at ACM TKDD 2022. Details of reference are as follows.

Haoyi Xiong, Ruosi Wan, Jian Zhao, Zeyu Chen, Xingjian Li, Zhanxing Zhu, and Jun Huan. GroD: Deep Learning with Gradients Orthogonal Decomposition for Knowledge Transfer, Distillation, and Adversarial Training. ACM Transactions on Knowledge Discovery from Data (TKDD-22), 2022, ACM.


Models：存放了两个基于paddle动态图的神经网络。
	simpleNet: paddle 官方教程提供的model，用来训练MNIST，模型简单，训练速度快，具体动态图可以移步到动态图教程
https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/dygraph/DyGraph.html
	ResNet：根据https://github.com/PaddlePaddle/models/tree/develop/dygraph/resnet提取出来的。
resnet_params: 在flower数据集上训练ResNet50后保存下来的checkpoint。
mnist_params: 在MNIST上训练simpleNet后保存下来的checkpoint。
grod.py: 继承paddle的optimizer，grod算法实现。
cifar.py: 基于grod算法训练CIFAR10
mnist.py: 基于grod算法训练mnist
