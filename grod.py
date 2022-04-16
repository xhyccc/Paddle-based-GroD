from __future__ import print_function
import numpy as np
from paddle.fluid.optimizer import Optimizer
from collections import defaultdict

from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program

from paddle.fluid import framework
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_
from paddle.fluid.clip import append_gradient_clip_ops, error_clip_callback
from paddle.fluid.framework import program_guard
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import ops
from paddle.fluid.regularizer import append_regularization_ops
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager


class GrodOptimizer(Optimizer):
    """

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        momentum (float): momentum factor
        use_nesterov (bool): enables Nesterov momentum
        regularization: A Regularizer, such as
                        fluid.regularizer.L2DecayRegularizer.
        name: A optional name prefix.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                moment_optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
                moment_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 use_nesterov=False,
                 regularization=None,
                 name=None,
                 initial_point=None):
        assert learning_rate is not None
        assert momentum is not None
        super(GrodOptimizer, self).__init__(
            learning_rate=learning_rate,
            regularization=regularization,
            name=name,)
        self.initial_point=initial_point
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)
        self.start_flag = True
        self.epoch = 1

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Velocity": velocity_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "VelocityOut": velocity_acc
            },
            attrs={"mu": self._momentum,
                   "use_nesterov": self._use_nesterov},
            stop_gradient=True)

        return momentum_op

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 grad_clip=None):
        """
        Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `backward()` and
        `apply_gradients()` into one.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.
            grad_clip (GradClipBase|None) : Gradient clip strategy

        Returns:
            tuple: (optimize_ops, params_grads) which are, list of operators appended;
            and list of (param, grad) Variables pair for optimization.
        """
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        if grad_clip is not None and framework.in_dygraph_mode():
            # TODO(hongyu): FIX later, this is only for dygraph, should be work for static mode
            params_grads = grad_clip(params_grads)

        import paddle.fluid as fluid
        for param, grad in params_grads:
            # print(param.name)
            if 'FC' not in param.name:
                grad_value = grad._ivar.value()
                grad_tensor = grad_value.get_tensor()
                grad_np = np.array(grad_tensor)

                param_value = param._ivar.value()
                param_tensor = param_value.get_tensor()
                param_np = np.array(param_tensor)
                # self.initial_point
                # param_tgt = self.initial_point[param.name.replace('MNIST_1', 'MNIST_0')]
                param_tgt = self.initial_point[param.name]
                param_tgt_value = param_tgt._ivar.value()
                param_tgt_tensor = param_tgt_value.get_tensor()
                param_tgt_np = np.array(param_tgt_tensor)
                #
                alpha = 0.1
                grod_decay = 0.99
                epoch = self.epoch // 5000
                # print(epoch)
                norm1 = np.linalg.norm(param_np)
                inerp = np.dot(param_np.flatten(), (param_np - param_tgt_np).flatten())
                coef = alpha * grod_decay ** epoch
                if not self.start_flag:
                    grad_np -= (-coef * (param_np - param_tgt_np) + coef * inerp / (norm1 ** 2) * grad_np)
                    grad_tensor.set(grad_np, fluid.framework._current_expected_place())
                    self.epoch += 1

            # np_param = np.array(param)
            # np_grad = np.array(grad)
            # norm1 = np.linalg.norm(np_grad)
            # inerp = np.dot(np_grad.flatten(),

            # tensor.set(np_grad_like.astype(np.float32), fluid.framework._current_expected_place())

            # if 'weights' in param.name:
            #     print('param name is ', param.name, 'grad name is ', grad.name )

            # for name, param in model.named_parameters():
            #     if not name.startswith('fc.'):
            #         norm1 = torch.norm(param.grad)
            #         inerp = torch.dot(param.grad.view(-1, ), (param - param_dict[name]).view(-1, ))
            #         coef = alpha * args.grod_decay ** epoch
            #         param.grad.data.sub_(-coef * (param - param_dict[name]) + coef * inerp / (norm1 ** 2) * param.grad)

        if self.start_flag:
            self.start_flag = False

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        if framework.in_dygraph_mode():
            framework._dygraph_tracer()._clear_ops()

        return optimize_ops, params_grads
