import logging
from datetime import datetime

import numpy as np
import torch

from ignite.metrics import Metric
import losswise

from common_utils.misc import tuplify
from common_utils.torch.helpers import variable, save_weights


class InferenceFunction(object):
    def __init__(self, model):
        self.model = model

    def _forward_pass(self, batch):
        batch = variable(batch)

        inputs, targets = batch
        inputs = tuplify(inputs)
        targets = tuplify(targets)

        outputs = self.model(*inputs)

        outputs = tuplify(outputs)

        return outputs, targets

    def __call__(self, engine, batch):
        self.model.eval()

        outputs, _ = self._forward_pass(batch)

        return outputs


class UpdateFunction(InferenceFunction):
    def __init__(self, criterion, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.criterion = criterion
        self.optimizer = optimizer

    def backward_completed_callback(self):
        return

    def __call__(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs, targets = self._forward_pass(batch)

        losses = self.criterion(*outputs, *targets)
        losses = tuplify(losses)
        loss_total = sum(losses)

        loss_total.backward()
        self.backward_completed_callback()
        self.optimizer.step()

        losses_values = [float(l) for l in losses]
        if len(losses_values) == 0:
            losses_values = losses_values[0]

        return losses_values


class GradientClippingUpdateFunction(UpdateFunction):
    def __init__(self, parameters, clip_grad_norm, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parameters = parameters
        self.clip_grad_norm = clip_grad_norm

    def backward_completed_callback(self):
        torch.nn.utils.clip_grad_norm(self.parameters, self.clip_grad_norm)


class LossAggregator(Metric):
    def __init__(self, losses_names, mean=True):
        if not isinstance(losses_names, (list, tuple)):
            losses_names = (losses_names,)

        self.losses_names = losses_names
        self.mean = mean

        self._epoch_losses = {}
        self.nb_batches = 0

        super().__init__()

    def reset(self):
        for loss_name in self.losses_names:
            self._epoch_losses[loss_name] = 0

        self.nb_batches = 0

    def update(self, output):
        for loss_name, loss_value in zip(self.losses_names, output):
            self._epoch_losses[loss_name] += loss_value

        self.nb_batches += 1

    def compute(self):
        if self.mean:
            losses = {l: self._epoch_losses[l] / self.nb_batches for l in self.losses_names}
        else:
            losses = {l: self._epoch_losses[l] for l in self.losses_names}

        return losses


class LosswiseLogger(object):
    def __init__(self, api_key, metric_name, model_name=None, model_params=None):
        super().__init__()

        self.metric_name = metric_name

        losswise.set_api_key(api_key)
        self._session = losswise.Session(tag=model_name, params=model_params)
        self._graph = self._session.graph('loss', kind='min')

    def _extract_losses(self, engine, prefix):
        losses = engine.state.metrics[self.metric_name]
        losses = {f'{prefix}_{l}': v for l, v in losses.items()}
        return losses

    def __call__(self, engine_train, engine_val=None):
        epoch = engine_train.state.epoch

        losses = self._extract_losses(engine_train, 'train')
        if engine_val is not None:
            losses_val = self._extract_losses(engine_val, 'val')
            losses.update(**losses_val)

        self._graph.append(epoch, losses)

    def done(self):
        self._session.done()


class ModelSaver(object):
    def __init__(self, model, model_params, filename, metric_name, loss_name=None):
        super().__init__()

        self.filename = filename
        self.model = model
        self.model_params = model_params

        self.metric_name = metric_name
        self.loss_name = loss_name

        self._best_loss = np.inf

    def __call__(self, engine):
        losses = engine.state.metrics[self.metric_name]

        if self.loss_name is None:
            loss_current = sum(losses.values())
        else:
            loss_current = losses[self.loss_name]

        if loss_current <= self._best_loss:
            save_weights(self.model, self.filename)
            self._best_loss = loss_current


# class SacredInfoCallback(object):
#     def __init__(self, exp, metric_name):
#         self.exp = exp
#         self.metric_name = metric_name
#
#         self.exp.info['losses_train'] = []
#         self.exp.info['losses_val'] = []
#
#     def __call__(self, state, state_val):
#         losses_train = state.metrics[self.metric_name]
#         losses_val = state_val.metrics[self.metric_name]
#
#         self.exp.info['losses_train'].append(losses_train)
#         self.exp.info['losses_val'].append(losses_val)
#

# class EvaluatorRunner(object):
#     def __init__(self, evaluator, callbacks=None):
#         super().__init__()
#
#         self.evaluator = evaluator
#
#         if callbacks is not None and not isinstance(callbacks, (tuple, list)):
#             callbacks = [callbacks, ]
#         self.callbacks = callbacks
#
#     def __call__(self, engine, state, data):
#         state_val = self.evaluator.run(data)
#
#         if self.callbacks is not None:
#             should_terminate = any(callback(state, state_val) for callback in self.callbacks)
#             if should_terminate:
#                 engine.should_terminate = True
#
#
# class EarlyStoppingLossThresholdCallback(object):
#     def __init__(self, threshold, metric_name, loss_name=None):
#         super().__init__()
#
#         self.threshold = threshold
#         self.metric_name = metric_name
#         self.loss_name = loss_name
#
#     def __call__(self, state, state_val):
#         losses = state_val.metrics[self.metric_name]
#
#         if self.loss_name is None:
#             loss_current = sum(losses.values())
#         else:
#             loss_current = losses[self.loss_name]
#
#         should_terminate = loss_current < self.threshold
#
#         return should_terminate
#
#
# class EarlyStoppingElapsedTimeCallback(object):
#     def __init__(self, max_elapsed_time):
#         super().__init__()
#
#         self.max_elapsed_time = max_elapsed_time
#
#         self._time_start = datetime.now()
#
#     def __call__(self, state, state_val):
#         time_now = datetime.now()
#
#         time_delta = time_now - self._time_start
#
#         should_terminate = time_delta > self.max_elapsed_time
#
#         return should_terminate
#
#
