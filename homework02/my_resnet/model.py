import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from collections import namedtuple
from tqdm import tqdm


def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


if in_ipynb():
    from IPython.display import clear_output


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=kernel_size // 2,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=kernel_size // 2,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.1, inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Sequential):
    def __init__(self, block, n_blocks, n_classes, inplanes=64, planes=64, kenel_size=7,
                 stride=2, zero_init_residual=False, checkpointing=True):
        super().__init__()

        self.inplanes = inplanes
        self.n_blocks = n_blocks
        self.kernel_size = kenel_size
        self.stride = stride
        self.checkpointing = checkpointing

        self.conv1 = nn.Conv2d(3, 64, kernel_size=kenel_size, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = self._make_blocks(block, planes, n_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes, n_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.relu(out)

        if self.checkpointing:
            out = checkpoint_sequential(self.blocks, int(np.sqrt(self.n_blocks)), out)
        else:
            out = self.blocks.forward(out)

        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out

    def _make_blocks(self, block, planes, n_blocks):
        return nn.Sequential(*[block(self.inplanes, planes) for _ in range(n_blocks)])


def compute_loss(model, X_batch, y_batch, C=0, criterion=nn.L1Loss(size_average=False)):
    logits = model.cuda()(X_batch.cuda())

    loss = nn.functional.cross_entropy(logits, y_batch.cuda()).mean()

    if C:
        reg_loss = 0
        for param in model.parameters():
            reg_loss += criterion(param, torch.zeros(param.shape).cuda())

        loss += C * reg_loss

    return logits, loss


Logger = namedtuple('Logger', ['train_loss_', 'train_accuracy_', 'val_loss_', 'val_accuracy_'])


def get_logger():
    return Logger(*[[] for _ in range(4)])


def compute_accuracy(y_true, y_pred):
    return np.mean((y_true.cpu() == y_pred.cpu()).numpy())


def evaluate_model(model, val_batch_gen, C=0, logs=None):
    if not logs:
        logs = get_logger()

    model.train(False)
    for X_batch, y_batch in val_batch_gen:
        logits, loss = compute_loss(model, X_batch, y_batch, C)
        y_pred = logits.max(1)[1].data
        logs.val_loss_.append(loss.cpu().data.numpy())
        logs.val_accuracy_.append(compute_accuracy(y_batch, y_pred))

    return logs


def train_model(model, train_batch_gen, val_batch_gen, n_epochs=50, C=0, plot_graph=True,
                show_progress=False, profile=False, **optimizer_params):
    opt = torch.optim.Adam(model.parameters(), **optimizer_params)
    logs = get_logger()

    range_gen = range(n_epochs)
    if show_progress:
        range_gen = tqdm(range_gen)

    if profile:
        batch_counters = []
        epoch_counters = []

    for epoch in range_gen:
        epoch_logs = get_logger()

        if profile:
            epoch_counter = time.perf_counter()

        model.train(True)
        for X_batch, y_batch in train_batch_gen:
            if profile:
                batch_counter = time.perf_counter()

            logits, loss = compute_loss(model, X_batch, y_batch, C)
            y_pred = logits.max(1)[1].data
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_logs.train_loss_.append(loss.cpu().data.numpy())
            epoch_logs.train_accuracy_.append(compute_accuracy(y_batch, y_pred))

            if profile:
                batch_counters.append(time.perf_counter() - batch_counter)


        evaluate_model(model, val_batch_gen, C, epoch_logs)

        if profile:
            epoch_counters.append(time.perf_counter() - epoch_counter)

        for attr in dir(epoch_logs):
            if attr.endswith('_') and not attr.endswith('__'):
                getattr(logs, attr).append(np.mean(getattr(epoch_logs, attr)))

        if plot_graph:
            if in_ipynb():
                clear_output()

            plt.figure(figsize=(15, 5))
            grid = np.arange(1, epoch + 2)

            fig = plt.subplot(121)
            fig.set_title('Loss in dependency on number of epochs')
            fig.plot(grid, logs.train_loss_, label='train')
            fig.plot(grid, logs.val_loss_, label='val')
            fig.set_xlabel('epoch')
            fig.set_ylabel('loss')
            fig.grid()

            fig = plt.subplot(122)
            fig.set_title('Accuracy in dependency on number of epochs')
            fig.plot(grid, logs.train_accuracy_, label='train')
            fig.plot(grid, logs.val_accuracy_, label='val')
            fig.set_xlabel('epoch')
            fig.set_ylabel('accuracy')
            fig.grid()

            plt.show()

    if profile:
        print('Average epoch time: {:.3}s\nAverage batch time: {:.2}s'.format(
            np.mean(epoch_counters), np.mean(batch_counters)))

    return logs
