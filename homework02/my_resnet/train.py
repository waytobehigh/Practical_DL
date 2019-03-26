import getopt
import sys
import torch

import numpy as np

from model import ResNet, ResBlock, train_model, evaluate_model
from utils import make_datasets, make_datagens


from warnings import filterwarnings
filterwarnings('ignore')

N_BLOCKS = 5
SHOW_PROGRESS = False
PROFLE = False
DATASET_DIR = 'tiny-imagenet-200/'
N_EPOCHS = 100
BATCH_SIZE = 50
CHECKPOINTING = False


def process_argv(argv):
    global N_BLOCKS
    global SHOW_PROGRESS
    global PROFLE
    global DATASET_DIR
    global N_EPOCHS
    global BATCH_SIZE
    global CHECKPOINTING

    try:
        opts, args = getopt.getopt(
            argv, 'hd:n:b:', [
                'help',
                'show_progress',
                'profile',
                'checkpointing',
                'dataset_dir=',
                'n_epochs=',
                'n_blocks=',
                'batch_size='
            ]
        )
    except getopt.GetoptError:
        print('train.py -d <dataset_dir> -n <n_epochs> -b <batch_size>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('train.py -d <dataset_dir> -n <n_epochs> -b <batch_size>')
            sys.exit()
        elif opt in ('--show_progress',):
            SHOW_PROGRESS = True
        elif opt in ('--profile',):
            PROFLE = True
        elif opt in ('--checkpointing',):
            CHECKPOINTING = True
        elif opt in ("-d", "--dataset_dir"):
            DATASET_DIR = arg
        elif opt in ("-n", "--n_epochs"):
            N_EPOCHS = int(arg)
        elif opt in ("--n_blocks",):
            N_BLOCKS = int(arg)
        elif opt in ("-b", "--batch_size"):
            BATCH_SIZE = int(arg)


def main(argv):
    process_argv(argv)

    datasets = make_datasets(DATASET_DIR)
    train_batch_gen, val_batch_gen, test_batch_gen = make_datagens(datasets, batch_size=BATCH_SIZE, shuffle=True)

    labels = sorted(np.unique([data[1] for data in datasets[0]]))
    n_classes = len(labels)

    model = ResNet(ResBlock, N_BLOCKS, n_classes, checkpointing=CHECKPOINTING)
    train_model(model, train_batch_gen, val_batch_gen, n_epochs=N_EPOCHS, plot_graph=False,
                show_progress=SHOW_PROGRESS, profile=PROFLE)

    logs = evaluate_model(model, test_batch_gen)

    print('TEST LOSS: {:.5}\nTEST ACC: {:.3}%'.format(np.mean(logs.val_loss_), np.mean(logs.val_accuracy_) * 100))
    if PROFLE:
        print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")


if __name__ == '__main__':
    main(sys.argv[1:])
