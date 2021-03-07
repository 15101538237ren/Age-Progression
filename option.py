# -*- coding: utf-8
import os
import argparse
from util import default_train_results_dir
def str_to_gender(s):
    s = str(s).lower()
    if s in ('m', 'man', '0'):
        return 0
    elif s in ('f', 'female', '1'):
        return 1
    else:
        raise KeyError("No gender found")

parser = argparse.ArgumentParser(description="AgeProgression on PyTorch.")
parser.add_argument("--mode", choices=["train", "test"], default="train")

# train params
parser.add_argument(
        '--models-saving', '--ms', dest='models_saving', choices=('always', 'last', 'tail', 'never'), default='always', type=str,
        help='Model saving preference.{br}'
             '\talways: Save trained model at the end of every epoch (default){br}'
             '\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}'
             '\tlast: Save trained model only at the end of the last epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a costly operation.{br}'
             '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a cheap operation.{br}'
             '\tnever: Don\'t save trained model{br}'
             '\tUse this option if you only wish to collect statistics and validation results.{br}'
             'All options except \'never\' will also save when interrupted by the user.'.format(br=os.linesep)
    )
parser.add_argument('--epochs', '-e', default=150, type=int)
parser.add_argument('--batch-size', '--bs', dest='batch_size', default=64, type=int)
parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-5, type=float)
parser.add_argument('--learning-rate', '--lr', dest='learning_rate', default=2e-4, type=float)
parser.add_argument('--b1', '-b', dest='b1', default=0.5, type=float)
parser.add_argument('--b2', '-B', dest='b2', default=0.999, type=float)
parser.add_argument('--shouldplot', '--sp', dest='sp', default=False, type=bool)

# test params
parser.add_argument('--age', '-a', required=False, type=int)
parser.add_argument('--gender', '-g', required=False, type=str_to_gender)
parser.add_argument('--watermark', '-w', action='store_true')

# shared params
parser.add_argument('--cpu', '-c', action='store_true', help='Run on CPU even if CUDA is available.')
parser.add_argument('--load', '-l', required=False, default=None, help='Trained models path for pre-training or for testing')
parser.add_argument('--input', '-i', default=None, help='Training dataset path (default is {}) or testing image path'.format(default_train_results_dir()))
parser.add_argument('--output', '-o', default='')
parser.add_argument('-z', dest='z_channels', default=50, type=int, help='Length of Z vector')


