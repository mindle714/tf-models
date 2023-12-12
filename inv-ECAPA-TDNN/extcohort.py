'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader_v2 import train_loader_v2
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=500,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/data/hejung/vox2/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/data/sv/vox2/dev/aac/",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/data/hejung/vox1/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/data/sv/vox1/test/wav/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

parser.add_argument('--use_full', dest='use_full', action='store_true')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

## Define the data loader
batch_size = 400 if not args.use_full else 1
trainloader = train_loader_v2(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)

if args.model == "":
	import sys
	sys.exit(0)

s = ECAPAModel(**vars(args), lr=1, lr_decay=1)
s.load_parameters(args.model)

save_dir = os.path.dirname(args.model)
name = '{}.cohort'.format(os.path.splitext(os.path.basename(args.model))[0])
if args.use_full:
        name = '{}_full'.format(name)
save_path = os.path.join(save_dir, name)
print(save_path)

import numpy as np
c = s.cohorts(trainLoader, args.use_full)

print(c.shape)

np.save(save_path, c)
