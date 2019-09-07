# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Vscode
#   File name   : Config.py
#   Author      : 
#   Created date: 
#   Description :
#
#================================================================
from core import utils

IMAGE_H         = 416
IMAGE_W         = 416
CLASSES         = utils.read_coco_names('./data/wider/wider.names')
NUM_CLASSES     = len(CLASSES)
ANCHORS         = utils.get_anchors('./data/wider/wider_anchors.txt', IMAGE_H, IMAGE_W)


#Training
BATCH_SIZE      = 16
STEPS           = 40000
Learning_Rate   = 0.002                                # beginning laerining
DECAY_STEPS     = 5000                                #
DECAY_RATE      = 0.9                                 # every decay_steps,new_lr=current_lr *dacay_rate
SHUFFLE_SIZE    = 200                                 # for data shuffle
MAX_TO_KEEP     = 2                                   # training checkpoint
SAVE_INTERNAL   = 400                                 # every each steps,save weights to checkpoint
EVAL_INTERNAL   = 100                                 # every each steps, evaluate training result
Train_tfrecord  = "./data/wider/train.tfrecords"
Valid_tfrecord  = './data/wider/valid.tfrecords'
RESTORE_CKPT    = './checkpoint/yolov3.ckpt'            # path point to restore checkpoint(darknet-53)
SAVE_CKPT       = ' '                                   # path point to save checkpoint
Restore_vars    = 'yoloAt/darknet-53'                   # only update this variable_scope's weights
Update_vars     = ' '                                   #n eed to be changed

SAVE_CKPT_13    = 'checkpoint/wider/yoloAt-13/yoloAt.ckpt' # for different network,save_ckpt saves in different dirs
SAVE_CKPT_52    = 'checkpoint/wider/yoloAt-52/yoloAt.ckpt'

Update_vars_13  = 'yoloAt/yoloAt-13'                   # for different network,we update diffenent parts of the network
Update_vars_52  = 'yoloAt/yoloAt-52'



batch_norm_decay=0.9
leaky_relu=0.1
#Testing
CKPT            = ' '
CKPT_13         = './checkpoint/wider/yoloAt-13/yoloAt.ckpt-4400'
CKPT_52         = './checkpoint/wider/yoloAt-52/yoloAt.ckpt-40000'

#Evaluate
Evaluate_tfrecord = './data/wider/valid.tfrecords'
Evaluate_ckpt     = './checkpoint/wider/yolov3.ckpt'

#Control  #control screen apperance
F_Black           = '\033[30m'
F_Red             = '\033[31m'
F_Yellow          = '\033[33m'
F_Purple          = '\033[36m'

F_Init            = '\033[0m'
F_Clear           = '\033[2J'

