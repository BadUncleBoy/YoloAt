# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Vscode
#   File name   : train.py
#   Author      : 
#   Created date: 
#   Description :
#
#================================================================
import tensorflow as tf
import numpy as np
from core import utils, Config
from core.YoloAt_13 import YoloAt_13
from core.YoloAt_52 import YoloAt_52
from core.dataset import dataset, Parser

#learning_rate = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

def network_input( choice):
    parser         = Parser(int(choice))
    trainset       = dataset(parser, Config.Train_tfrecord, Config.BATCH_SIZE, shuffle=Config.SHUFFLE_SIZE)
    validset       = dataset(parser, Config.Valid_tfrecord, 1, shuffle=None)
    example        = tf.cond(is_training, lambda: trainset.get_next(), lambda: validset.get_next())
    images, y_true = example
    return images, y_true

def build_network(images, y_true, choice):
    if(choice == '1'):
        print("\n\tBuild YoloAt-13 Network for Train\n")
        model = YoloAt_13()
        Config.Update_vars = Config.Update_vars_13
        Config.SAVE_CKPT   = Config.SAVE_CKPT_13
    else :
        print("\n\tBuild YoloAt-52 Network for Train\n")
        model = YoloAt_52()
        Config.Update_vars = Config.Update_vars_52
        Config.SAVE_CKPT   = Config.SAVE_CKPT_52
    with tf.variable_scope('yoloAt'):
        pred_feature_map = model.forward(images, is_training=is_training)
        loss             = model.compute_loss(pred_feature_map, y_true)
        y_pred           = model.predict(pred_feature_map)
    return loss, y_pred

def optimize(global_step, loss):
    learning_rate = tf.train.exponential_decay(Config.Learning_Rate, global_step, decay_steps=Config.DECAY_STEPS, 
                                                decay_rate=Config.DECAY_RATE, staircase=True)
    optimizer   = tf.train.AdamOptimizer(learning_rate)
    update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_vars = tf.contrib.framework.get_variables_to_restore(include=[Config.Update_vars])
    '''
    count = 0
    for i in range(len(update_vars)):    
        a = 1
        num_list = update_vars[i].shape.as_list()
        for m in num_list:
            a = a*int(m) 
        count = count + a
    print(count)
    '''
    # set dependencies for BN ops
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, var_list=update_vars, global_step=global_step)
    return train_op

def train():
    with tf.Session() as sess:
        #is_training = tf.placeholder(tf.bool)
        global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

        #choice = input("\nchoice to build network\n\t(1)YoloAt-13\n\t(2)YoloAt-52\n\nYour choice:")
        choice = '2'
        images, y_true = network_input(choice)
        loss, y_pred   = build_network(images, y_true, choice)
        
        #restore vars
        restore_vars = tf.contrib.framework.get_variables_to_restore(include=[Config.Restore_vars])
        saver_to_restore = tf.train.Saver(var_list=restore_vars)
        saver_to_restore.restore(sess, Config.RESTORE_CKPT)
        saver = tf.train.Saver(max_to_keep=Config.MAX_TO_KEEP)

        train_op = optimize(global_step,loss[0])

        #variables initializer
        if(tf.__version__.startswith("0.") and int(tf.__version__.split(".")[1])<12):
            sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
        else: 
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        last_map = -1
        for step in range(Config.STEPS):
            run_items = sess.run([train_op, y_pred, y_true] + loss, feed_dict={is_training:True})
            print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
                %(step+1, run_items[4], run_items[5], run_items[6], run_items[7]))
            if (step+1) % Config.SAVE_INTERNAL == 0:
                MAP = evaluate(sess, y_pred, y_true)
                print("\n=> STEP %10d Valid Dataset MAP:%7.4f\n"%(step+1, MAP))
                if MAP > last_map:
                    last_map = MAP
                    saver.save(sess, save_path=Config.SAVE_CKPT, global_step=step+1)

def evaluate(sess, y_pred, y_true):
    NUM_CLASSES = Config.NUM_CLASSES
    CLASSES     = Config.CLASSES
    all_detections   = []
    all_annotations  = []
    all_aver_precs   = {CLASSES[i]:0. for i in range(NUM_CLASSES)}
    for _ in range(868):
        y_pred_o, y_true_o = sess.run([y_pred, y_true],feed_dict={is_training:False})
        pred_boxes = y_pred_o[0]
        pred_confs = y_pred_o[1]
        pred_probs = y_pred_o[2]
            
        true_labels_list, true_boxes_list = [], []
        true_probs_temp = y_true_o[..., 5: ]
        true_boxes_temp = y_true_o[..., 0:4]
        object_mask     = true_probs_temp.sum(axis=-1) > 0
        true_probs_temp = true_probs_temp[object_mask]
        true_boxes_temp = true_boxes_temp[object_mask]

        true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
        true_boxes_list  += true_boxes_temp.tolist()

        pred_boxes, pred_scores, pred_labels = utils.cpu_nms(pred_boxes, pred_confs*pred_probs, NUM_CLASSES,
                                                        score_thresh=0.3, iou_thresh=0.5)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:,0:2], true_boxes[:,2:4]

        true_boxes[:,0:2] = box_centers - box_sizes / 2.
        true_boxes[:,2:4] = true_boxes[:,0:2] + box_sizes
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()

        all_detections.append( [pred_boxes, pred_scores, pred_labels_list])
        all_annotations.append([true_boxes, true_labels_list])

    for idx in range(NUM_CLASSES):
        true_positives  = []
        scores = []
        num_annotations = 0

        for i in range(len(all_annotations)):
            pred_boxes, pred_scores, pred_labels_list = all_detections[i]
            true_boxes, true_labels_list              = all_annotations[i]
            detected                                  = []
            num_annotations                          += true_labels_list.count(idx)

            for k in range(len(pred_labels_list)):
                if pred_labels_list[k] != idx: continue

                scores.append(pred_scores[k])
                ious = utils.bbox_iou(pred_boxes[k:k+1], true_boxes)
                m    = np.argmax(ious)
                if ious[m] > 0.5 and pred_labels_list[k] == true_labels_list[m] and m not in detected:
                    detected.append(m)
                    true_positives.append(1)
                else:
                    true_positives.append(0)

        num_predictions = len(true_positives)
        true_positives  = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sorted by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)
        # compute recall and precision
        recall    = true_positives / np.maximum(num_annotations, np.finfo(np.float64).eps)
        precision = true_positives / np.maximum(num_predictions, np.finfo(np.float64).eps)
        # compute average precision
        average_precision = utils.compute_ap(recall, precision)
        all_aver_precs[CLASSES[idx]] = average_precision
    MAP = sum(all_aver_precs.values()) / NUM_CLASSES
    return MAP
if __name__ == '__main__':
    train()
