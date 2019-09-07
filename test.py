import tensorflow as tf
import numpy as np 
import os
from PIL import Image

from core import utils
from core.YoloAt_13 import YoloAt_13
from core.YoloAt_52 import YoloAt_52
from core import Config

with tf.Session() as sess:
    inputs_image        = tf.placeholder(tf.float32,[1,Config.IMAGE_H ,Config.IMAGE_W ,3])
    choice = input(Config.F_Clear + "Choice to build network\n\t" +Config.F_Yellow + "(1)YoloAt-13\n\t(2)YoloAt-52\n" + Config.F_Init + "Your choice:")
    if choice =='1':
        print(Config.F_Purple + "Build YoloAt-13 network for predicting" + Config.F_Init)
        model       = YoloAt_13()
        Config.CKPT = Config.CKPT_13
    else:
        print(Config.F_Purple + "Build YoloAt-52 network for predicting" + Config.F_Init)
        model       = YoloAt_52()
        Config.CKPT = Config.CKPT_52
    with tf.variable_scope("yoloAt"):
        p_feature_map       = model.forward(inputs_image,is_training=False)

    sess.run(tf.global_variables_initializer())

    saver=tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,Config.CKPT)
    print(Config.F_Yellow + "starting predicting" + Config.F_Init)    
    for pic in os.listdir('./data/test_pic'):
        img         = Image.open('./data/test_pic/' + pic)
        img_resized = np.array(img.resize(size=(Config.IMAGE_H,Config.IMAGE_W)),dtype=np.float32)
        img_resized = img_resized / 255
        boxes, confs, probs = sess.run(model.predict(p_feature_map), feed_dict={inputs_image:np.expand_dims(img_resized,axis=0)})
        scores = confs * probs
        boxes,scores,labels = utils.cpu_nms(boxes,scores,Config.NUM_CLASSES)
        image = utils.draw_boxes(img, boxes, scores, labels, Config.CLASSES,[Config.IMAGE_H,Config.IMAGE_W],show=False)
        image.save('./jpg/pre_{}'.format(pic))
        print("finished predicting {}".format(Config.F_Red + pic + Config.F_Init))
        
