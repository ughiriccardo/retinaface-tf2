from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from PIL import Image 
from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output,
                           get_bbox_imgs, get_one_image, get_faces)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')

def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th, score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    if not os.path.exists(FLAGS.img_path):
        print(f"cannot find image path from {FLAGS.img_path}")
        exit()

    print("[*] Processing on single image {}".format(FLAGS.img_path))

    img_raw = cv2.imread(FLAGS.img_path)
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if FLAGS.down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor, fy=FLAGS.down_scale_factor, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # run model
    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    # draw and save results
    imgs = []
    DIM = 64;
    save_img_path = os.path.join('data/out_' + os.path.basename(FLAGS.img_path))
    for prior_index in range(9):
      if(prior_index < len(outputs)):
        img = get_bbox_imgs(img_raw, outputs[prior_index], img_height_raw, img_width_raw)
        img = cv2.resize(img, (DIM, DIM)) 
        imgs.append(img)
      else:
        imgs.append(Image.new('RGB', (DIM, DIM)))
    imga = imgs[0]
    for img in imgs[1:3]:
      imga = np.concatenate((imga, img), axis=1)     
    imgb = imgs[3]
    for img in imgs[4:6]:
      imgb = np.concatenate((imgb, img), axis=1)  
    imgf = np.concatenate((imga, imgb), axis=0)
    imgc = imgs[6]
    for img in imgs[7:9]:
      imgc = np.concatenate((imgc, img), axis=1)  
    imgf = np.concatenate((imgf, imgc), axis=0)
    cv2.imwrite(save_img_path, imgf)

    print(f"[*] save result at {save_img_path}")

    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
