import cv2
import yaml
import sys
import time
import numpy as np
import os
import tensorflow as tf
from modules.models import RetinaFaceModel
from PIL import Image 
from absl import logging
from modules.dataset import load_tfrecord_dataset

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_dataset(cfg, priors, shuffle=True, buffer_size=10240):
    """load dataset"""
    logging.info("load dataset from {}".format(cfg['dataset_path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=cfg['dataset_path'],
        batch_size=cfg['batch_size'],
        img_dim=cfg['input_size'],
        using_bin=cfg['using_bin'],
        using_flip=cfg['using_flip'],
        using_distort=cfg['using_distort'],
        using_encoding=True,
        priors=priors,
        match_thresh=cfg['match_thresh'],
        ignore_thresh=cfg['ignore_thresh'],
        variances=cfg['variances'],
        shuffle=shuffle,
        buffer_size=buffer_size)
    return dataset


class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()


###############################################################################
#   Testing                                                                   #
###############################################################################
def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


###############################################################################
#   Visulization                                                              #
###############################################################################
def draw_bbox_landm(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    text = "{:.4f}".format(ann[15])
    cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        cv2.circle(img, (int(ann[4] * img_width),
                         int(ann[5] * img_height)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(ann[6] * img_width),
                         int(ann[7] * img_height)), 1, (0, 255, 255), 2)
        cv2.circle(img, (int(ann[8] * img_width),
                         int(ann[9] * img_height)), 1, (255, 0, 0), 2)
        cv2.circle(img, (int(ann[10] * img_width),
                         int(ann[11] * img_height)), 1, (0, 100, 255), 2)
        cv2.circle(img, (int(ann[12] * img_width),
                         int(ann[13] * img_height)), 1, (255, 0, 100), 2)


def draw_anchor(img, prior, img_height, img_width):
    """draw anchors"""
    x1 = int(prior[0] * img_width - prior[2] * img_width / 2)
    y1 = int(prior[1] * img_height - prior[3] * img_height / 2)
    x2 = int(prior[0] * img_width + prior[2] * img_width / 2)
    y2 = int(prior[1] * img_height + prior[3] * img_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

def get_one_image(images):
        img_list = []
        padding = 200
        for img in images:
            img_list.append(cv2.imread(img))
        max_width = []
        max_height = 0
        for img in img_list:
            max_width.append(img.shape[0])
            max_height += img.shape[1]
        w = np.max(max_width)
        h = max_height + padding

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((h, w, 3), dtype=np.uint8)

        current_y = 0  # keep track of where your current image was last placed in the y coordinate
        for image in img_list:
            # add an image to the final array and increment the y coordinate
            final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
            current_y += image.shape[0]
        cv2.imwrite('out.png', final_image)

def get_bbox_imgs(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    imgs = None
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    d = 6
    x1 -= d
    if(x1 < 0):
      x1 = 0
    x2 += d
    if(x2 > img_width):
      x2 = img_width
    y1 -= d
    if(y1 < 0):
      y1 = 0
    y2 += d
    if(y2 > img_height):
      y2 = img_height
    return img[y1:y2, x1:x2]


###############################################################################
#   Detect faces                                                              #
###############################################################################
def get_model(cfg_path):
    # init
    IOU_TH = 0.5
    SCORE_TH = 0.4
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=IOU_TH, score_th=SCORE_TH)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()
        
    return model

def get_faces(model, cfg_path, img_path, save_path, **keys):
    
    print("[*] Processing on single image {}".format(img_path))
    
    if not os.path.exists(img_path):
        print(f"cannot find image path from {img_path}")
        exit()

    cfg = load_yaml(cfg_path)
            
    img_raw = cv2.imread(img_path)
    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # run model
    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    # draw and save results
    imgs = []
    DIM = 110;
    save_img_path = os.path.join('/content/FaceDataset/', save_path, os.path.basename(img_path))
    if(keys['full']):
        save_img_path = os.path.join('/content/FaceDatasetFull/', save_path, os.path.basename(img_path))
    elif keys['extra']:
        save_img_path = os.path.join('/content/FaceDatasetExtra/', save_path, os.path.basename(img_path))
    img_last = Image.new('RGB', (DIM, DIM))
    for prior_index in range(9):
      if(prior_index < len(outputs)):
        img = get_bbox_imgs(img_raw, outputs[prior_index], img_height_raw, img_width_raw)
        img = cv2.resize(img, (DIM, DIM)) 
        imgs.append(img)
        if(keys['full']):
            img_last = img 
      else:
        imgs.append(img_last)
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
    imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2RGB)
    return imgf

    print(f"[*] save result at {save_img_path}")

