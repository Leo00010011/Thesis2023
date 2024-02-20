from __future__ import division
from localization.visualization import visualize_rois
from processing.reading import get_cropped_img
from processing.preprocessing import enhance_image_quality
import cv2
import numpy as np
import sys
import pickle
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from localization.keras_frcnn import roi_helpers
import time
import os
import tensorflow as tf
import multiprocessing
import json
from pathlib import Path
# to use CPU instead of GPU in tensorflow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('.\\localization\\')

class Faster_RCNN:
    def __init__(self, config_output_filename="config.pickle", weight_path='model_frcnn.hdf5') -> None:
        self.config_output_filename = config_output_filename
        self.class_to_color = None
        self.model_rpn = None
        self.model_classifier_only = None
        self.model_classifier = None
        self.class_mapping = None
        self.weight_path = weight_path
        self.C = None

    def prepare(self):
        '''
        Inicializa la red y carga los pesos, después de llamar este método se puede llamar `get_roi`
        '''
        K.set_learning_phase(0)

        sys.setrecursionlimit(40000)

        with open(self.config_output_filename, 'rb') as f_in:
            self.C = pickle.load(f_in)

        if self.C.network == 'resnet50':
            import localization.keras_frcnn.resnet as nn
        elif self.C.network == 'vgg':
            import localization.keras_frcnn.vgg as nn

        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_180 = False

        class_mapping = self.C.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        self.class_mapping = {v: k for k, v in class_mapping.items()}
        print(class_mapping)
        self.class_to_color = {class_mapping[v]: np.random.randint(
            0, 255, 3) for v in class_mapping}
        self.C.num_rois = 32

        if self.C.network == 'resnet50':
            num_features = 1024
        elif self.C.network == 'vgg':
            num_features = 512

        if K.image_data_format() == 'channels_first':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * \
            len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(
            class_mapping), trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model(
            [feature_map_input, roi_input], classifier)

        self.model_classifier = Model(
            [feature_map_input, roi_input], classifier)

        try:
            print('Loading weights from {}'.format(self.weight_path))
            self.model_rpn.load_weights(self.weight_path, by_name=True)
            self.model_classifier.load_weights(self.weight_path, by_name=True)
        except Exception as e:
            print(e)
            print('Could not load pretrained model weights. Weights can be found in our one drive folder \
				https://1drv.ms/u/s!AoVbNqANaM4XgdIfpWPQmDwEtNQGZg?e=FMbVw2')

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def get_roi(self, img):
        '''
        Dada una imagen retorna los bboxes indicando las posiciones de las UPD y las probabilidades

        [[x1,y1,x2,y2,prob]...]
        '''
        X, ratio = format_img(img, self.C)

        if K.image_data_format() == 'channels_last':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_data_format(
        ), overlap_thresh=0.7, max_boxes=self.C.roi_proposal)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        bbox_threshold = 0.8

        boxes = []

        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(
                R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(
                        x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(
                bbox, np.array(probs[key]), overlap_thresh=0.3, max_boxes=self.C.roi_proposal)

            for jk in range(new_boxes.shape[0]):

                (x1, y1, x2, y2) = new_boxes[jk, :]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(
                    ratio, x1, y1, x2, y2)
                row = [real_x1, real_y1, real_x2, real_y2, new_probs[jk]]
                boxes.append(row)

        tf.keras.backend.clear_session()
        return boxes





def format_img_size(img, C):
    """ formats the image size based on config """
    # im_size = 480
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height),
                     interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def get_clean_img_path(cleaning_data_path):
    cleaning_data = None
    with open(cleaning_data_path, 'r') as f:
        cleaning_data = json.load(f)

    return [item['img_path'] for item in cleaning_data['good']]


def correct_path(path):
    return str(Path(path))


def join_paths(path_A, path_B):
    return str(Path('\\'.join([path_A, path_B])))


def get_good_img_path(data_path, json_path):
    cleaning_data = None
    with open(json_path, 'r') as f:
        cleaning_data = json.load(f)

    img_path = [join_paths(data_path, path)
                for path in cleaning_data['img_path']]
    label_path = [join_paths(data_path, path)
                  for path in cleaning_data['label_path']]
    return img_path, label_path


def preprocess_img(img):
    img = get_cropped_img(img)
    return enhance_image_quality(img)


def get_already_processed_imgs(data_path, result):
    return [join_paths(data_path, bbox['path']) for bbox in result]

def get_images(data_path):
    data_path = Path(data_path)
    images = [str(images) for images in data_path.rglob('*.jpg')]
    images.sort()
    return images

def check_rois(data_path,results_path):
    result = None
    with open(results_path, 'r') as f:
        result = json.load(f)
    for roi in result:
        img = cv2.imread(join_paths(data_path, roi['path']))
        if not roi['empty']:
            visualize_rois(img, roi['bboxes'], (0, 0, 255))
        else:
            cv2.imshow('NOT ROI', img)
            cv2.waitKey()
        cv2.destroyAllWindows()


def check_preprocess(data_path):
    result = None
    with open('rois.json', 'r') as f:
        result = json.load(f)
    for roi in result:
        img = cv2.imread(join_paths(data_path, roi['path']))
        img = preprocess_img(img)
        cv2.imshow('NOT ROI', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def roi_process(data_path, config_path, detector_wights, results_path):
    '''
    Proceso que lee las imágenes de `data_path` con `config_path` y `detector_weights` inicializa el detector y guarda los resultados en
    `result_path`
    
    Está preparado para que si se interrumpe y se vuelve a empezar continue por donde se quedó
    '''
    print('LOADING IMAGES')
    img_path_list = get_images(data_path)
    result = []
    count = 0
    # quitar detecciones ya hechas
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            result = json.load(f)
        already_processed = get_already_processed_imgs(data_path, result)
        path_list = []
        for path in img_path_list:
            if str(path) in already_processed:
                count += 1
            else:
                path_list.append(path)
    else:
        path_list = img_path_list

    print('PREPARING')
    detector = Faster_RCNN(config_path, detector_wights)
    detector.prepare()

    for img_path in path_list:
        print(f'>>>>>{count + 1}/{len(path_list)}')
        img_path = str(img_path)
        img = cv2.imread(img_path)
        st = time.time()
        bbox_list = detector.get_roi(img)
        elapsed = time.time() - st
        if len(bbox_list) == 0:
            result.append({
                'path': img_path[len(data_path):],
                'empty': True
            })
        else:
            bboxes = [{'x1': bbox_list[i][0],
                       'y1': bbox_list[i][1],
                       'x2': bbox_list[i][2],
                       'y2': bbox_list[i][3],
                       'prob': float(bbox_list[i][4])
                       } for i in range(len(bbox_list))]
            result.append({'path': img_path[len(data_path):],
                           'empty': False,
                           'bboxes': bboxes,
                           'time': elapsed})

        with open(results_path, 'w') as f:
            json.dump(result, f)
        count += 1

    return result

