import sys
sys.path.append(
    'C:\\Users\\53588\\Desktop\\Tesis\\dev\\SAM_faster_rcnn_performance')
from localization.detect_frcnn_dfu import join_paths
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import os
from processing.reading import get_cropped_img
from gzip import GzipFile
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import time

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32

# Model types:
TYPE_VIT_H = 'vit_h'
TYPE_VIT_L = 'vit_l'
TYPE_VIT_B = 'vit_b'
# Model paths:
WPATH_VIT_H = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_h_4b8939.pth"
WPATH_VIT_L = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_l_0b3195.pth"
WPATH_VIT_B = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_b_01ec64.pth"

SCAPE = 27
C = 99
S = 115
ENTER = 13
LEFT = 2424832
RIGHT = 2555904
embedding_path = "C:\\Users\\53588\\Desktop\\Results\\image_embeddings\\data.json"
MASK_COLOR = np.array([255, 144, 30], dtype=np.uint8)


def apply_the_mask(img, mask, color):
    beta = .30
    alpha = 1 - beta
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return cv2.addWeighted(img, alpha, mask_image, beta, 0), mask_image


class SAM_Model:
    def __init__(self, model_type, model_path) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.predictor = None

    def prepare(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

    def set_embedding(self, img):
        self.predictor.set_image(img)

    def save_embedding(self, path):
        if not self.predictor.is_image_set:
            raise RuntimeError('No hay embedding')
        res = {
            'original_size': self.predictor.original_size,
            'input_size': self.predictor.input_size,
            'features': self.predictor.features,
            'is_image_set': True
        }
        torch.save(res, path)

    def load_embedding(self, path):
        self.predictor.reset_image()
        res = torch.load(path, self.predictor.device)
        for k, v in res.items():
            setattr(self.predictor, k, v)

    def segmentate_from_points(self, fg_points, bg_points, multimask=True, mask_input=None):
        points = fg_points + bg_points
        labels = [1]*len(fg_points) + [0]*len(bg_points)
        mask, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=multimask,
            mask_input=mask_input
        )
        return mask, scores, logits

    def segmentate_from_midlepoint(self, x1, y1, x2, y2, multimask=True, mask_input=None):
        center = [x1 + (x2 - x1)//2, y1 + (y2 - y1)//2]
        input_points = [center]
        masks, scores, logits = self.segmentate_from_points(
            input_points, [], multimask, mask_input)
        return masks, scores, logits

    def segmentate_from_box(self, x1, y1, x2, y2, multimask=True, mask_input=None):
        input_box = np.array([x1, y1, x2, y2])
        mask, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=multimask,
            mask_input=mask_input
        )
        return mask, scores, logits

    def segmentate_from_box_and_middle_point(self, x1, y1, x2, y2, multimask=True, mask_input=None):
        input_box = np.array([x1, y1, x2, y2])
        center = [x1 + (x2 - x1)//2, y1 + (y2 - y1)//2]
        input_points = np.array([center])
        label = np.array([1])
        mask, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=label,
            box=input_box[None, :],
            multimask_output=multimask,
            mask_input=mask_input
        )
        return mask, scores, logits


def get_all_segmentations(sam_model: SAM_Model, x1, y1, x2, y2):
    img_shape = sam_model.predictor.original_size
    result = np.empty((24, *img_shape), dtype=np.dtype('bool'))
    final_scores = []
    # <<<<< MULTIMASK NO LOGITS >>>>> 9 + 3 scores + 3 logits
    # Midle point(save logics)
    # midle point
    result[0:3, :, :], scores, logits = sam_model.segmentate_from_midlepoint(
        x1, y1, x2, y2, multimask=True)
    final_scores.append([float(score) for score in scores])
    # Select best logits
    best_logits = logits[np.argmax(scores), :, :][None, :, :]
    # bbox
    result[3:6, :, :], scores, _ = sam_model.segmentate_from_box(
        x1, y1, x2, y2, multimask=True)
    final_scores.append([float(score) for score in scores])
    # Midle point + bbox
    result[6:9, :, :], scores, _ = sam_model.segmentate_from_box_and_middle_point(
        x1, y1, x2, y2, multimask=True)
    final_scores.append([float(score) for score in scores])

    # <<<<< MULTIMASK LOGITS >>>>>> 9 + 3 scores + 3 logits
    # midle point
    result[9:12, :, :], scores, logits = sam_model.segmentate_from_midlepoint(
        x1, y1, x2, y2, multimask=True, mask_input=best_logits)
    final_scores.append([float(score) for score in scores])
    # bbox
    result[12:15, :, :], scores, _ = sam_model.segmentate_from_box(
        x1, y1, x2, y2, multimask=True, mask_input=best_logits)
    final_scores.append([float(score) for score in scores])
    # Midle point + bbox
    result[15:18, :, :], scores, _ = sam_model.segmentate_from_box_and_middle_point(
        x1, y1, x2, y2, multimask=True, mask_input=best_logits)
    final_scores.append([float(score) for score in scores])

    # <<<< ONE MASK NO LOGITS >>>> 3
    # midle point
    result[18, :, :], _, _ = sam_model.segmentate_from_midlepoint(
        x1, y1, x2, y2, multimask=False)
    # bbox
    result[19, :, :], _, _ = sam_model.segmentate_from_box(
        x1, y1, x2, y2, multimask=False)
    # Midle point + bbox
    result[20, :, :], _, _ = sam_model.segmentate_from_box_and_middle_point(
        x1, y1, x2, y2, multimask=False)

    # <<<< ONE MASK Logits >>>> 3
    # midle point
    result[21, :, :], _, _ = sam_model.segmentate_from_midlepoint(
        x1, y1, x2, y2, multimask=False, mask_input=best_logits)
    # bbox
    result[22, :, :], _, _ = sam_model.segmentate_from_box(
        x1, y1, x2, y2, multimask=False, mask_input=best_logits)
    # Midle point + bbox
    result[23, :, :], _, _ = sam_model.segmentate_from_box_and_middle_point(
        x1, y1, x2, y2, multimask=False, mask_input=best_logits)

    return result, final_scores


def segmentate_with_simple_sam(data_path, roi_list_path, segmentation_folder, weights_path, weights_type):
    print('Initialasing model')
    model = SAM_Model(weights_type, weights_path)
    model.prepare()
    print('Reading images and rois')
    roi_list = None
    with open(roi_list_path, 'r') as f:
        roi_list = json.load(f)
    json_path = os.path.join(segmentation_folder, 'data.json')
    results = []
    for index, roi in enumerate(roi_list):
        print(f'seg ({(index + 1)}/{len(roi_list)})')
        print('loading image')
        img_path = join_paths(data_path, roi['path'])
        img = cv2.imread(img_path)
        img = get_cropped_img(img)
        time_dict = {}
        if not roi['empty']:
            print('generating embedding')
            start_time = time.time()
            model.set_embedding(img)
            time_dict['set_embeding'] = time.time() - start_time
            print('obtaining segmentation')
            start_time = time.time()
            seg_chunk, scores = get_all_segmentations(
                model,
                roi['selected_bbox']['x1'],
                roi['selected_bbox']['y1'],
                roi['selected_bbox']['x2'],
                roi['selected_bbox']['y2'])
            time_dict['get_all_segmentations'] = time.time() - start_time
            # SAVE SEGMENTATION
            print('saving segmentation')
            save_segmentation(
                model, json_path, roi['selected_bbox'], segmentation_folder, scores, roi['path'], seg_chunk, results, time_dict)
        else:
            print('Not Roi')
            results.append({
                'img_path': roi['path'],
                'empty': True
            })
            # cv2.imshow('Not ROI',img)
            # cv2.waitKey()
        # cv2.destroyAllWindows()




def save_segmentation(sam_model: SAM_Model,
                      json_path,
                      bbox,
                      segmentation_folder,
                      scores_multimask,
                      img_path,
                      all_segmentations,
                      current_result,
                      time_dict):
    # https://stackoverflow.com/questions/22400652/compress-numpy-arrays-efficiently

    if not os.path.exists(segmentation_folder):
        os.mkdir(segmentation_folder)

    img_name = Path(img_path).name.split('.')[0]

    seg_path = os.path.join(segmentation_folder, (img_name + '_seg.npy.gz'))
    with GzipFile(seg_path, 'w') as f:
        np.save(f, all_segmentations)

    emb_path = os.path.join(segmentation_folder, (img_name + '_emb.torch.gz'))
    with GzipFile(emb_path, 'w') as f:
        sam_model.save_embedding(f)

    # sam_model.save_embedding(emb_path)

    current_result.append({
        'img_path': img_path,
        'seg_path': seg_path,
        'img_emb_path': emb_path,
        'bbox': bbox,
        'empty': False,
        'times': time_dict,
        'seg_info': {
            'multimask_no_logits':
            {
                'midle_point': {
                    'whole': 0,
                    'part': 1,
                    'subpart': 2,
                    'scores': scores_multimask[0]
                },
                'bbox': {
                    'whole': 3,
                    'part': 4,
                    'subpart': 5,
                    'scores': scores_multimask[1]
                },
                'midle_point_bbox': {
                    'whole': 6,
                    'part': 7,
                    'subpart': 8,
                    'scores': scores_multimask[2]
                }
            },
            'multimask_logits':
            {
                'midle_point': {
                    'whole': 9,
                    'part': 10,
                    'subpart': 11,
                    'scores': scores_multimask[3]
                },
                'bbox': {
                    'whole': 12,
                    'part': 13,
                    'subpart': 14,
                    'scores': scores_multimask[4]
                },
                'midle_point_bbox': {
                    'whole': 15,
                    'part': 16,
                    'subpart': 17,
                    'scores': scores_multimask[5]
                }
            },
            'one_mask_no_logits': {
                'midle_point': 18,
                'bbox': 19,
                'midle_point_bbox': 20
            },
            'one_mask_logits': {
                'midle_point': 21,
                'bbox': 22,
                'midle_point_bbox': 23
            }
        }
    })

    with open(json_path, 'w') as f:
        json.dump(current_result, f)


# results_path = 'C:\\Users\\53588\\Desktop\\Tesis\\dev\\SAM_faster_rcnn_performance\\rois.json'
# data_path = "C:\\Users\\53588\\Desktop\\Results\\dfu_segmentation\\datas\\azh_wound_care_center_dataset"
# indexes_path = 'indexes.json'
# bboxes_des = 'final.json'
# sam_checkpoint = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# segmentate_with_simple_sam(data_path, bboxes_des,
#                            'test_results', sam_checkpoint, model_type)
