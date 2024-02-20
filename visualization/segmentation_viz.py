import sys
sys.path.append(
    'C:\\Users\\53588\\Desktop\\Tesis\\dev\\SAM_faster_rcnn_performance')
import cv2
import json
from gzip import GzipFile
from segmentation.sam_segmentation import apply_the_mask
from processing.reading import get_cropped_img
from detection.detect_frcnn_dfu import join_paths
import numpy as np

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32
D = 100
MASK_COLOR = np.array([255, 144, 30], dtype=np.uint8)


def put_rectangle_simple(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

def put_midle_point(img, x1, y1, x2, y2):
    center = (x1 + (x2 - x1)//2,y1 + (y2 - y1)//2)
    cv2.circle(img, center, 3, (0,255,255), -1)
    


def put_name(img, name):
    (retval, _) = cv2.getTextSize(
        name, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
    w, h = retval
    img_w, img_h = img.shape[1], img.shape[0]
    textOrg = ((img_w - w - 1), (img_h - h - 1))
    cv2.putText(img, name, textOrg, cv2.FONT_HERSHEY_DUPLEX,
                0.3, (255, 255, 255), 1)


def get_names():
    names = []
    plot_hint = []
    names.append('multimask--midle--whole')
    plot_hint.append((True,False))
    names.append('multimask--midle--part')
    plot_hint.append((True,False))
    names.append('multimask--midle--subpart')
    plot_hint.append((True,False))
    names.append('multimask--bbox--whole')
    plot_hint.append((False,True))
    names.append('multimask--bbox--part')
    plot_hint.append((False,True))
    names.append('multimask--bbox--subpart')
    plot_hint.append((False,True))
    names.append('multimask--midle_bbox--whole')
    plot_hint.append((True,True))
    names.append('multimask--midle_bbox--part')
    plot_hint.append((True,True))
    names.append('multimask--midle_bbox--subpart')
    plot_hint.append((True,True))
    names.append('multimask_logits--midle--whole')
    plot_hint.append((True,False))
    names.append('multimask_logits--midle--part')
    plot_hint.append((True,False))
    names.append('multimask_logits--midle--subpart')
    plot_hint.append((True,False))
    names.append('multimask_logits--bbox--whole')
    plot_hint.append((False,True))
    names.append('multimask_logits--bbox--part')
    plot_hint.append((False,True))
    names.append('multimask_logits--bbox--subpart')
    plot_hint.append((False,True))
    names.append('multimask_logits--midle_bbox--whole')
    plot_hint.append((True,True))
    names.append('multimask_logits--midle_bbox--part')
    plot_hint.append((True,True))
    names.append('multimask_logits--midle_bbox--subpart')
    plot_hint.append((True,True))
    names.append('unique--midle')
    plot_hint.append((True,False))
    names.append('unique--bbox')
    plot_hint.append((False,True))
    names.append('unique--midle_bbox')
    plot_hint.append((True,True))
    names.append('unique_logits--midle')
    plot_hint.append((True,False))
    names.append('unique_logits--bbox')
    plot_hint.append((False,True))
    names.append('unique_logits--midle_bbox')
    plot_hint.append((True,True))
    return names, plot_hint


def fill_mask(mask, shape):
    mask_h, mask_w = mask.shape
    h, w = shape
    result = np.zeros((h, w))
    result[:mask_h, :mask_w] = mask
    result = result.astype(np.dtype('uint8'))
    return result


def review_masks(data_path, json_path):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)
    current_index = 0
    while True:
        # mostrar imagen
        ori_img = cv2.imread(join_paths(
            data_path, data[current_index]["img_path"]))
        is_empty = data[current_index ]['empty']
        if is_empty:
            put_name(ori_img,'EMPTY')
        cv2.imshow('Segmentations', ori_img)
        key = cv2.waitKeyEx()
        # si espacio mostrar las mascaras
        if key == SPACE_BAR:
            if is_empty:
                continue
            all_masks = None
            with GzipFile(data[current_index]['seg_path'], 'r') as f:
                all_masks = np.load(f)
            names, plot_hint = get_names()
            inner_index = 0
            only_mask = None
            view_only_mask = False
            current_image = None
            while True:
                if view_only_mask:
                    current_image = only_mask
                else:
                    mask = all_masks[inner_index, :, :]
                    mask = fill_mask(mask, ori_img.shape[:2])
                    img_copy = ori_img.copy()
                    img_masked, only_mask = apply_the_mask(img_copy, mask, MASK_COLOR)
                    current_image = img_masked
                put_name(current_image, names[inner_index])
                has_point, has_bbox = plot_hint[inner_index]
                if has_bbox:
                    put_rectangle_simple(
                        current_image,
                        data[current_index]['bbox']['x1'],
                        data[current_index]['bbox']['y1'],
                        data[current_index]['bbox']['x2'],
                        data[current_index]['bbox']['y2']
                    )
                if has_point:
                    put_midle_point(
                        current_image,
                        data[current_index]['bbox']['x1'],
                        data[current_index]['bbox']['y1'],
                        data[current_index]['bbox']['x2'],
                        data[current_index]['bbox']['y2']
                    )
                cv2.imshow('Segmentations', current_image)
                key = cv2.waitKeyEx()
            #   derecha e izda para moverse por los indeces
                if key == LEFT_ARROW:
                    view_only_mask = False
                    inner_index = max(0, inner_index - 1)
                    continue
                elif key == RIGHT_ARROW:
                    view_only_mask = False
                    inner_index = min(inner_index + 1, all_masks.shape[0] - 1)
                    continue
            #   espacio para salir
                elif key == SPACE_BAR:
                    view_only_mask = False
                    break
                elif key == D:
                    view_only_mask = not view_only_mask
                    continue
        # si derecha avanzar un indece y continuar
        elif key == LEFT_ARROW:
            current_index = max(current_index - 1, 0)
            continue
        # si izda retroceder un indece y continuar
        elif key == RIGHT_ARROW:
            current_index = min(current_index + 1, len(data) - 1)
            continue
        elif key == SCAPE:
            break
        else:
            print(key)



results_path = 'C:\\Users\\53588\\Desktop\\Tesis\\dev\\SAM_faster_rcnn_performance\\rois.json'
data_path = "C:\\Users\\53588\\Desktop\\Results\\dfu_segmentation\\datas\\azh_wound_care_center_dataset"
indexes_path = 'indexes.json'
bboxes_des = 'final.json'
sam_checkpoint = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_b_01ec64.pth"
model_type = "vit_b"

review_masks(data_path, 'test_results\\data.json')
