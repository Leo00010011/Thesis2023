import json
import os
import cv2
from localization.visualization import put_rectangle
import numpy as np
import sys
from localization.detect_frcnn_dfu import join_paths
sys.path.append('..')
A = 97
D = 100

def get_bbox_prob(bbox_data):
    bbox = {
        'x1': bbox_data['x1'],
        'y1': bbox_data['y1'],
        'x2': bbox_data['x2'],
        'y2': bbox_data['y2']
    }
    prob = bbox_data['prob']
    return bbox, prob
    

def choose_roi_rois(img: np.ndarray, bboxes):
    '''
    Método para seleccionar manualmente un bbox de un conjunto de bboxes retornando el índice seleccionado
    '''
    # putting bboxes
    img_rois = img.copy()
    cv2.imshow('Img', img_rois)
    cv2.waitKey()
    cv2.destroyAllWindows()

    for index, bbox_data in enumerate(bboxes):
        bbox, _ = get_bbox_prob(bbox_data)
        put_rectangle(img_rois, bbox, str(index))
    # selecting bboxes
    selected_bbox = None
    index = None
    while True:
        cv2.imshow('Img', img_rois)
        while True:
            index = cv2.waitKey() - 48
            if 0 <= index < len(bboxes):
                selected_bbox, _ = get_bbox_prob(bboxes[index])
                break
        # showing selected bbox
        cv2.destroyAllWindows()
        img_sel = img.copy()
        put_rectangle(img_sel, selected_bbox, str(index))
        cv2.imshow('Img', img_sel)
        key = cv2.waitKey()
        if key == A:
            cv2.destroyAllWindows()
            break
        else:
            continue

    return index


def desambiguate_all(bboxes_path, data_path, desambiguate_path):
    '''
    Método que lee el json con una lista de bboxes de detección y retorna una lista de índices seleccionados, con -1 en caso de que no se haya detectado nada
    ,0 en caso de que se haya detectado una sola y y permite seleccionar el índice en caso de que haya más de 1 bbox seleccionado.
    '''
    data = None
    with open(bboxes_path, 'r') as f:
        data = json.load(f)

    indexes = []

    for i in range(len(data)):
        if data[i]['empty']:
            indexes.append(-1)
        else:
            if len(data[i]['bboxes']) > 1:
                img_path = join_paths(data_path, data[i]['path'])
                img = cv2.imread(img_path)
                indexes.append(choose_roi_rois(img, data[i]['bboxes']))
            else:
                indexes.append(0)
        with open(desambiguate_path, 'w') as f:
            json.dump(indexes, f)



