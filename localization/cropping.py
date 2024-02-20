import open3d as o3d
import time
import cv2
import os
from reconstruction.utility.file import get_rgbd_file_list2
import numpy as np
from localization.detect_frcnn_dfu import Faster_RCNN
from localization.tracker import TrackerCSRT, BoundingBox
import json
from localization.similarity_measures import overlapping_area_bboxes
from localization.visualization import put_rectangle
from localization.unified_bbox import get_unified_bbox
from localization.desambiguate import choose_roi_rois

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32


def always_rec(rec_bboxes, track_bbox):
    '''
    Una forma trivial de rectificar la localización que es siempre seleccionando la primera de las detecciones
    '''
    return rec_bboxes[0]


def create_ulcer_rec(config_filename,weight_path):
    '''
    Crea un delegado que dada una imagen retorna las detecciones de UPD con el formato
    (BoundingBox,probabilidad)
    '''
    detector = Faster_RCNN(config_filename, weight_path)
    detector.prepare()

    def ulcer_rec(img):
        boxes = detector.get_roi(img)
        fixed_boxes = []
        for box in boxes:
            [x1, y1, x2, y2, prob] = box
            fixed_boxes.append(
                (BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), float(prob)))
        return fixed_boxes
    return ulcer_rec

def rectifie(img, rec_bboxes, track_bbox=None):
    '''
    Método que retorna el bbox de reconocimiento que tiene más intercepción con el tracking, y en caso de que no haya seguimiento
    permite seleccionarlo manualmente
    '''
    final = None
    if len(rec_bboxes) > 1:
        # Si hay más de una ulcera reconocida toca desambiguar
        if track_bbox is None:
            img_copy = img.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            final = choose_roi_rois(img_copy, rec_bboxes)
        else:
            # Calcular el solapamiento del bbox trackeado con cada uno reconocido
            overlapp = [overlapping_area_bboxes(
                rec_bbox, track_bbox) for rec_bbox, _ in rec_bboxes]
            # Obtener el máximo
            max_index = 0
            max_value = overlapp[0]
            for index, area in enumerate(overlapp):
                if area > max_value:
                    max_value = area
                    max_index = index
            final, _ = rec_bboxes[max_index]
    elif len(rec_bboxes) == 1:
        final, _ = rec_bboxes[0]
    else:
        final = None
    return final


def cropp_the_images_with_stepped_recognition(config, tracker: TrackerCSRT, ulcer_rec, steps_for_rec=20, rectifie_func=always_rec):
    '''
    Método para realizar el seguimiento de la UPD, rectificándolo con reconocimiento cada `steps_for_rec` pasos
    '''
    print('Loading images')
    [color_files, depth_files] = get_rgbd_file_list2(
        config["path_dataset"], False, config['frame_step'])
    
    color_files = color_files[config['start']:config['end'] + 1]
    depth_files = depth_files[config['start']:config['end'] + 1]

    rec_bboxes = None
    current_bbox = None
    track_bbox = None
    bbox_data = {
        'steps_for_rec': steps_for_rec,
        'bbox_per_frame': [],
        'time_per_tracking': [],
        'time_per_rec': [],
    }
    total = len(color_files)
    start_total_time = time.time()
    for index, (color_file, depth_file) in enumerate(zip(color_files, depth_files)):
        print(f'cropping: {index + 1}/{total}')
        # read the images
        current_img_data = {
            'color_file': color_file,
            'depth_file': depth_file
        }
        color = np.asarray(o3d.io.read_image(color_file))
        # cropp the images
        if index == 0:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            track_bbox = cv2.selectROI(
                    'Select the ulcer', color, fromCenter=False, showCrosshair=False)
            current_img_data['rect_bbox'] = {
                'x':track_bbox[0],
                'y':track_bbox[1],
                'w':track_bbox[2],
                'h':track_bbox[3]
                }
            cv2.destroyAllWindows()
            tracker.set_bbox_to_follow(color, BoundingBox(*track_bbox))
            bbox_data['bbox_per_frame'].append(current_img_data)
            continue
        if index != 0 and index % steps_for_rec == 0:
            start = time.time()  # time for rec
            rec_bboxes = ulcer_rec(color)
            bbox_data['time_per_rec'].append(time.time() - start)
            serializable_rec_bboxes = [(bbox.get_as_dict(), prob)
                                       for bbox, prob in rec_bboxes]
            current_img_data['rec_bbox'] = serializable_rec_bboxes
            start = time.time()
            track_bbox: BoundingBox = tracker.update(color)
            bbox_data['time_per_tracking'].append(time.time() - start)
            current_img_data['track_bbox'] = track_bbox.get_as_dict()
            if len(rec_bboxes) == 0:
                current_bbox = track_bbox
                current_img_data['rect_bbox'] = 'Empty'
            else:
                current_bbox = rectifie_func(color, rec_bboxes, track_bbox)
                current_img_data['rect_bbox'] = current_bbox.get_as_dict()
            tracker.set_bbox_to_follow(color, current_bbox)
        else:
            start = time.time()
            current_bbox = tracker.update(color)
            bbox_data['time_per_tracking'].append(time.time() - start)
            current_img_data['track_bbox'] = current_bbox.get_as_dict()
        # save the images
        bbox_data['bbox_per_frame'].append(current_img_data)
    bbox_data['total_time'] = time.time() - start_total_time


    return bbox_data


def cropp_the_images_with_initial_bbox_and_tracking(config, tracker: TrackerCSRT):
    '''
    Método para realizar el seguimiento de la UPD solo con tracking
    '''
    [color_files, depth_files] = get_rgbd_file_list2(
        config["path_dataset"], False, config['frame_step'])
    color_files = color_files[config['start']:config['end'] + 1]
    depth_files = depth_files[config['start']:config['end'] + 1]
    track_bbox: BoundingBox = None
    bboxes_data = {'bboxes': []}
    all_time_start = None
    for color_file, depth_file in zip(color_files, depth_files):
        color = np.asarray(o3d.io.read_image(color_file))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        step_time = time.time()
        if not track_bbox:
            while not track_bbox:
                track_bbox = cv2.selectROI(
                    'Select the ulcer', color, fromCenter=False, showCrosshair=False)
            cv2.destroyAllWindows()
            tracker.set_bbox_to_follow(color,BoundingBox(*track_bbox))
            track_bbox = BoundingBox(*track_bbox).get_as_dict()
            # Empezar a contar el tiempo después de seleccionar el ROI
            all_time_start = time.time()
        else:
            track_bbox = tracker.update(color)
            track_bbox = track_bbox.get_as_dict()

        bboxes_data['bboxes'].append({
            'color_path': color_file,
            'depth_path': depth_file,
            'bbox': track_bbox,
            'time': time.time() - step_time
        })

    bboxes_data['all_time'] = time.time() - all_time_start
    return bboxes_data
