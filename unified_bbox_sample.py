from localization.cropping import cropp_the_images_with_stepped_recognition, create_ulcer_rec, rectifie
import json
from localization.tracker import TrackerCSRT
from localization.visualization import show_bboxes, add_bboxes_tracking_with_rec
import os

def loc_with_standar_configuration(save_path):
    config = {
        'path_dataset':'output',
        'frame_step': 15,
        'start':16,
        'end': 34
    }

    tracker = TrackerCSRT()
    ulcer_rec = create_ulcer_rec(config_filename='config.pickle',
                                weight_path= 'weights\\model_frcnn.hdf5')
    bbox_data = cropp_the_images_with_stepped_recognition(config,tracker,ulcer_rec,steps_for_rec = 3, rectifie_func = rectifie)


    with open(save_path, 'w') as f:
        json.dump(bbox_data, f)
    return bbox_data


result_path = 'output'
save_path = os.path.join(result_path,'bboxes.json')


bbox_data = []
# bbox_data = loc_with_standar_configuration(save_path)

with open(save_path,'r') as f:
    bbox_data = json.load(f)


show_bboxes(bbox_data,add_bboxes_tracking_with_rec)