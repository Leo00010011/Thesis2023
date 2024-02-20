from localization.cropping import cropp_the_images_with_initial_bbox_and_tracking
from localization.tracker import TrackerCSRT
from localization.visualization import show_tracking, add_bboxes_simple_tracking

config = {
    'path_dataset':'output',
    'frame_step': 15,
    'start':16,
    'end': 34
}

tracker = TrackerCSRT()
bboxes_data = []
bboxes_data = cropp_the_images_with_initial_bbox_and_tracking(config,tracker)

show_tracking(bboxes_data['bboxes'],add_bboxes_simple_tracking)