from localization.detect_frcnn_dfu import Faster_RCNN
from localization.visualization import visualize_rois
import cv2

config_path = 'config.pickle'
weight_path = 'weights\\model_frcnn.hdf5'
img_path = 'output\\color\\00477.jpg'
img = cv2.imread(img_path)
detector = Faster_RCNN(config_path,weight_path)
detector.prepare()
bboxes = detector.get_roi(img)  
bboxes = [{'x1':bbox[0],'y1':bbox[1],'x2':bbox[2],'y2':bbox[3],'prob':round(float(bbox[4]),2)} for bbox in bboxes]
visualize_rois(img,bboxes,(255,0,0))

