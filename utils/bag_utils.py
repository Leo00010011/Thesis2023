'''
Methods to use bag files:
- Review IQA
- Render Depth
- Compress Bags
- Playback
- Record
'''
import threading
# import open3d as o3d
import pyrealsense2 as rs
import numpy as np
from rosbag.bag import Bag
from rosbag.rosbag_main import ProgressMeter
from rosbag import Compression
import cv2
import os
from itertools import islice
import open3d as o3d

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32


class CompressionState:
    def __init__(self) -> None:
        self.full_size = None
        self.current_size = None
        self.end = None
        self.stop = threading.Event()

    def reset(self):
        self.full_size = None
        self.current_size = None
        self.end = None
        self.stop.clear()

compression_state = CompressionState()

def show_img(img):
    cv2.imshow('cosa', img)
    cv2.waitKey()


class DepthColorizer:
    def __init__(self, arr) -> None:
        mins = []
        for i in range(arr.shape[0]):
            current_img = arr[i, :, :]
            mins.append(current_img[current_img > 0].min())
        self.min = int(min(mins))
        maxs = [np.percentile(arr[index, :, :], 98.5)
                for index in range(arr.shape[0])]
        self.max = int(min(maxs))
        self.update_lut(self.max,self.min)


    def update_lut(self, max_depth,min_depth):
        self.max = max_depth
        self.min = min_depth
        color1 = (0, 0, 255)
        color2 = (0, 127, 255)
        color3 = (0, 255, 255)
        color4 = (127, 255, 127)
        color5 = (255, 255, 0)
        color6 = (255, 127, 0)
        color7 = (255, 0, 0)
        color8 = (128, 64, 64)
        colorArr = np.array(
            [[color1, color2, color3, color4, color5, color6, color7, color8]], dtype=np.uint8)
        self.lut2 = cv2.resize(
            colorArr, (self.max - self.min + 2, 1), interpolation=cv2.INTER_LINEAR)


    def colorize(self, depth):
        depth = np.copy(depth)
        depth[depth > self.max] = self.max
        depth = depth - (self.min - 1)
        depth[depth > self.max] = 0
        lut = self.lut2[0, :, :]
        lut[0, :] = [0, 0, 0]
        result = lut[depth]
        return result

    def compare_coloricer(depth, coloricer_list):
        color_depth = [func(depth) for func in coloricer_list]
        return np.concatenate(color_depth, 1)


class ScoreToColorConv:
    def __init__(self, min, max):
        color1 = (0, 0, 255)
        color2 = (0, 165, 255)
        color3 = (0, 255, 255)
        color4 = (0, 255, 165)
        color5 = (0, 255, 0)
        colorArr = np.array(
            [[color1, color2, color3, color4, color5]], dtype=np.uint8)
        self.min = min
        self.max = max
        # resize lut to 256 (or more) values
        self.lut = cv2.resize(colorArr, (256, 1),
                              interpolation=cv2.INTER_LINEAR)

    def get_color(self, score):
        scl_score = int(((score - self.min)/(self.max - self.min)) * 255)

        return int(self.lut[0, scl_score, 0]), int(self.lut[0, scl_score, 1]), int(self.lut[0, scl_score, 2])


def put_scale_bar(img, org, levels, height, cnv):
    h, w, _ = img.shape
    bar_scale = int((1/256)*w)
    r_s, c_s = org
    for i in range(levels):
        score = cnv.max - i/(levels-1)*(cnv.max - cnv.min)
        color = cnv.get_color(score)
        img[r_s + i*height:r_s + (i + 1)*height -
            bar_scale, c_s:c_s + bar_scale, :] = color
        cv2.putText(img, '%.2f' % score, (w - bar_scale*20, r_s +
                    int((i + 1/2)*height)), 0, .55, color, thickness=1,)


def put_score_bar(scr_list, img, cnv: ScoreToColorConv, index, org=None, scale=None, height=None):
    h, w, _ = img.shape
    bar_scale = int((1/256)*w)
    count = scr_list.shape[0]
    if not scale:
        scale = w//count
    scale = min(int(w/count), scale)
    if not height:
        height = bar_scale

    if not org:
        org = (h - 4*height, (w - count*scale)//2)

    # putting scale bar
    number_of_scales = 6
    r_s = int(h/8)
    c_s = w - bar_scale*4
    height_s = int((h/4)/number_of_scales)
    put_scale_bar(img, (r_s, c_s), number_of_scales, height_s, cnv)

    r, c = org
    # putting index
    arrow_col = index*scale + int(scale/2)
    cv2.arrowedLine(img, (c + arrow_col, r - height),
                    (c + arrow_col, r), (0, 255, 0))

    # putting score bar
    for i in range(count):
        if not cnv:
            img[r:r + height, c + i*scale:c + (i + 1)*scale, :] = scr_list[i]
        else:
            img[r:r + height, c + i*scale:c +
                (i + 1)*scale, :] = cnv.get_color(scr_list[i])


class BagReview:
    def __init__(self, path) -> None:
        self.path = path

    def _get_image_from_msg(msg):
        nparr = np.frombuffer(msg.data, dtype=np.uint8)
        nparr = nparr.reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR)

    def _get_depth_from_msg(msg):
        nparr = np.frombuffer(msg.data, dtype=np.uint16)
        nparr = nparr.reshape(msg.height, msg.width)
        return nparr

    def is_compressed(self):
        b = Bag(self.path)
        result = b._has_compressed_chunks
        b.close()
        return result


    def change_compression(self,output_path,compression):
        inbag = Bag(self.path)
        outbag = Bag(output_path, 'w')
        outbag.compression = compression
        compression_state.current_size = 0
        compression_state.full_size = inbag._uncompressed_size
        compression_state.end = False
        # meter = ProgressMeter(outbag.filename, inbag._uncompressed_size)
        total_bytes = 0
        for topic, msg, t, conn_header in inbag.read_messages(raw=True, return_connection_header=True):
            if compression_state.stop.is_set():
                inbag.close()
                outbag.close()
                # meter.finish()
                compression_state.end = True
                os.remove(output_path)
                return
            msg_type, serialized_bytes, md5sum, pos, pytype = msg
            outbag.write(topic, msg, t, raw=True,
                         connection_header=conn_header)

            total_bytes += len(serialized_bytes)
            compression_state.current_size = total_bytes
            # meter.step(total_bytes)
        inbag.close()
        outbag.close()
        # meter.finish()
        compression_state.end = True

    def compress_bag(self, output_path):
        self.change_compression(output_path,Compression.BZ2)

    def decompress_bag(self,output_path):
        self.change_compression(output_path,Compression.NONE)

    def get_all_depth_frames(self):
        bag = Bag(self.path)
        key = "/device_0/sensor_0/Depth_0/image/data"

        count = bag.get_type_and_topic_info().topics[key].message_count
        first = True
        result = None
        for index, msg in enumerate(bag.read_messages(topics=[key])):
            msg = msg.message
            if first:
                first = False
                result = np.empty(
                    shape=(count, msg.height, msg.width), dtype=np.uint16)
            result[index, :, :] = BagReview._get_depth_from_msg(msg)
        bag.close()
        return result

    def get_all_frames(self):
        bag = Bag(self.path)
        key = None
        try:
            bag.get_type_and_topic_info(
            ).topics['/device_0/sensor_0/Color_0/image/data']
            key = '/device_0/sensor_0/Color_0/image/data'
        except:
            key = '/device_0/sensor_1/Color_0/image/data'

        count = bag.get_type_and_topic_info().topics[key].message_count
        first = True
        result = None
        for index, msg in enumerate(bag.read_messages(topics=[key])):
            msg = msg.message
            if first:
                first = False
                result = np.empty(
                    shape=(count, msg.height, msg.width, 3), dtype=np.uint8)
            result[index, :, :, :] = BagReview._get_image_from_msg(msg)
        bag.close()
        return result
    
    def get_one_frame(self,index = 0):
        bag = Bag(self.path)
        key = None
        try:
            bag.get_type_and_topic_info(
            ).topics['/device_0/sensor_0/Color_0/image/data']
            key = '/device_0/sensor_0/Color_0/image/data'
        except:
            key = '/device_0/sensor_1/Color_0/image/data'
        
        messages = islice(bag.read_messages(topics=[key]),index,index + 1) 
        for msg in messages:
            result =  BagReview._get_image_from_msg(msg.message)
            bag.close()
            return result
    

    def _show_data(self, arr, img_extractor,img_inter = None):
        index = 0
        count = arr.shape[0]
        cv2.namedWindow(self.path)
        while True:
            img = img_extractor(arr, index)
            cv2.imshow(self.path, img)
            code = cv2.waitKeyEx()
            if code == LEFT_ARROW:
                index = max(index - 1, 0)
            elif code == RIGHT_ARROW:
                index = min(count - 1, index + 1)
            elif code == SCAPE:
                break
            elif code == SPACE_BAR:
                if img_inter:
                    img_inter(img,index,self.path)
            else:
                print(code)
        cv2.destroyAllWindows()


    def _image_extractor(arr, index):
        return arr[index, :, :, :]

    def IQA_extractor_decorator(extractor, arr, IQA):
        scores = np.array([IQA(extractor(arr, index))
                          for index in range(arr.shape[0])])
        scores = (scores - scores.mean())/scores.std()
        cnv = ScoreToColorConv(scores.min(), scores.max())

        def mod_extractor(arr, index):
            img = extractor(arr, index)
            color = cnv.get_color(scores[index])
            put_score_bar(scores, img, cnv, index)
            cv2.putText(img, 'Score:%.2f' %
                        scores[index], (0, 30), 0, .75, color, thickness=2)
            return img
        return mod_extractor

    def review_color_frames(self,img_inter = None):
        arr = self.get_all_frames()
        self._show_data(arr, BagReview._image_extractor,img_inter)
        return arr


    def review_frames_IQA(self, IQA):
        arr = self.get_all_frames()
        self._show_data(arr, BagReview.IQA_extractor_decorator(
            BagReview._image_extractor, arr, IQA))
    
    def DMQA_extractor_decorator(arr,DMQA):
        print('Computing scores')
        scores = np.array([DMQA(arr[index,:,:])
                          for index in range(arr.shape[0])])
        scores = (scores - scores.mean())/scores.std()
        print('Initialazing  Colorizer')
        colorizer = DepthColorizer(arr)
        cnv = ScoreToColorConv(scores.min(), scores.max())
        def mod_extractor(arr,index):
            img = colorizer.colorize(arr[index,:])
            color = cnv.get_color(scores[index])
            put_score_bar(scores, img, cnv, index)
            cv2.putText(img, 'Score:%.2f' %
                        scores[index], (0, 30), 0, .75, (255,255,255), thickness=3)
            cv2.putText(img, 'Score:%.2f' %
                        scores[index], (0, 30), 0, .75, color, thickness=2)
            return img
        return mod_extractor
    

    def DMQA_extractor_decorator_with_color(arr_dmap,arr_color,DMQA_color,color_extractor):
        print('Computing scores')
        scores = np.array([DMQA_color(arr_dmap[index,:,:],color_extractor(arr_color,index))
                          for index in range(min(arr_dmap.shape[0],arr_color.shape[0]))])
        scores = (scores - scores.mean())/scores.std()
        print('Initialazing  Colorizer')
        colorizer = DepthColorizer(arr_dmap)
        cnv = ScoreToColorConv(scores.min(), scores.max())
        def mod_extractor(arr,index):
            img = colorizer.colorize(arr[index,:])
            color = cnv.get_color(scores[index])
            put_score_bar(scores, img, cnv, index)
            cv2.putText(img, 'Score:%.2f' %
                        scores[index], (0, 30), 0, .75, (255,255,255), thickness=3)
            cv2.putText(img, 'Score:%.2f' %
                        scores[index], (0, 30), 0, .75, color, thickness=2)
            return img
        return mod_extractor
        
    def review_frames_DMQA_color(self,DMQA):
        print('Reading the bag')
        arr_dmap = self.get_all_depth_frames()
        arr_color = self.get_all_frames()
        self._show_data(arr_dmap,BagReview.DMQA_extractor_decorator_with_color(arr_dmap,arr_color,DMQA,BagReview._image_extractor))

    def review_frames_DMQA(self,DMQA):
        print('Reading the bag')
        arr = self.get_all_depth_frames()
        self._show_data(arr, BagReview.DMQA_extractor_decorator(arr,DMQA))

    def create_depth_extractor(arr):
        color = DepthColorizer(arr)

        def depth_extractor(arr, index):
            return color.colorize(arr[index, :, :])
        return depth_extractor

    def review_depth_frames(self):
        arr = self.get_all_depth_frames()
        self._show_data(arr, BagReview.create_depth_extractor(arr))
    
    def save_all_frames(self, output_path):
        rgbd_video = o3d.t.io.RGBDVideoReader.create(self.path)
        rgbd_video.save_frames(output_path)

    


# class RSReader:

#     def start_camera(self):
#         if not o3d.t.io.RealSenseSensor.list_devices():
#             raise Exception("Camera unavailable")
#         cfg = o3d.t.io.RealSenseSensorConfig(
#             {'serial': '', 'color_format': 'RS2_FORMAT_RGB8', 'color_resolution': '1280,720', 'depth_resolution': '1280,720','color_fps':'30','depth_fps':'30'})
#             # {'serial': '', 'color_format': 'RS2_FORMAT_RGB8', 'color_resolution': '1280,720', 'depth_resolution': '1280,720'})

#         self.rs = o3d.t.io.RealSenseSensor()
#         self.rs.init_sensor(cfg, 0, 'C:\\Users\\53588\\Desktop\\Tesis\\dev\\DFU-Measure\\tools')
#         self.rs.start_capture(False)

#     def get_frames(self):
#         while True:
#             im_rgbd = self.rs.capture_frame(True, True)
#             color = np.array(im_rgbd.color)
#             depth = np.array(im_rgbd.depth)
#             yield color, depth

#     def get_color_frames(self):
#         while True:
#             im_rgbd = self.rs.capture_frame(True, True)
#             color = np.array(im_rgbd.color)
#             yield color


#     def stop_camera(self):
#         self.rs.stop_capture()

#     def save_intrinsic(self):
#         json_obj = str(self.rs.get_metadata())
#         with open('output/reconstruction/intrinsic.json', 'w') as intrinsic_file:
#             intrinsic_file.write(json_obj)

#     def get_resolution(self):
#         return (self.rs.get_metadata().height,
#                 self.rs.get_metadata().width)


class RSPlayback:
    '''
    To get frames as nparray first call start_camera and the use get_frames
    '''

    def __init__(self, path) -> None:
        self.path = path

    def start_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(self.path)
        self.profile = self.pipeline.start(self.config)
        self.stream_color = self.profile.get_stream(rs.stream.color)
        self.stream_depth = self.profile.get_stream(rs.stream.depth)
        self.device = self.profile.get_device()
        self.playback = self.device.as_playback()

    def get_frames(self):

        align = rs.align(rs.stream.color)
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            yield color_frame, depth_frame


    def stop_camera(self):
        self.pipeline.stop()

    def get_resolution(self):
        return (self.stream_color.as_video_stream_profile().intrinsics.height,
                self.stream_color.as_video_stream_profile().intrinsics.width)

    # def save_intrinsic(self):
    #     bag_reader = o3d.t.io.RSBagReader()
    #     bag_reader.open(self.path)
    #     json_obj = str(bag_reader.metadata)
    #     with open('output/reconstruction/intrinsic.json', 'w') as intrinsic_file:
    #         intrinsic_file.write(json_obj)
    #     bag_reader.close()


class RSRecord:
    '''
    To get frames as nparray first call start_camera and the use get_frames
    '''

    def __init__(self, path) -> None:
        self.path = path

    def start_camera(self):
        self.pipeline = rs.pipeline()
        self.config:rs.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280,720, rs.format.rgb8, 30)
        self.config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        self.config.enable_record_to_file(self.path)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_frames(self):

        align = rs.align(rs.stream.color)
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            yield color_frame, depth_frame

    def get_color_frames(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            yield color_frame

    def stop_camera(self):
        self.pipeline.stop()

    