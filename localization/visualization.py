import cv2
from localization.tracker import BoundingBox
from localization.unified_bbox import get_unified_bbox
from reconstruction.utility.file import get_rgbd_file_list2
LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32
A = 97
D = 100


def put_rectangle(img, bbox:BoundingBox, textLabel, color=(255, 0, 0)):
    '''
    Plotea un bounding box en la imagen
    '''
    x1, y1 = bbox.x, bbox.y
    x2, y2 = x1 + bbox.w, y1 + bbox.h
    cv2.rectangle(img, (x1, y1), (x2, y2), color)
    (retval, baseLine) = cv2.getTextSize(
        textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
    textOrg = (x1, y1-0)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5),
                  (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5),
                  (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
    cv2.putText(img, textLabel, textOrg,
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return img


def add_bboxes_simple_tracking(index, img, data):
    img = put_rectangle(img, BoundingBox(**data[index]['bbox']), 'ulcer')


unified_bboxes = None

def add_bboxes_tracking_with_rec(index, img, data):
    '''
    Dada la data que guarda `cropp_the_images_with_stepped_recognition`, una imagen y el índice correspondiente a esa imagen en la data
    se plotean el unfied bbox, el tracking bbox y el rec bbox

    El método está pensado para ser utilizado para revisar la secuencia entera, por lo que se guarda en caché los unified_bboxes
    '''
    steps_for_rec = data['steps_for_rec']
    global unified_bboxes
    if unified_bboxes is None:
        print('Computing unified bboxes')
        unified_bboxes = get_unified_bbox(data)
    put_rectangle(img, unified_bboxes[index], 'Unified', (255, 255, 0))

    if index == 0:
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['rect_bbox']), 'rect_bbox', (0, 0, 255))
    elif index % steps_for_rec == 0 and  data['bbox_per_frame'][index]['rect_bbox'] != 'Empty':
        for i, [bbox, _] in enumerate(data['bbox_per_frame'][index]['rec_bbox']):
            put_rectangle(img, BoundingBox(**bbox),
                          f'rec_bbox_{i + 1}', (0, 0, 255))
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['rect_bbox']), 'rect_bbox', (0, 255, 0))
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['track_bbox']), 'track_bbox')
    else:
        put_rectangle(img, BoundingBox(**
                                       data['bbox_per_frame'][index]['track_bbox']), 'track_bbox')

def select_borders(config):
    
    [color_files, _] = get_rgbd_file_list2(
        config["path_dataset"], False, config['frame_step'])
    index = 0
    count = 0
    result = [-1,-1]
    while True:
        color = cv2.imread(color_files[index])
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(color_files):
                index = len(color_files) - 1
        elif code == SPACE_BAR:
            # cv2.imwrite(
            #     f'results with only tracker\\sample_{count}.jpg', color)
            # count += 1
            continue
        elif code == A:
            result[0] = index
            continue
        elif code == D:
            result[1] = index
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()
    return result


def show_tracking(bboxes,add_bboxes_func):
    index = 0
    count = 0
    result = 'None'
    while True:
        color = cv2.imread(bboxes[index]['color_path'])
        add_bboxes_func(index, color, bboxes)
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(bboxes):
                index = len(bboxes) - 1
        elif code == SPACE_BAR:
            # cv2.imwrite(
            #     f'results with only tracker\\sample_{count}.jpg', color)
            # count += 1
            continue
        elif code == A:
            result = 'good'
            continue
        elif code == D:
            result = 'wrong'
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()
    return result

def show_bboxes(bboxes, add_bboxes_func):
    index = 0
    count = 0
    result = 'None'
    while True:
        color = cv2.imread(bboxes['bbox_per_frame'][index]['color_file'])
        add_bboxes_func(index, color, bboxes)
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(bboxes['bbox_per_frame']):
                index = len(bboxes['bbox_per_frame']) - 1
        elif code == SPACE_BAR:
            # cv2.imwrite(
            #     f'results with only tracker\\sample_{count}.jpg', color)
            # count += 1
            continue
        elif code == A:
            result = 'good'
            continue
        elif code == D:
            result = 'wrong'
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()
    return result

def crop_image(img, bbox: BoundingBox):
    return img[bbox.y: bbox.y + bbox.h, bbox.x: bbox.x + bbox.w]


def show_bboxes_cropped(bboxes):
    index = 0
    count = 0
    unified_bboxes = get_unified_bbox(bboxes)
    while True:
        color = cv2.imread(bboxes['bbox_per_frame'][index]['color_file'])
        color = crop_image(color,unified_bboxes[index])
        cv2.imshow('bbox', color)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index -= 1
            if index < 0:
                index = 0
        elif code == RIGHT_ARROW:
            index += 1
            if index == len(bboxes['bbox_per_frame']):
                index = len(bboxes['bbox_per_frame']) - 1
        elif code == SPACE_BAR:
            cv2.imwrite(
                f'results with only tracker\\sample_{count}.jpg', color)
            count += 1
            continue
        elif code == SCAPE:
            break
        else:
            print(code)
    cv2.destroyAllWindows()

def visualize_rois(img, bboxes, color):
    '''
    Plotea los bboxes en el formato {'x1':#,'y1':#,'x2':#,'y2':#,'prob':#}en la imagen especificada en color
    '''
    for bbox in bboxes:
        x1, y1, x2, y2, prob = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'], bbox['prob']
        put_rectangle(img,BoundingBox(x1,y1,x2- x1, y2 - y1),f'p:{prob}',color)
    cv2.imshow('Rois', img)
    cv2.waitKey()


        
