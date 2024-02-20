from localization.tracker import BoundingBox


def overlapping_area_bboxes(bbox1: BoundingBox, bbox2: BoundingBox):
    '''
    Calcula el área de solapamiento de bbox1 con bbox2
    '''
    y_dif = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h) - max(bbox1.y, bbox2.y)
    x_dif = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w) - max(bbox1.x, bbox2.x)
    if y_dif*x_dif > 0:
        return y_dif*x_dif
    else:
        return 0