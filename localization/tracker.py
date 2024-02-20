import cv2

# Tracker CSR-DCT
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerMIL_create()


class BoundingBox:
    def __init__(self,x,y,w,h) -> None:
         self.x = x   
         self.y = y   
         self.w = w   
         self.h = h   

    def get_as_array(self):
        return [self.x,self.y,self.w,self.h]
    
    def get_as_dict(self):
        return{
            'x':self.x,
            'y':self.y,
            'w':self.w,
            'h':self.h
        }

    def __repr__(self) -> str:
        return str(self.get_as_dict())

class TrackerCSRT:
    def __init__(self) -> None:
        self.tracker:cv2.TrackerCSRT = cv2.TrackerCSRT_create()
    
    def set_bbox_to_follow(self,color_img, box: BoundingBox):
        self.tracker.init(color_img, box.get_as_array())

    def update(self,color_img):
        _, box = self.tracker.update(color_img)
        return BoundingBox(*[int(v) for v in box])

