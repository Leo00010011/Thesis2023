import torch
import cv2
from pathlib import Path
import os
import json
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32

# Model types:
TYPE_VIT_H = 'vit_h'
TYPE_VIT_L = 'vit_l'
TYPE_VIT_B = 'vit_b'
# Model paths:
WPATH_VIT_H = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_h_4b8939.pth"
WPATH_VIT_L = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_l_0b3195.pth"
WPATH_VIT_B = "C:\\Users\\53588\\Desktop\\Tesis\\Info\\Mejorar Segmentacion\\SAM\\SAM\\sam_vit_b_01ec64.pth"

SCAPE = 27
C = 99
S = 115
ENTER = 13
LEFT = 2424832
RIGHT = 2555904
embedding_path = "C:\\Users\\53588\\Desktop\\Results\\image_embeddings\\data.json"
MASK_COLOR = np.array([255, 144, 30],dtype = np.uint8)

class SesionInfo:
    prompts = []
    prompt_labels = []
    masks = None
    mask_index = None
    scores = None
    ori_img : np.ndarray = None
    current_img : np.ndarray = None
    window_name :str = None
    model = None


def click_event(event, x, y, flags, params):
   img = SesionInfo.current_img
   if event == cv2.EVENT_LBUTTONDOWN:
      SesionInfo.prompts.append([x,y])
      SesionInfo.prompt_labels.append(1)
      # draw point on the image
      cv2.circle(img, (x,y), 3, (0,255,255), -1)
      cv2.imshow(SesionInfo.window_name,img)
   if event == cv2.EVENT_RBUTTONDOWN:
      SesionInfo.prompts.append([x,y])
      SesionInfo.prompt_labels.append(0)
      # draw point on the image
      cv2.circle(img, (x,y), 3, (0,0,255), -1)
      cv2.imshow(SesionInfo.window_name,img)

def click_event_empty(event, x, y, flags, params):
    return

def apply_the_mask(img,mask,color):
    beta = .15
    alpha = 1 - beta
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return cv2.addWeighted(img,alpha,mask_image,beta,0), mask_image

def update_img(new_img):
    SesionInfo.current_img = new_img
    cv2.putText(SesionInfo.current_img, 'SAM', (0, 30), 0, .75, (0,0,255), thickness=2)
    cv2.imshow(SesionInfo.window_name,SesionInfo.current_img)

def clear_sesion():
    SesionInfo.prompts = []
    SesionInfo.prompt_labels = []
    SesionInfo.masks = None
    SesionInfo.mask_index = None
    SesionInfo.scores = None
    SesionInfo.current_img : np.ndarray = None
    SesionInfo.ori_img : np.ndarray = None
    SesionInfo.window_name :str = None


def reset_sesion():
    SesionInfo.prompt_labels = []
    SesionInfo.prompts = []
    SesionInfo.masks = None
    SesionInfo.mask_index = None
    SesionInfo.scores = None
    update_img(SesionInfo.ori_img.copy())

def create_cached_sam_inter(model_type,model_path):
    model = SAM_Model(model_type,model_path)
    model.prepare()
    def cached_sam_inter(img,embedding_path,window_name,image_path):
        SesionInfo.ori_img = img
        SesionInfo.window_name = window_name
        # cargar el modelo
        print('cargando el modelo')
        # haciendo el encoding
        print("encoding de image")
        model.load_embedding(embedding_path)
        print("Encoding terminado")
        cv2.setMouseCallback(SesionInfo.window_name,click_event)
        reset_sesion()
        while True:
            # Pedir el prompt
            cv2.imshow(SesionInfo.window_name,SesionInfo.current_img)
            k = cv2.waitKeyEx(0)
            if k == SCAPE:
                break
            elif k == C:
                reset_sesion()
            elif k == ENTER:
                print('Comenzando a segmentar')
                mask,scores,logits = model.predictor.predict(point_coords= np.array(SesionInfo.prompts),
                                                    point_labels=np.array(SesionInfo.prompt_labels),
                                                    multimask_output= True)
                SesionInfo.masks = mask
                print('Segmentación Terminada')
                SesionInfo.mask_index = 0
                new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                update_img(new_img)
            elif k == RIGHT:
                if not (SesionInfo.masks is None):
                    SesionInfo.mask_index = (abs(SesionInfo.mask_index + 1))%3
                    new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                    update_img(new_img)
            elif k == LEFT:
                if not (SesionInfo.masks is None):
                    SesionInfo.mask_index = (abs(SesionInfo.mask_index - 1))%3
                    new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                    update_img(new_img)
            elif k == S:
                if SesionInfo.prompt_labels is None:
                    continue
                folder_path = 'intresting segmentations'
                image_path = Path(image_path)
                save_name = '-'.join([image_path.parent.name,  image_path.name.split('.')[0]])
                numbers = [int(rpath.name.split('.')[1].split('_')[-2]) for rpath in Path(folder_path).rglob(f'{save_name}*_IMG.png')]
                count  = 1
                if len(numbers) != 0:
                    count = max(numbers) + 1
                save_name += f'_{count}'
                save_path = os.path.join(folder_path,save_name)
                img_masked, mask = apply_the_mask(SesionInfo.ori_img,SesionInfo.masks,MASK_COLOR)
                for point, label in zip(SesionInfo.prompts,SesionInfo.prompt_labels):
                    if label == 1:
                        cv2.circle(img_masked, (point[0],point[1]), 3, (0,255,255), -1)
                    else:
                        cv2.circle(img_masked, (point[0],point[1]), 3, (0,0,255), -1)
                cv2.imwrite(save_path + '_IMG.png',img_masked)
                cv2.imwrite(save_path + '_MASK.png',mask)
                print('image saved')
            else:
                print(k)
        cv2.setMouseCallback(SesionInfo.window_name,click_event_empty)
        cv2.imshow(SesionInfo.window_name,SesionInfo.ori_img)
        clear_sesion()
    
    return cached_sam_inter

def review_images_with_saved_emb(emb_data_path,img_inter):
    emb_data = None
    with open(emb_data_path,'r') as f:
        emb_data = json.load(f)
    arr = None
    first = True
    for index,data in enumerate(emb_data):
        img = cv2.imread(data['path'])
        if first:
            first = False
            arr = np.ndarray(shape = (len(emb_data),*img.shape),dtype=img.dtype)
        arr[index,:,:,:] = img
        
    index = 0
    count = arr.shape[0]
    cv2.namedWindow('REVIEW')
    while True:
        img = arr[index,:,:,:]
        cv2.imshow('REVIEW', img)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index = max(index - 1, 0)
        elif code == RIGHT_ARROW:
            index = min(count - 1, index + 1)
        elif code == SCAPE:
            break
        elif code == SPACE_BAR:
            if img_inter:
                print('>>>> ' + emb_data[index]['path'])
                img_inter(img,emb_data[index]['emb_path'],'REVIEW',emb_data[index]['path'])
        else:
            print(code)
    cv2.destroyAllWindows()

def review_images(img_path_list, img_inter):
    first = True
    arr = None
    for index,img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        if first:
            first = False
            arr = np.ndarray(shape = (len(img_path_list),*img.shape),dtype=img.dtype)
        arr[index,:,:,:] = img
        
    index = 0
    count = arr.shape[0]
    window_name = 'REVIEW'
    cv2.namedWindow(window_name)
    while True:
        img = arr[index,:,:,:]
        cv2.imshow(window_name, img)
        code = cv2.waitKeyEx()
        if code == LEFT_ARROW:
            index = max(index - 1, 0)
        elif code == RIGHT_ARROW:
            index = min(count - 1, index + 1)
        elif code == SCAPE:
            break
        elif code == SPACE_BAR:
            if img_inter:
                print('>>>> ' + img_path_list[index])
                img_inter(img,index,window_name)
        else:
            print(code)
    cv2.destroyAllWindows()   

def create_sam_inter(model_type,model_path):
    print('cargando el modelo')
    model = SAM_Model(model_type,model_path)
    model.prepare()
    def sam_inter(img,index,window_name):
        SesionInfo.ori_img = img
        SesionInfo.window_name = window_name
        # cargar el modelo

        # haciendo el encoding
        print("encoding de image")
        model.set_embedding(SesionInfo.ori_img) 
        print("Encoding terminado")
        cv2.setMouseCallback(SesionInfo.window_name,click_event)
        reset_sesion()
        while True:
            # Pedir el prompt
            cv2.imshow(SesionInfo.window_name,SesionInfo.current_img)
            k = cv2.waitKeyEx(0)
            if k == SCAPE:
                break
            elif k == C:
                reset_sesion()
            elif k == ENTER:
                print('Comenzando a segmentar')
                mask,scores,logits = model.predictor.predict(point_coords= np.array(SesionInfo.prompts),
                                                    point_labels=np.array(SesionInfo.prompt_labels),
                                                    multimask_output= True)
                print('Segmentación Terminada')
                SesionInfo.masks = mask
                SesionInfo.mask_index = 0
                new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                update_img(new_img)
            elif k == RIGHT:
                if not (SesionInfo.masks is None):
                    SesionInfo.mask_index = (abs(SesionInfo.mask_index + 1))%3
                    new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                    update_img(new_img)
            elif k == LEFT:
                if not (SesionInfo.masks is None):
                    SesionInfo.mask_index = (abs(SesionInfo.mask_index - 1))%3
                    new_img,_ = apply_the_mask(SesionInfo.ori_img,mask[SesionInfo.mask_index],MASK_COLOR)
                    update_img(new_img)
            else:
                print(k)
        cv2.setMouseCallback(SesionInfo.window_name,click_event_empty)
        cv2.imshow(SesionInfo.window_name,SesionInfo.ori_img)
        clear_sesion()
    return sam_inter


class SAM_Model:
    def __init__(self,model_type, model_path) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.predictor = None

    def prepare(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path) 
        sam.to(device = "cuda")
        self.predictor =  SamPredictor(sam)

    def set_embedding(self,img,format = 'RGB'):
        self.predictor.set_image(img,format) 
    
    def save_embedding(self,path):
        if not self.predictor.is_image_set:
            raise RuntimeError('No hay embedding')
        res = {
            'original_size': self.predictor.original_size,
            'input_size':self.predictor.input_size,
            'features':self.predictor.features,
            'is_image_set': True
        }
        torch.save(res,path)
    
    def load_embedding(self,path):
        self.predictor.reset_image()
        res = torch.load(path, self.predictor.device)
        for k, v in res.items():
            setattr(self.predictor,k,v)

    def segmentate_from_box(self,x1,y1,x2,y2):
        input_box = np.array([x1, y1, x2, y2])
        masks, _, _ = self.predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
        )
        return masks
    
    def segmentate_from_points(self,fg_points,bg_points,multimask = True):
        points = fg_points + bg_points
        labels = [1]*len(fg_points) + [0]*len(bg_points)
        mask,scores,logits = self.predictor.predict(point_coords= np.array(points),
                                    point_labels=np.array(labels),
                                    multimask_output= multimask)
        return mask
    
    def segmentate_from_box_and_middle_point(self,x1,y1,x2,y2):
        input_box = np.array([x1, y1, x2, y2])
        input_points = np.array([[(x1 + x2)/2,(y1 + y2)/2]])
        label = np.array([1])
        masks, _, _ = self.predictor.predict(
        point_coords=input_points,
        point_labels=label,
        box=input_box[None, :],
        multimask_output=False,
        )
        return masks


def plot_rois(img,x1,y1,x2,y2,color,prob):
	cv2.rectangle(img,(x1, y1), (x2, y2), color)
	textLabel = f'ulcer: {int(100*prob)}'
	(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
	textOrg = (x1, y1-0)
	cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
	cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
	cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        
def save_segmentation(segmentation_path,path,img_masked,mask,current_result,sufix = ''):
    path = Path(path)
    video_name = path.parent.name
    img_name = path.name.split('.')[0]
    folder_path = os.path.join(segmentation_path,video_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    seg_name = ''.join([img_name,sufix, '_seg.png'])
    mask_name = ''.join([img_name,sufix, '_mask.png'])
    seg_path = os.path.join(folder_path,seg_name)
    mask_path = os.path.join(folder_path,mask_name)
    cv2.imwrite(seg_path,img_masked)
    cv2.imwrite(mask_path,mask)
    current_result.append({
        'path':str(path),
        'seg_path':str(seg_path),
        'mask_path':str(mask_path)
    })
    data_path = os.path.join(segmentation_path,'data.json')
    with open(data_path,'w') as f:
        json.dump(current_result,f)


def segmentate_with_simple_sam():
    print('Initialasing model')
    model = SAM_Model(TYPE_VIT_B,WPATH_VIT_B)
    model.prepare()
    print('Reading images and rois')
    roi_list = None
    roi_list_path = "C:\\Users\\53588\\Desktop\\Tesis\\dev\\testing rccn with SAM\\rois.json" 
    with open(roi_list_path,'r') as f:
        roi_list = json.load(f) 
    result_path = 'sam_results'
    results = []
    for roi in  roi_list:
        print('loading image')
        img = cv2.imread(roi['path'])
        if not roi['empty']:
            print('generating embedding')
            model.set_embedding(img)
            print('Segmentating')
            mask = model.segmentate_from_box(roi['x1'],roi['y1'],roi['x2'],roi['y2'])
            img,mask_img = apply_the_mask(img,mask,MASK_COLOR)
            plot_rois(img,roi['x1'],roi['y1'],roi['x2'],roi['y2'],(0,0,255),roi['prob'])
            save_segmentation(roi['path'],img,mask_img,results)
            # cv2.imshow('ROI',img)
            # cv2.imshow('Mask',mask_img)
            # cv2.waitKey()
        else:
            print('Not Roi')
            # cv2.imshow('Not ROI',img)
            # cv2.waitKey()
        # cv2.destroyAllWindows()



def segment_with_middle_point_sam(embedding_path,result_path):
    print('Initialasing model')
    embedding_list = None
    with open(embedding_path,'r') as f:
        embedding_list = json.load(f)
    model = SAM_Model(TYPE_VIT_B,WPATH_VIT_B)
    model.prepare()
    print('Reading images and rois')
    roi_list = None
    roi_list_path = "C:\\Users\\53588\\Desktop\\Tesis\\dev\\testing rccn with SAM\\rois.json" 
    with open(roi_list_path,'r') as f:
        roi_list = json.load(f) 
    embedding_data = []
    results = []
    embedding_index = 0
    for index, roi in  enumerate(roi_list):
        print(f'loading image {index}/{len(roi_list)}')
        img = cv2.imread(roi['path'])
        if not roi['empty']:
            embedding = embedding_list[embedding_index]
            print('generating embedding')
            model.load_embedding("C:\\Users\\53588\\Desktop\\Results\\" + embedding['emb_path'])
            print('Segmentating')
            mask = model.segmentate_from_box_and_middle_point(roi['x1'],roi['y1'],roi['x2'],roi['y2'])
            uni,mask_uni = apply_the_mask(img,mask[0],MASK_COLOR)
            plot_rois(uni,roi['x1'],roi['y1'],roi['x2'],roi['y2'],(0,0,255),roi['prob'])
            cv2.circle(uni, (int((roi['x1'] + roi['x2'])/2),int((roi['y1'] + roi['y2'])/2)), 3, (0,255,255), -1)
            # cv2.imshow('ROI_uni',uni)
            # cv2.imshow('Mask',mask_uni)
            # cv2.waitKey()
            save_segmentation(result_path,roi['path'],uni,mask_uni,results,'')
            embedding_index += 1
        else:
            print('Not Roi')
            # cv2.imshow('Not ROI',img)
            # cv2.waitKey()
        # cv2.destroyAllWindows()

def save_embeddings(path_list,out_folder,model_type,model_path):
    # Crear el modelo
    model = SAM_Model(model_type,model_path)
    model.prepare()

    result = []
    # crear carpeta para los resultados
    if not os.path.exists(out_folder):
       os.mkdir(out_folder)

    data_path = os.path.join(out_folder,'data.json')
    for index, img_path in enumerate(path_list):
        print(f'{index + 1}/{len(path_list)}')
        # Cargar la imagen
        img = cv2.imread(str(img_path))
        # Crear el embedding
        model.set_embedding(img)
        # Crear el path del embedding
        path = Path(img_path)
        img_name = path.name.split('.')[0]
        emb_name = img_name + '.torch'
        emb_path = os.path.join(out_folder,emb_name)
        # Salvando el embedding
        model.save_embedding(emb_path)
        
        # Anotando los datos
        result.append({
            'path':str(img_path),
            'emb_path':str(emb_path),
        })
        with open(data_path,'w') as f:
            json.dump(result,f)
    return data_path

# save_embeddings("C:\\Users\\53588\\Desktop\\Results\\bad_data",'bad_frames_emb_vit_L',TYPE_VIT_L,WPATH_VIT_L)

    


# segment_with_middle_point_sam('mixing_box_and_point')
# # cv2.imwrite('.\\sam_results\\pedro\\test2.png',img)
# review_images_in_a_folder('bad_frames_emb_vit_L\data.json',create_sam_inter(TYPE_VIT_L,WPATH_VIT_L))
# print('desco')