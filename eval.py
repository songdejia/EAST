import cv2
import time
import math
import os
import numpy as np
from PIL import Image
import locality_aware_nms as nms_locality
import lanms
import shutil
import torch
import model
from data_utils import restore_rectangle, polygon_area
from torch.autograd import Variable
import config as cfg
import sys
from torchvision import transforms
import model
test_data_path = cfg.test_img_path


def rotate(box_List,image):
    #xuan zhuan tu pian

    n=len(box_List)
    c=0;
    angle=0
    for i in range(n):
        box=box_List[i]
        y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
        x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
        for j in range(4):
            if(box[j][1]==y2):
                k1=j
        for j in range(4):
            if(box[j][0]==x2 and j!=k1):
                k2=j
        c=(box[k1][0]-box[k2][0])*1.0/(box[k1][1]-box[k2][1])
        if(c<0):
            c=-c
        if(c>1):
            c=1.0/c
        angle=math.atan(c)+angle
    angle=angle/n
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    scale=1
    M = cv2.getRotationMatrix2D(center,angle, scale)
    image_new = cv2.warpAffine(image, M, (w, h))
    return image_new

def get_images_for_test():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    
    # print('Find {} images'.format(len(files)))
    return files

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    """
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    """

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    #resize_h, resize_w = 512, 512
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def detect(score_map, geo_map, timer, score_map_thresh=1e-5, box_thresh=1e-8, nms_thres=0.1):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def change_box(box_List):
    n=len(box_List)
    for i in range(n):
        box=box_List[i]
        y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
        x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
        box[0][1]=y1
        box[0][0]=x1
        box[1][1]=y1
        box[1][0]=x2
        box[3][1]=y2
        box[3][0]=x1
        box[2][1]=y2
        box[2][0]=x2
        box_List[i]=box
    return box_List

def save_box(box_List,image,img_path):
    n=len(box_List)
    box_final = []
    for i in range(n):
        box=box_List[i]
        y1_0 = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))
        y2_0 = int(max(box[0][1], box[1][1], box[2][1], box[3][1]))
        x1_0 = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))
        x2_0 = int(max(box[0][0], box[1][0], box[2][0], box[3][0]))
        y1 = max(int(y1_0 - 0.1 * (y2_0 - y1_0)), 0)
        y2 = min(int(y2_0 + 0.1 * (y2_0 - y1_0)), image.shape[0] - 1)
        x1 = max(int(x1_0 - 0.25 * (x2_0 - x1_0)), 0)
        x2 = min(int(x2_0 + 0.25 * (x2_0 - x1_0)), image.shape[1] - 1)
        image_new=image[y1:y2,x1:x2]

        # # 图像处理
        gray_2 = image_new[:,:,0]
        gradX = cv2.Sobel(gray_2, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        gradY = cv2.Sobel(gray_2, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
        blurred = cv2.blur(gradX, (2, 2))
        (_, thresh) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        # closed = cv2.erode(thresh, None, iterations = 1)
        # closed = cv2.dilate(closed, None, iterations = 1)
        closed = thresh
        x_plus = []
        x_left = 1
        x_right = closed.shape[1]
        for jj in range(0, closed.shape[1]):
            plus = 0
            for ii in range(0, closed.shape[0]):
                plus = plus + closed[ii][jj]
            x_plus.append(plus)

        for jj in range(0, int(closed.shape[1] * 0.5 - 1)):
            if(x_plus[jj] > 0.4 * max(x_plus)):
                x_left = max(jj - 5, 0)
                break
        for ii in range(closed.shape[1] - 1, int(closed.shape[1] * 0.5 + 1), -1):
            if(x_plus[ii] > 0.4 * max(x_plus)):
                x_right = min(ii + 5, closed.shape[1] - 1)
                break

        image_new = image_new[:, x_left:x_right]
        cv2.imwrite("." + img_path.split(".")[1]+'_'+str(i)+".jpg", image_new)
        box[0][1]=y1
        box[0][0]=x1 + x_left
        box[1][1]=y1
        box[1][0]=x1 + x_right
        box[3][1]=y2
        box[3][0]=x1 + x_left
        box[2][1]=y2
        box[2][0]=x1 + x_right
        box_List[i]=box
    return box_List

def transform_for_test():
    """
    CV2 => PI => tensor
    """
    #image = Image.fromarray(np.uint8(img))

    transform_list = []
    
    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))

    transform = transforms.Compose(transform_list)

    return transform


def predict(model, criterion, epoch):
    # prepare ooutput directory
    print('EAST <==> TEST <==> Create Res_file and Img_with_box <==> Begin')
    result_root = os.path.abspath(cfg.result)
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    output_dir_txt = os.path.join(result_root, 'epoch_'+str(epoch)+'_gt')
    output_dir_pic = os.path.join(result_root, 'epoch_'+str(epoch)+'_img_with_box')

    try:
        shutil.rmtree(output_dir_txt)
    except:
        print('Dir {} is not exists, make it'.format(output_dir_txt))
        
    try:
        shutil.rmtree(output_dir_pic)
    except:
        print('Dir {} is not exists, make it'.format(output_dir_pic))

    try:
        os.mkdir(output_dir_txt)
        os.mkdir(output_dir_pic)

    except OSError as e:
        if e.errno != 17:
            raise

    print('EAST <==> TEST <==> Create Res_file and Img_with_box Directory ')

    model = model.eval()
    im_fn_list = get_images_for_test()
    start = time.time()

    for idx, im_fn in enumerate(im_fn_list):    
        print('EAST <==> TEST <==> idx:{} <==> Begin'.format(idx))
        im = cv2.imread(im_fn)[:, :, ::-1]


        #transform = transform_for_test()


        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(im)
        im_resized = im_resized.astype(np.float32)
        #image = Image.fromarray(np.uint8(im_resized))

        #im_resized = transform(image) 
        im_resized = im_resized.transpose(2, 0, 1)
        im_resized = torch.from_numpy(im_resized)
        im_resized = im_resized.cuda()
        im_resized = im_resized.unsqueeze(0)
        #im_resized = im_resized.permute(0, 3, 1, 2)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        
        score, geometry = model(im_resized)

        timer['net'] = time.time() - start
        print('EAST <==> TEST <==> idx:{} <==> model  :{:.2f}ms'.format(idx, timer['net']*1000))

        score = score.permute(0, 2, 3, 1)
        geometry = geometry.permute(0, 2, 3, 1)
        score = score.data.cpu().numpy()
        geometry = geometry.data.cpu().numpy()

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        print('EAST <==> TEST <==> idx:{} <==> restore:{:.2f}ms'.format(idx, timer['restore']*1000))
        print('EAST <==> TEST <==> idx:{} <==> nms    :{:.2f}ms'.format(idx, timer['nms']*1000))


        print('EAST <==> TEST <==> Record and Save <==> id:{} <==> Begin'.format(idx))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        if boxes is not None:
            res_file = os.path.join(output_dir_txt, 'res_img_{}.txt'.format(os.path.basename(im_fn).split('_')[-1].strip('.jpg')))
            with open(res_file, 'w') as f:
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))

                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        #print('wrong direction')
                        continue
                    
                    if box[0, 0] < 0 or box[0, 1] < 0 or box[1,0] < 0 or box[1,1] < 0 or box[2,0]<0 or box[2,1]<0 or box[3,0] < 0 or box[3,1]<0:
                        continue
                        
                    poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]], [box[3, 0], box[3, 1]]])
                    
                    p_area = polygon_area(poly)
                    if p_area > 0:
                        poly = poly[(0, 3, 2, 1), :]

                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(poly[0, 0], poly[0, 1], poly[1, 0], poly[1, 1], poly[2, 0], poly[2, 1], poly[3, 0], poly[3, 1],))
                    cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
        
        save_img_path = os.path.join(output_dir_pic, os.path.basename(im_fn))
        cv2.imwrite(save_img_path, im[:, :, ::-1])
        print('EAST <==> TEST <==> Save txt   at:{} <==> Done'.format(res_file))
        print('EAST <==> TEST <==> Save image at:{} <==> Done'.format(save_img_path))

        print('EAST <==> TEST <==> Record and Save <==> ids:{} <==> Done'.format(idx))
    return  output_dir_txt

if __name__ == "__main__":
    predict()
