#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('./')
import json
import StringIO
import zipfile
import re
from datetime import datetime
import importlib
import sqlite3
import rrc_evaluation_funcs
from config import *
import pickle
import cv2
import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
import operator

try:
    from bottle import route, run, request, static_file, url, template, TEMPLATE_PATH, HTTPResponse, redirect
except ImportError:
    print """Required module not found: Bottle. Installation: pip install --user bottle"""
    sys.exit(-1)

try:
    from PIL import Image
except ImportError:
    print """Required module not found: Pillow. Installation: pip install --user Pillow"""
    sys.exit(-1)

p = {
    'g': os.path.dirname(os.path.abspath(__file__)) + "/gt/gt." + gt_ext,
    # 's': '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/icdar15_rbox/icdar15_submit.zip',
    'o': os.path.dirname(os.path.abspath(__file__)) + "/output",
    'p': evaluation_params
}

img_dir = '/home/mhliao/data/icdar15/test_images/'


def list_from_str(st):
    line = st.split(',')
    new_line = [float(a) for a in line[0:8]] + [float(line[-1])]
    return new_line


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    # polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def nms(boxes, overlap):
    rec_scores = [b[-1] for b in boxes]
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    box_num = len(boxes)
    nms_flag = [True] * box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj]
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj]
            # str1 = box1[9]
            # str2 = box2[9]
            # box_i = [box1[0], box1[1], box1[4], box1[5]]
            # box_j = [box2[0], box2[1], box2[4], box2[5]]
            poly1 = polygon_from_list(box1[0:8])
            poly2 = polygon_from_list(box2[0:8])
            iou = polygon_iou(box1[0:8], box2[0:8])
            thresh = overlap

            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area <= poly2.area:
                    nms_flag[ii] = False
                    break
            '''
            if abs((box_i[3]-box_i[1])-(box_j[3]-box_j[1]))<((box_i[3]-box_i[1])+(box_j[3]-box_j[1]))/2:
                if abs(box_i[3]-box_j[3])+abs(box_i[1]-box_j[1])<(max(box_i[3],box_j[3])-min(box_i[1],box_j[1]))/3:
                    if box_i[0]<=box_j[0] and (box_i[2]+min(box_i[3]-box_i[1],box_j[3]-box_j[1])>=box_j[2]):
                        nms_flag[jj] = False
            '''
    return nms_flag


def packing(save_dir, pack_dir, pack_name):
    files = os.listdir(save_dir)
    if not os.path.exists(pack_dir):
        os.mkdir(pack_dir)
    os.system('zip -r -j ' + os.path.join(pack_dir, pack_name + '.zip') + ' ' + save_dir + '/*')

def test_single(dt_dir, score_det, overlap, zip_dir, pack_dir, result_list, vis_path=''):
    print score_det, overlap
    if not os.path.exists(zip_dir):
        os.mkdir(zip_dir)
    nms_dir = os.path.join(zip_dir, str(score_det) + '_over' + str(overlap))
    if not os.path.exists(nms_dir):
        os.mkdir(nms_dir)
    for i in range(1, 501):
        img = 'img_' + str(i) + '.jpg'
        print img
        image = cv2.imread(os.path.join(img_dir, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_img = 'gt_img_' + str(i) + '.txt'
        with open(os.path.join('./gt/gt', gt_img)) as f:
            ori_gt_lines = [o.decode('utf-8-sig').encode('utf-8') for o in f.readlines()]
        ori_gt_coors = [g.strip().split(',')[0:8] for g in ori_gt_lines]
        ori_gt_lines = [g.strip().split(',') for g in ori_gt_lines]
        for ii, g in enumerate(ori_gt_coors):
            x1 = int(g[0])
            y1 = int(g[1])
            x2 = int(g[2])
            y2 = int(g[3])
            x3 = int(g[4])
            y3 = int(g[5])
            x4 = int(g[6])
            y4 = int(g[7])
            ori_gt_coors[ii] = [x1, y1, x2, y2, x3, y3, x4, y4]
        with open(os.path.join(dt_dir, 'res_img_' + str(i) + '.txt'), 'r') as f:
            dt_lines = [a.strip() for a in f.readlines()]
        dt_lines = [list_from_str(dt) for dt in dt_lines]
        test_coors = []
        for t in dt_lines:
            if t[8] > score_det:
                test_coors.append(t[0:8])

        dt_lines = [dt for dt in dt_lines if dt[-1] > score_det]
        nms_flag = nms(dt_lines, overlap)
        boxes = []
        for k in range(len(dt_lines)):
            dt = dt_lines[k]
            if nms_flag[k]:
                if dt not in boxes:
                    boxes.append(dt)

        with open(os.path.join(nms_dir, 'res_img_' + str(i) + '.txt'), 'w') as f:
            for g in boxes:
                gt_coors = [int(b) for b in g[0:8]]
                # gt_coor_strs = [str(a) for a in gt_coors]+ [g[-2]]
                gt_coor_strs = [str(a) for a in gt_coors]
                f.write(','.join(gt_coor_strs) + '\r\n')

        if vis_path:
            if len(boxes) > 0:
                hit_gts = []
                bad_dts = []
                hit_dts = []
                max_iou = []
                dictlist = [defaultdict(int) for x in range(len(ori_gt_lines))]
                for index_dt in range(len(boxes)):
                    ori_dt = boxes[index_dt][0:8]
                    ious = []
                    for index_gt in range(len(ori_gt_coors)):
                        ori_gt = ori_gt_coors[index_gt]
                        poly1 = polygon_from_list(ori_gt)
                        poly2 = polygon_from_list(ori_dt)
                        iou = polygon_iou(ori_gt, ori_dt)
                        # print 'iou',iou
                        ious.append(iou)
                    max_iou = max(ious)
                    max_index = ious.index(max_iou)
                    if max_iou > 0.5:
                        dictlist[max_index][index_dt] = max_iou
                dt_gt = defaultdict(int)
                for index_gt_dts, gt_dts in enumerate(dictlist):
                    if len(gt_dts) == 0:
                        continue
                    else:
                        sorted_dts = sorted(gt_dts.items(), key=operator.itemgetter(1))
                        dt_gt[sorted_dts[0][0]] = index_gt_dts

                for index_dt, bb in enumerate(boxes):
                    # matched gt and dt
                    if index_dt in dt_gt.keys():
                        index_gt = dt_gt[index_dt]
                        hit_gts.append(ori_gt_lines[index_gt])
                        hit_dts.append(bb)
                        # print 'match gt,dt',ori_gt_lines[index_gt],bb
                bad_dts = [item for item in boxes if item not in hit_dts]
                # miss_gts = list(set(ori_gt_lines)^set(hit_gts))
                miss_gts = [item for item in ori_gt_lines if item not in hit_gts]
                miss_gts = [gg for gg in miss_gts if '#' not in gg[-1]]

                if not os.path.exists(vis_path):
                    os.mkdir(vis_path)
                plt.clf()

                plt.imshow(image, aspect='normal')
                currentAxis = plt.gca()
                for index in range(len(hit_dts)):
                    res = hit_dts[index][0:8]
                    res = [int(a) for a in res]
                    res = np.array(res).reshape(4, 2)
                    currentAxis.add_patch(plt.Polygon(res, fill=None, edgecolor='#00FF00', linewidth=1))
                    # rec_str = hit_dts[index][-2]
                    axis_x = res[0][0] - 4
                    axis_y = res[0][1] - 4
                    # currentAxis.text(axis_x, axis_y, rec_str, color='#FFFF00',fontsize=5,fontweight='bold')
                for index in range(len(bad_dts)):
                    res = bad_dts[index][0:8]
                    res = [int(a) for a in res]
                    res = np.array(res).reshape(4, 2)
                    currentAxis.add_patch(plt.Polygon(res, fill=None, edgecolor='r', linewidth=1))
                    # rec_str = bad_dts[index][-2]
                    axis_x = res[0][0] - 10
                    axis_y = res[0][1] - 10
                    # currentAxis.text(axis_x, axis_y, rec_str, color='#FFFF00',fontsize=12)
                for index in range(len(miss_gts)):
                    res = miss_gts[index][0:8]
                    res = [int(a) for a in res]
                    res = np.array(res).reshape(4, 2)
                    currentAxis.add_patch(plt.Polygon(res, fill=None, linestyle='dashdot', edgecolor='r', linewidth=1))
                    # rec_str = miss_gts[index][-2]
                    axis_x = res[0][0]
                    axis_y = res[0][1]
                    # currentAxis.text(axis_x, axis_y, rec_str, color='#FFFF00',fontsize=18)
                plt.axis('off')
                plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                                labeltop='off', labelright='off', labelbottom='off')
                plt.savefig(os.path.join(vis_path, img), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
    pack_name = str(score_det) + '_over' + str(overlap)

    packing(nms_dir, pack_dir, pack_name)
    submit_dir = os.path.join(pack_dir, pack_name + '.zip')

    res = online_test(submit_dir, result_list)
    res = [str(float(a)) for a in res]
    res = [str(score_det), str(overlap)] + res
    with open(result_list, 'a') as f1:
        f1.write(','.join(res) + '\n')


def test_all(dt_dir, zip_dir, pack_dir, result_list, pkl_file, vis_path=''):
    score_det_range = [0]
    # score_rec_range = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
    score_rec_range = [0.05, 0.06]
    overlap_range = [a / 100.0 for a in range(22, 27, 2)]
    score_det = 0
    for score_rec in score_rec_range:
        for overlap in overlap_range:
            if not os.path.exists(pkl_file):
                tested_list = []
                with open(pkl_file, 'wb') as f:
                    pickle.dump(tested_list, f)
            else:
                with open(pkl_file, 'rb') as f:
                    tested_list = pickle.load(f)
                to_test = [score_det, score_rec, overlap]
                to_test = [str(i) for i in to_test]
                to_test = ','.join(to_test)
                if to_test not in tested_list:
                    test_single(dt_dir, score_det, score_rec, overlap, zip_dir, pack_dir, result_list, vis_path)
                    tested_list.append(to_test)
                    with open(pkl_file, 'wb') as f:
                        pickle.dump(tested_list, f)
                else:
                    continue

def online_test(submit_dir, result_list):
    for k, _ in submit_params.iteritems():
        p['p'][k] = request.forms.get(k)
    module = importlib.import_module("config." + evaluation_script)
    # resDict = rrc_evaluation_funcs.main_evaluation(p,module.default_evaluation_params,module.validate_data,module.evaluate_method)
    evalParams = module.default_evaluation_params()
    print(evalParams)
    if 'p' in p.keys():
        evalParams.update(p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]))
    module.validate_data(p['g'], submit_dir, evalParams)
    evalData = module.evaluate_method(p['g'], submit_dir, evalParams)
    resDict = {'calculated': True, 'Message': '', 'method': '{}', 'per_sample': '{}'}
    resDict.update(evalData)
    # print(resDict['method'])
    precision = resDict['method']['precision']
    recall = resDict['method']['recall']
    fmeasure = resDict['method']['hmean']
    print 'p,r,f', precision, recall, fmeasure
    return [precision, recall, fmeasure]


def visulization(img_dir, bbox_dir, visu_dir):
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            # print(file)
            # if file!='img_75.jpg':
            #   continue
            print(file)
            image_name = file
            img_path = os.path.join(img_dir, image_name)
            img = cv2.imread(img_path)
            plt.clf()
            plt.imshow(img)
            currentAxis = plt.gca()
            bbox_name = 'res_' + file[0:len(file) - 3] + 'txt'
            bbox_path = os.path.join(bbox_dir, bbox_name)
            if os.path.isfile(bbox_path):
                with open(bbox_path, 'r') as f:
                    count = 1
                    for line in f.readlines():
                        line = line.strip()
                        x1 = line.split(',')[0]
                        y1 = line.split(',')[1]
                        x2 = line.split(',')[2]
                        y2 = line.split(',')[3]
                        x3 = line.split(',')[4]
                        y3 = line.split(',')[5]
                        x4 = line.split(',')[6]
                        y4 = line.split(',')[7]
                        rbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        color_rbox = 'r'
                        currentAxis.add_patch(plt.Polygon(rbox, fill=False, edgecolor=color_rbox, linewidth=1))
                        # currentAxis.text(int(x1), int(y1), str(count), bbox={'facecolor':'white', 'alpha':0.5})
                        count = count + 1

                plt.axis('off')
                plt.savefig(visu_dir + image_name, dpi=300)


if __name__ == '__main__':
    dt_dir = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/icdar15_polygon_multiscale0/'
    score_det = 0.6
    overlap = 0.2
    zip_dir = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/detection_zip/'
    pack_dir = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/detection_zip/'
    result_list = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/detection_zip/result.txt'
    test_single(dt_dir, score_det, overlap, zip_dir, pack_dir, result_list, vis_path='./visu_result/')

