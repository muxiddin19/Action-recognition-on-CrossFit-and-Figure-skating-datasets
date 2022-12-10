import random
import json
import os
import pickle
import csv
import argparse
import numpy as np
import math
from tqdm import tqdm
import shutil
import cv2 as cv
import sys
import tarfile
import glob
from pathlib import Path
from os.path import join,getsize

def load_data(jsons, path, label):
    i = 0
    data= {}
    data['frame_dir'] = path
    data['label'] = label
    data['img_shape'] = (1920,1080)
    data['original_shape'] = (1920,1080)
    
    frames = {}
    scores = {}
    for json in jsons:
        with open(json, 'r') as f:
#        with open(json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        start_frame = data['annotations'][0]['start_frame']
        end_frame = data['annotations'][0]['end_frame']
    return start_frame, end_frame

def get_start_end_frame(path0):
    #path = 'E:/Posec3d_new/data/elancer/20220803/AI/크로스핏/데드리프트/데드리프트/가슴및고관절오류/중급/박기형/1/camera1/video/annotation.json'
    #path = str(path)
    #path = 
    with open(path0, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    try:     

        start_frame0 = data1['annotations'][0]['start_frame']
        end_frame0 = data1['annotations'][0]['end_frame']
    except:
        print('There is no metadata provided in path = ', path0)
        start_frame0 = 0   
        end_frame0 = 1000
    return start_frame0, end_frame0

def load_json(path):
    jsonList = []
    with open(os.path.join(path), "r") as f:
    #with open(os.path.join(path), "r", encoding='UTF8') as f:
        for jsonObj in f:
            jsonList = json.loads(jsonObj)
    #print(jsonList) 
    return jsonList

def cut_frame(pairs, name, is_full):
    print('Cutting frames ..., Making train set...')
    #label_dict = {}
    #train = []
    #val = []
   # test = []
   # i = 0
    for items in pairs.keys():
#    for label, items in pairs.keys():
        #if not is_full and '오류' in label:
           # continue
        #tmp_train = []
        #tmp_val = []
        #tmp_test = []
    
        for item in items:
            #jsons= load_json(item)
            #key, values = load_data(jsons, key, i)
            #print('item = ', item)
            #path = item
            #with open(path, 'r') as f: 
            #with open(path, 'r', encoding='utf-8') as f: 
                #data = json.load(f)
                #s = data['annotations'][0]['start_frame']
                #e = data['annotations'][0]['end_frame']
#            path = r'../item' + '/color'
            start_frame, end_frame = get_start_end_frame(item)
            path = r'../item'
            path = os.path.abspath('..')
            path = path + '/color'
            #path = r'E:\Posec3d_new\data\elancer\20220705~20220711\AI\크로스핏\데드리프트\데드리프트\오류1\고급\김태경\1\camera0\color'
            os.chdir(path)
            fname = os.listdir(".")[0]
            #fname
            if fname.endswith("tar"):
                tar= tarfile.open(fname, "r:")
                tar.extractall()
                tar.close
#out_path = 'E:\Posec3d_new\data\elancer\20220705~20220711\AI\크로스핏\데드리프트\데드리프트\오류1\고급\김태경\1\camera0\color'
            out_path = '../item/color'
            out_video_name = 'result.mp4'
            out_video_full_path = out_path + '\\' + out_video_name
#s = 129
#e = 347
            pre_imgs = os.listdir(path)
#print(pre_imgs)
            img = []
            for i in pre_imgs:
                i = path + i
    #print(i)
                img.append(i)
#print((img[1]))
#print(img)
            cv_fourcc = cv.VideoWriter_fourcc(*'mp4v')
            frame = cv.imread(img[0])
            size = (1920, 1080)
#print(size)
#print(frame)
            video = cv.VideoWriter(out_video_full_path, cv_fourcc, 24, size)

#for i in range(len(img)):
            for i in range(len(img[e-s+1])):
                video.write(cv.imread(img[s+i]))
                print('frame', s+i, ' of ', len(img))
            video.release()
#print(out_video_full_path)
    return out_video_full_path   

def make_train_val_test(pairs, name, is_full):
    print("Making Trainset...")
    label_dict = {}
    train = []
    val = []
    test = []	
    i = 0
    for label, items in pairs.items():
        print('label = ', label)
        print('items = ', items)
        if not is_full and "오류" in label:
            continue
        tmp_train = []
        tmp_test = []
        tmp_val = []
        
        for item in items:
            
            jsons = load_json(item)
            a_data, ids = load_data(jsons, item, i)
            tmp_train.append(a_data)
        n_val = math.ceil(len(tmp_train) * 0.1)
        n_test = math.ceil(len(tmp_train) * 0.1)
        	
        for _ in range(n_test):
            tmp_test.append(tmp_train.pop(random.choice(range(len(tmp_train)))))
        #for _ in range(n_val):
	#tmp_val.append(tmp_train.pop(random.choice(range(len(tmp_train)))))

        train += tmp_train
        
        #for _ in range(n_val):
	#tmp_val.append(tmp_train.pop(random.choice(range(len(tmp_train)))))

         
        val += tmp_val
 
        test += tmp_test
        print(label, len(items), "->", len(train), len(val), len(test))
        label_dict[label] = i
        i += 1
    with open("./data/normal_label_dict.pkl", "wb") as f:
#    with open("./data/elancer_only_normal.pkl", "wb") as f:

        pickle.dump(label_dict, f)
    with open("./data/"+name+"_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("./data/"+name+"_val.pkl", "wb") as f:
        pickle.dump(val, f)

    with open("./data/"+name+"_test.pkl", "wb") as f:
        pickle.dump(test, f)
    return train, val, test 

def make_dict(dict0, folder):
    #details = {'full_path' : full_path, 'frame' : {s_f, e_f}}
    #with open('meta_data_KF' + folder + '.pkl', 'wb') as f:   #KrossFit
    with open('meta_data_FS' + folder + '.pkl', 'wb') as f:   #Figure skating
        pickle.dump(dict0, f)
        #for key, value in details.items():
            #f.write(json.dumps(details))
            #f.write('%s:%s\n' % (key, value))
    return

parser = argparse.ArgumentParser(description='run scripts.')

parser.add_argument("--folders",nargs="+", type=str)
parser.add_argument("--full", action='store_true', default=False)
args = parser.parse_args()

folders = args.folders
train_all = []
val_all = []
test_all = []
dict = []
dict1 = []
for folder in folders:
    json_path = "data/elancer/"+folder+"/"   #Figure skating

    #json_path = "data/Fugure_scating/" + folder +"/"
    #json_path = "H:/" + folder +"/"  #Krossfit
    #json_path = folder+"/"
    #json_path = folder+"/"
    json_data_pairs = {}
    for path in os.listdir(json_path):
        tmp_path = os.path.join(json_path, path)
        #print('tmp_path=', tmp_path)
        #if "*.xlsx" in tmp_path:
            #continue
        for path1 in os.listdir(tmp_path):
            tmp_path1 = os.path.join(tmp_path, path1)
            #print('tmp_path1=', tmp_path1)
            #if "*.xlsx" in tmp_path1:
                #continue
            #if 'Labelling_version_1.1.11.exe' in tmp_path1:
                #continue
            for path2 in os.listdir(tmp_path1):
                tmp_path2 = os.path.join(tmp_path1, path2)
                #if 'Labelling_version_1.1.11.exe' in tmp_path2:
                  #  continue
               # if "*.xlsx" in tmp_path2:
                   # continue
              #  if not "annotation.json" in tmp_path2:
                   # continue
                #print('tmp_path2=', tmp_path2)
                for path3 in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, path3)
                    #print('tmp_path3=', tmp_path3)
                   # if not "annotation.json" in tmp_path3:
                       # continue
                    for path4 in os.listdir(tmp_path3):
                        tmp_path4 = os.path.join(tmp_path3, path4)
                        #print('tmp_path2=', tmp_path2)

                        for path5 in os.listdir(tmp_path4):
                            tmp_path5 = os.path.join(tmp_path4, path5)
                           # print('tmp_path2=', tmp_path2)

                            label = path3 + "_" + path4 + "_" + path5
                           # print('label=', label)

                            for path6 in os.listdir(tmp_path5):
                                tmp_path6 = os.path.join(tmp_path5, path6)
                                #print('tmp_path6=',tmp_path6)
                                for path7 in os.listdir(tmp_path6):
                                    tmp_path7 = os.path.join(tmp_path6, path7)
                                    #print('tmp_path7=',tmp_path7)
                                    for path8 in os.listdir(tmp_path7):
                                        target = os.path.join(tmp_path7, path8)
                                        #print('target=',target)
                                        tmp_path8 = os.path.join(tmp_path7, path8)
                                        
                                        #make_video(path_v, start_frame, end_frame)
                                        if 'result_label.save.done' in target:
                                            continue
                                        if 'result_label.save.md5hash.txt' in target:
                                            continue
                                        if 'result_label.save.temp' in target:
                                            continue
                                        if 'ErrorMemo.txt' in target:
                                                continue
                                       # print('tmp_path8=', tmp_path8)
                                        #path_v = open_tar_file(tmp_path8)
                                        for path9 in os.listdir(tmp_path8):
                                            #if not 'annotation.json' in tmp_path9:
                                               #continue
                                            tmp_path9 = os.path.join(tmp_path8, path9)
                                            #print('tmp_path9=',tmp_path9)
                                            #if not 'annotation.json' in tmp_path9:
                                                #continue
                                            #if 'ErrorMemo.txt' in tmp_path9:
                                                #continue
                                            #print('tmp_path9=',tmp_path9)
                                            #open_tar_file(tmp_path9)
                                            #print(type(tmp_path9))
                                            for path10 in os.listdir(tmp_path9):
                                                target = os.path.join(tmp_path9, path10)
                                                #open_tar_file(target)
                                                #if target.endswith("tar"):
                                                    #target1 = os.path.join(tmp_path9, path10)
                                                    #open_tar_file(target)
                                                if not "annotation.json" in target:
                                                    continue
                                                #print('target =', target)    
                                                if label not in json_data_pairs.keys():
                                                    json_data_pairs[label] = [target]
                                                else:
                                                    json_data_pairs[label].append(target)
                                                #print('Suspicious target =', target)    
                                                start_frame, end_frame = get_start_end_frame(target)
                                                dict1 = {'full_path' : target, 'frame' : [start_frame, end_frame]} 
                                                #print('dict1 = ', dict1)
                                                #print('Len dict1 = ', len(dict1))
                                                #dict += dict1
                                                dict.append(dict1)
                                                #print()
                                                #print('dict = ', dict)
    make_dict(dict, folder)
    #make_dict(dict)
    #train, val, test = make_train_val_test(json_data_pairs, folder, args.full)
    #cut_frame(json_data_pairs, folder, args.full)

    #train_all += train
    #val_all += val
    #test_all += test

#load_json(path)
#if len(folders) > 1:
 #   with open("./data/elancer/train.pkl", "wb") as f:
  #      pickle.dump(train_all, f)
   # with open("./data/elancer/val.pkl", "wb") as f:
    #    pickle.dump(val_all, f)
    #with open("./data/elancer/test.pkl", "wb") as f:
     #   pickle.dump(test_all, f)