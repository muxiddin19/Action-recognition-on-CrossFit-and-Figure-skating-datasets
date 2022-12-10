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
import cv2
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
        leng = 3
        instance = []
        score = []
        for i in range(int(len(json["keypoints"])/3)):
            instance.append([json["keypoints"][i*leng], json["keypoints"][i*leng+1]])
            score.append(json["keypoints"][i*leng+2])
        key = int(json["image_id"].split(".")[0])
        if key not in frames.keys():
            frames[key] = [instance]
            scores[key] = [score]
        else:
            frames[key].append(instance)
            scores[key].append(score)
    frame = [value[0] for key, value in frames.items()]
    ids = [key for key,value in frames.items()]
    score = [value[0] for key, value in scores.items()] 
    data['total_frames'] = len(frames)
    data['keypoint'] = np.array([frame])
    data['keypoint_score'] = np.array([score])
    return data, ids

def load_json(path):
    jsonList = []
    with open(os.path.join(path), "r") as f:
        for jsonObj in f:
            jsonList = json.loads(jsonObj)
    return jsonList

def make_train_test(pairs, name, is_full):
    print("Making Trainset...")
    label_dict = {}
    train = []
    val = []
    test = []
    i = 0
    for label, items in pairs.items():
        if not is_full and "오류" in label:
            continue
        tmp_train = []
        tmp_val = []
        tmp_test = []
        for item in items:
            jsons = load_json(item)
            a_data, ids = load_data(jsons, item, i)
            tmp_train.append(a_data)
        n_val = math.ceil(len(tmp_train) * 0.1)
        n_test = math.ceil(len(tmp_train) * 0.1)
        for _ in range(n_test):
            tmp_val.append(tmp_train.pop(random.choice(range(len(tmp_train)))))
            tmp_test.append(tmp_train.pop(random.choice(range(len(tmp_train)))))
        
        train += tmp_train
        val += tmp_val        
        test += tmp_test
        print(label, len(items), "->", len(train), len(test))
        label_dict[label] = i
        i += 1
    #with open("./data/dataset/full_dict.pkl", "wb") as f:
     #   pickle.dump(label_dict, f)
    #with open("./data/dataset/"+name+"_train.pkl", "wb") as f:
     #   pickle.dump(train, f)
    #with open("./data/dataset/"+name+"_val.pkl", "wb") as f:
     #   pickle.dump(val, f)
    #with open("./data/"+name+"_test.pkl", "wb") as f:
     #   pickle.dump(test, f)
    with open("./data/dataset/full_dict.pkl", "wb") as f:
        pickle.dump(label_dict, f)
    with open("./data/dataset/"+"_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open("./data/dataset/"+"_val.pkl", "wb") as f:
        pickle.dump(val, f)
    with open("./data/dataset/"+"_test.pkl", "wb") as f:
        pickle.dump(test, f)
    return train, val, test 

parser = argparse.ArgumentParser(description='run scripts.')

parser.add_argument("--folders",nargs="+", type=str)
parser.add_argument("--full", action='store_true', default=False)
args = parser.parse_args()

folders = args.folders
train_all = []
val_all = []
test_all = []
for folder in folders:
    json_path = "data/CrossFit/"+folder+"/"
    json_data_pairs = {}
    for path in os.listdir(json_path):
        tmp_path = os.path.join(json_path, path)
        for path1 in os.listdir(tmp_path):
            tmp_path1 = os.path.join(tmp_path, path1)
            for path2 in os.listdir(tmp_path1):
                tmp_path2 = os.path.join(tmp_path1, path2)
                for path3 in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, path3)
                    for path4 in os.listdir(tmp_path3):
                        tmp_path4 = os.path.join(tmp_path3, path4)
                        for path5 in os.listdir(tmp_path4):
                            tmp_path5 = os.path.join(tmp_path4, path5)
                            label = path3 + "_" + path4 + "_" + path5
                            for path6 in os.listdir(tmp_path5):
                                tmp_path6 = os.path.join(tmp_path5, path6)
                                for path7 in os.listdir(tmp_path6):
                                    tmp_path7 = os.path.join(tmp_path6, path7)
                                    for path8 in os.listdir(tmp_path7):
                                        target = os.path.join(tmp_path7, path8)
                                        print(target)
                                        if not ".json" in target:
                                            continue
                                        if label not in json_data_pairs.keys():
                                            json_data_pairs[label] = [target]
                                        else:
                                            json_data_pairs[label].append(target)
    train, val, test = make_train_test(json_data_pairs, folder, args.full)
    train_all += train
    val_all += val
    test_all += test
if len(folders) > 1:
    with open("./data/dataset/train_f1.pkl", "wb") as f:
        pickle.dump(train_all, f)
    with open("./data/dataset/val_f1.pkl", "wb") as f:
        pickle.dump(val_all, f)

    with open("./data/dataset/test_f1.pkl", "wb") as f:
        pickle.dump(test_all, f)
