import os, sys
import pickle
import cv2
from os.path import join,getsize
import tarfile
import json
import pathlib
import shutil
from tqdm import tqdm

class FindEveryPath():
    def __init__(self):
        self.paths = []
    
    def Find(self, path):
        if not os.path.isdir(path) or "System Volume Information" in path:
            return
        
        files = os.listdir(path)
        
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    self.Find(fullpath)
                else:
                    if "." in fullpath:
                        self.paths.append(fullpath)
    
    def FindAll(self, path):
        #if os.path.exists("./lists.pkl"):
            #with open("./lists.pkl", "rb") as f:
                #self.paths = pickle.load(f)
           # return
        self.Find(path)
        with open("./lists.pkl", "wb") as f:
            pickle.dump(self.paths, f)
        return

    def get_paths(self):
        return self.paths

#입력 데이터 위치를 넣어 주세요#######################################################
#_SRC_DIR_PATH = "E:/Posec3d_new/data/elancer/20220705~20220711"
#_SRC_DIR_PATH = "E:/Posec3d_new/data/Fugure_scating"
_SRC_DIR_PATH = "H:/20220819"

_LV2NUM = {
    "고급":1,
    "중급":2,
    "초급":3,
}
_SEX2NUM = {
    "여":0,
    "남":1,
}

def run():
    paths = FindEveryPath()
    paths.FindAll(_SRC_DIR_PATH)
    lists = list(set([path[:path.find("camera")-1] for path in paths.get_paths() if "tar" in path]))
    #print('paths.get_paths() = ', paths.get_paths())
    for item in tqdm(lists):
        main(item)

def main(src_dir_path):
    for (root, dirs, files) in os.walk(src_dir_path):
        if "result_label.save.done" not in files: continue
        camera_num2offset_dict = dict()
        marker_num2data_dict = dict()
        with open(join(root,"result_label.save.done"),"r",encoding="UTF8") as f:
            workspace = f.readline().split("|")[-1]
            camera_data = f.readline()

            offset_list = f.readline().rstrip("\n").split("|")[1:]
            for offset in offset_list:
                camera_num, offset = offset.split("=")
                camera_num2offset_dict[int(camera_num.split("_")[-1])] = int(offset)
            
            input_data = f.readlines()
            personInfo = [data for data in input_data if "personInfo" in data]
            if len(personInfo) == 0:
                print(root, ": No person Info. skipping...")
                continue
            marker = [data for data in input_data if "marker" in data]
            personInfo_data = [data for data in input_data if "personInfo" in data][0]
            marker = [data for data in input_data if "marker" in data]
            _,_, age, _, _,sex,level,cm,name = personInfo_data.split("|")
            name = name.split("=")[-1]
            age = age.split("=")[-1]
            sex = sex.split("=")[-1]
            sex = _SEX2NUM[sex]
            cm = cm.split("=")[-1]
            level = level.split("=")[-1]
            level = _LV2NUM[level]
            _, num_marker = marker[0].rstrip("\n").split("=")
            for i in range(int(num_marker)):
                _, marker_num, s_frame, e_frame, motion_name = marker[i+1].rstrip("\n") .split("|")
                s_frame = float(s_frame)
                e_frame = float(e_frame)
                marker_num2data_dict[int(marker_num)] = (s_frame,e_frame, motion_name)
            for camera_num in sorted(camera_num2offset_dict):
                offset = float(camera_num2offset_dict[camera_num])
                root_path = join(root,"camera"+str(camera_num))
                color_dir_path = join(root_path,"color")
                if not os.path.exists(color_dir_path):
                    print(root, "Error! There is no tar file")
                    continue
                tar_f_list = os.listdir(color_dir_path)
                tar_f_name = None
                for f_name in tar_f_list:
                    if f_name.endswith(".tar"):
                        tar_f_name = f_name
                if tar_f_name is None:
                    print(root, "Error! There is no tar file")
                    continue
                tar_file_path = join(color_dir_path,tar_f_name)

                f_name2img_dict =  dict()
                size = (1920, 1080)

                # tar 파일 이름을 비디오 이름으로 변경
                videoname = tar_f_name.split(".")[0]
                videofileNm = videoname + ".avi"
                dst_file_name = join(videofileNm)
                dst_dir_path = join(root_path, "video")
                if not os.path.exists(dst_dir_path):
                    os.mkdir(dst_dir_path)
                dst_file_path = join(dst_dir_path,dst_file_name)

                num_frame = 0
                for img_num in sorted(f_name2img_dict):
                    num_frame += 1

                avi_json_dict = dict()

                dir_name_tuple = pathlib.Path(dst_dir_path).parts[-10:]
                category_1,category_2,category_3,category_4 = dir_name_tuple[-9:-5]
                avi_json_dict["video_path"] = "/".join(dir_name_tuple) + "/" + videofileNm
                avi_json_dict["video_name"] = videofileNm
                avi_json_dict["video_duration"] = round(num_frame / 30, 2)
                avi_json_dict["video_type"] = "avi"
                avi_json_dict["annotations"] = list()
                num_marker_total = len(marker_num2data_dict)
                s_frame_prev = None
                e_frame_prev = None
                marker_num_dumped = 0
                for marker_num in sorted(marker_num2data_dict):
                    s_frame, e_frame, motion_name = marker_num2data_dict[marker_num]
                    if -offset < s_frame:
                        s_frame += offset
                        e_frame += offset
                    if (s_frame< 0) or (e_frame<0):
                        print(root, "Error! There is no Marker")
                        continue
                    if (num_marker_total != 1) and (marker_num != 0):
                        s_frame_interm = (e_frame_prev+1)
                        e_frame_interm = (s_frame - 1)
                        dict_tp = dict()
                        dict_tp["annotation_no"] = int(marker_num_dumped)
                        dict_tp["start_time"] = s_frame_interm / 30
                        dict_tp["end_time"] = e_frame_interm / 30
                        dict_tp["start_frame"] = int(s_frame_interm)
                        dict_tp["end_frame"] = int(e_frame_interm)
                        dict_tp["motion_category1"] = category_1
                        dict_tp["motion_category2"] = category_2
                        dict_tp["motion_category3"] = category_3
                        dict_tp["motion_category4"] = "기타"
                        avi_json_dict["annotations"].append(dict_tp)
                        marker_num_dumped += 1
                    # 준비동작, 마무리동작 현재상태에서는 필요없다고 하여 주석처리 22.09.29
                    # if marker_num == 0:
                    #     dict_tp = dict()
                    #     s_frame_init = max(0.0, s_frame-50)
                    #     e_frame_init = max(0.0,s_frame - 1)
                    #     if e_frame_init != 0:
                    #         dict_tp["annotation_no"] = int(0)
                    #         dict_tp["start_time"] = s_frame_init/30
                    #         dict_tp["end_time"] = e_frame_init / 30
                    #         dict_tp["start_frame"] = int(s_frame_init)
                    #         dict_tp["end_frame"] = int(e_frame_init)
                    #         dict_tp["motion_category1"] = category_1
                    #         dict_tp["motion_category2"] = category_2
                    #         dict_tp["motion_category3"] = category_3
                    #         dict_tp["motion_category4"] = "준비동작"
                    #         avi_json_dict["annotations"].append(dict_tp)
                    #         marker_num_dumped += 1
                    dict_tp = dict()
                    dict_tp["annotation_no"] = int(marker_num_dumped)
                    dict_tp["start_time"] = s_frame/30
                    dict_tp["end_time"] = e_frame/30
                    dict_tp["start_frame"] = int(s_frame)
                    dict_tp["end_frame"] = int(e_frame)
                    dict_tp["motion_category1"] = category_1
                    dict_tp["motion_category2"] = category_2
                    dict_tp["motion_category3"] = category_3
                    dict_tp["motion_category4"] = motion_name

                    avi_json_dict["annotations"].append(dict_tp)
                    marker_num_dumped += 1
                    s_frame_prev = s_frame
                    e_frame_prev = e_frame
                # dict_tp = dict()
                # s_frame_end = min(float(num_frame), e_frame + 1)
                # e_frame_end = min(float(num_frame), e_frame + 50)
                # dict_tp["annotation_no"] = int(marker_num_dumped)
                # dict_tp["start_time"] = s_frame_end/30
                # dict_tp["end_time"] = e_frame_end / 30
                # dict_tp["start_frame"] = int(s_frame_end)
                # dict_tp["end_frame"] = int(e_frame_end)
                # dict_tp["motion_category1"] = category_1
                # dict_tp["motion_category2"] = category_2
                # dict_tp["motion_category3"] = category_3
                # dict_tp["motion_category4"] = "마무리동작"
                # avi_json_dict["annotations"].append(dict_tp)

                avi_json_dict["actor"] = {
                    "actor_level": level,
                    "actor_height": float(cm),
                    "actor_gender": sex,
                    "actor_age": int(age)
                }

                # 체커보드 0이미지 복사
                # shutil.copy(checker_board_origin_img[camera_num], calibration_dir_path)
                with open(join(dst_dir_path, "annotation.json"), "w", encoding="UTF-8") as json_file:
                    json.dump(avi_json_dict, json_file, ensure_ascii=False)

                #print(root_path, "is dumpped!")


if __name__ == "__main__":
    run()
