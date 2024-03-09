# Action-recognition-on-CrossFit-and-Figure-skating-datasets
This project was done while creating CrossFit and Figure skating datasets
# Training 
python tools/train.py pth/figure_normal/figure_normal_config.py --work-dir work_dirs/figure_normal --validate --test-best --gpus 2 --seed 0 --deterministic

1. 새로운 create_data.py
  Alphapose에서 예상 외의 것들이 잡히는 프레임이 소수 발견되었고, 이것으로 인해 버그가 발생하였습니다. 하여 해당 버그를 처리하며 데이터를 만드는 create_data.py를 첨부 드립니다. 
코드를 이용해 기존의 json파일들에서 정상적인 train/test 데이터를 얻을 수 있습니다. 

  create_data.py가 있어야 할 위치
    Alphapose_1cycle 폴더와 같은 레벨에 위치 (반드시 코드를 읽어보고 위치를 맞춰서 넣으시기 바랍니다.) 

  사용법
    create_data.py [--folders folder names...]
      --folders : 폴더 이름들 (날짜 이름을 가진 폴더들입니다.)
    예1) python create_data.py --folders 20220705 20220706 

Json 파일에서도 오류를 제거한 뒤 다시 보내드리겠습니다.  .

2. 위의 create_data.py로 만든 데이터 셋 [Link https://drive.google.com/file/d/1X_LbRM7Li5SY3z1OL5ZOk20qO7VSifpG/view?usp=sharing]
train.pkl / test.pkl: 정상 동작 데이터
label_dict.pkl: Class 이름 dictionary
모델을 돌리기 전 config file에 이용을 할 데이터의 path를 반드시 지정해주셔야 합니다. 

3. pre-trained model path / config 파일
제가 실험에 이용한 config 세팅 및 pre-trained 된 모델을 첨부하여 보내드립니다.
반드시 config 파일을 읽어, 안의 내용이 자신의 환경과 세팅이 맞는지 확인 하고, 다르다면 바꿔서 돌려야 합니다. 

tools/test.py를 이용해서 잘 작동하는지 확인 부탁 드립니다.





아래 구글 드라이브에 각 클래스별로 영상들을 모아 압축하였고,
아래 폴더에서 다운 받으실 수 있습니다.
https://drive.google.com/drive/folders/127-hhaBebuUMqA38OxyMtLCrCzLjyYNA?usp=sharing 
스켈레톤이 렌더링된 영상 예시는 링크 https://drive.google.com/file/d/1wjaW8oeZeVYRyd-34HXQpJPhIukIa5Pk/view?usp=sharing 와 같습니다.




1) 샘플 데이터 수령
이랜서 정재현 부장님께서 잠시 학교를 방문하시어
크로스핏 샘플 데이터와 함께 오류분류체계 문서[[link]( https://drive.google.com/file/d/1dSMwXUPjj5K3wykjQpT334gI_yhOfJtn/view?usp=sharing  )][[link]( https://drive.google.com/file/d/18I6hDHEiCdbDNsGb2CtE3S-MaTOYZ09v/view?usp=sharing  )]를 전달해주셨습니다.
샘플 데이터는 1명이 촬영한 것으로 각 카메라별 약 90개 (총 카메라 8대 = 약 720개)의 영상입니다.
이랜서 측에 따르면, 전체 크로스핏 데이터는 60명이 촬영한 데이터로 구성될 것으로 예상되며,
1 cycle 평가를 위한 데이터는 12명이 촬영한 데이터로 구성될 것으로 예상된다고 합니다.
데이터에 대한 내용은 메일에 담기 복잡하여 구글 문서[[link](https://docs.google.com/document/d/1Z43uG2K3u-K-FpyuNZ2rZGVra3xBJa9eSH704Ma1FL4/edit?usp=sharing )]에 따로 정리하였습니다.
 
현재 alphapose를 활용하여 샘플 데이터로부터 스켈레톤을 추출하고 있습니다.
스쿼트 영상에 대해 스켈레톤이 렌더링된 영상[link https://drive.google.com/file/d/1yOdf2v6z2L6rQUPrRZovGQZkt91RDvFx/view?usp=sharing ]과 스켈레톤 데이터[link https://drive.google.com/file/d/1NTjOcqLnmnbnS3y5w_39qUCHKSLT-cjR/view?usp=sharing ] 예시를 업로드하였습니다.


This is the code that is post-processed to put the result from alphapose mentioned above into poseC3D.

python create_data.py --folders 20220705 20220706 20220707 20220708 20220711 --trainset True --full True You should check all the details inside the code.

Trainset True is a code that creates a csv and also creates a training set, so it has to be True to make a train test set, Full True is True: Make all normal error data.
False: make only normal
It's designed to be written like this.

The data created by Trainset True is for PoseC3D and may be different from the data format desired by eLancers.

The output of alphapose is in the form of a simple json file that does not contain metadata.
First of all, there is a code to create a csv because the data type you want in eLancer can be csv.

Alpha pose result data will be shared as soon as it is completed.
If you share it, it seems that you can do post-processing work with the code given above.

DATA CREATING
python create_data_1v.py --folders 20220830~20220901 20220902 20220903 20220904 20220905 20220906 20220907  20220908 20220909 20220911 20220912 20220914 --full

python create_data_1v.py --folders 20220705 20220706 20220707 20220708 20220711 20220725 20220726 20220727 20220728 20220729 20220801 20220802 20220803 20220804 20220805 20220808 20220809 20220810 20220811 20220812 20220816 20220817 20220818 20220819 20220822

python create_data_1v.py --folders 20220902 20220903 20220904 20220906 20220907 20220908 20220909 20220911 20220912 --full

python create_data_1cut.py --folders 20220705 20220706 20220707 20220708 20220711 20220725 20220726 20220727 20220728 20220729 20220801 20220802 20220803 20220804 20220805 20220808 20220809 20220810 20220811 20220812 20220816 20220817 20220818 20220819 20220822

python create_data_1cut.py --folders 20220830 20220831 20220901 20220902 20220903 20220904 20220906 20220907 20220908 20220909 20220911 20220912



TRAINING CODE RESUME
python tools/train.py configs/skeleton/posec3d/doppler_CF_full.py --work-dir work_dirs/custom_dopp_FC_full --resume-from work-dirs/custom_doppler/latest.pth --validate --test-best --gpus 2 --seed 0 --deterministic

python tools/train.py configs/skeleton/posec3d/elancer_only_normal.py --work-dir work_dirs/custom --validate --test-best --gpus 2 --seed 0 --deterministic

python tools/train.py configs/skeleton/posec3d/elancer_only_normal.py --work-dir work_dirs/custom --validate --test-best --gpus 2 --seed 0 --deterministic

python tools/train.py configs/skeleton/posec3d/elancer_only_normal.py --work-dir work_dirs/custom --validate --test-best --gpus 2 --seed 0 --deterministic


TESTING CODE
python tools/test.py data/crossfit_full/crossfit_full_config.py data/crossfit_full/crossfit_full_200.pth --eval top_k_accuracy mean_class_accuracy  --out result_cross_full_last.pkl

python tools/test.py configs/skeleton/posec3d/elancer_only_normal.py work_dirs/custom/best_top1_acc_epoch_200.pth --eval top_k_accuracy mean_class_accuracy  --out result.pkl

python tools/test.py configs/skeleton/posec3d/elancer_only_normal.py work_dirs/custom/best_top1_acc_epoch_200.pth --eval top_k_accuracy mean_class_accuracy  --out result.pkl

Resume interrupted training from the latest epoch:
python tools/train.py configs/skeleton/posec3d/doppler_figure_norm.py --resume-from work_dirs/custom_doppler/epoch_20(latest).pth --work-dir work_dirs/custom_doppler --validate --test-best --gpus 2 --seed 0 --deterministic

I think I told you that you have "make" the code to create your train/valid/test data. the only reason I gave you the code is to help you understand, not just running code and done. Because you have all json files, you can and have to preprocess the data into the form that you want for your model. 

to do that, of course you will already know, 
you have to check the details of your given data (json files that we sent.): length of json files, number of classes of your data, and so on. and check all of your preprocessed data after you create.

It is not my job to give you all ready-to-go data, which I actually gave you already (python code to create the data, and pkl files) because I wanted to help you. 

 this is one that I gave to your lab
https://drive.google.com/file/d/1PUUrvWkDaSlSQhtnymesGhS6qQatKZEf/view?usp=sharing

but you can't just use this. you HAVE TO PREPROCESS by yourself. because elancer wants to split train, valid, test data which has different from than now. keep in mind it and create your own code.


each folder contains 4 files. 

which are 
pth file for saved weights
config.py file
test pickle file
validation pickle file https://drive.google.com/file/d/11Hp2n4K-u1674ddxvomFlP-0AcEmpwRl/view?usp=share_link

I changed 20220905, 20220914 folders into right one and saved here.
https://drive.google.com/drive/folders/16N--bblvs2vtoU4tcqduxrEk9aTA2WK1?usp=share_link

https://github.com/open-mmlab/mmaction2/blob/master/docs/en/getting_started.md#training-setting
check the site and resume it
there should be way to start learning from last ended epoch

this is the link to download whole json files for figure data.

please use this data with metadata that you got few days ago.
https://drive.google.com/file/d/1mdjZr_dfJEVzy7f1dsHaT9bcLkA5le96/view?usp=share_link

each folder contains 4 files. 

which are 
pth file for saved weights
config.py file
test pickle file
validation pickle file https://drive.google.com/file/d/11Hp2n4K-u1674ddxvomFlP-0AcEmpwRl/view?usp=share_link



