# Action-recognition-on-CrossFit-and-Figure-skating-datasets
This project was done while creating CrossFit and Figure skating datasets
# Training 
python tools/train.py configs/skeleton/posec3d/custom.py --work-dir work_dirs/custom --validate --test-best --gpus 2 --seed 0 --deterministic

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
