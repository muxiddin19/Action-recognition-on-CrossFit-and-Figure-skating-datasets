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

2. 위의 create_data.py로 만든 데이터 셋 [Link]
train.pkl / test.pkl: 정상 동작 데이터
label_dict.pkl: Class 이름 dictionary
모델을 돌리기 전 config file에 이용을 할 데이터의 path를 반드시 지정해주셔야 합니다. 

3. pre-trained model path / config 파일
제가 실험에 이용한 config 세팅 및 pre-trained 된 모델을 첨부하여 보내드립니다.
반드시 config 파일을 읽어, 안의 내용이 자신의 환경과 세팅이 맞는지 확인 하고, 다르다면 바꿔서 돌려야 합니다. 

tools/test.py를 이용해서 잘 작동하는지 확인 부탁 드립니다.
