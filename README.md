# Action-recognition-on-CrossFit-and-Figure-skating-datasets
This project was done while creating CrossFit and Figure skating datasets
# Training 
python tools/train.py configs/skeleton/posec3d/custom.py --work-dir work_dirs/custom --validate --test-best --gpus 2 --seed 0 --deterministic
