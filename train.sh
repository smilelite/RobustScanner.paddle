# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r31_robustscanner.yml
