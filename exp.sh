nohup python main.py --batch_size 16 --dim 192 --lr 1e-4 --threshold 0.50 --end_epoch 200 &
# python predict.py --load_path checkpoints --predict_mode multiple --threshold 0.50 --dim 192