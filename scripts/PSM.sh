# 强制使用CPU模式运行，避免CUDA内存不足问题
export CUDA_VISIBLE_DEVICES=-1

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 32  --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25 --win_size 50
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 32     --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20 --win_size 50


