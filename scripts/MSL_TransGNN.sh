export CUDA_VISIBLE_DEVICES=0

# 使用AnomalyTransGNN模型训练MSL数据集
python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55 --model_type transgnn --gnn_type gcn --gnn_hidden 32 --fusion_method weighted --k_neighbors 5

# 使用AnomalyTransGNN模型测试MSL数据集
python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 256 --mode test --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55 --model_type transgnn --gnn_type gcn --gnn_hidden 32 --fusion_method weighted --k_neighbors 5 --pretrained_model 20