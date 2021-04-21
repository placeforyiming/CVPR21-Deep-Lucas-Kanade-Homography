python3 train_level_one.py --dataset_name=GoogleMap  --epoch_num=20
python3 train_level_two.py --dataset_name=GoogleMap  --epoch_load=20 --epoch_num=20
python3 train_level_three.py --dataset_name=GoogleMap  --epoch_load_one=20 --epoch_load_two=20 --epoch_num=20
python3 train_level_four.py --dataset_name=GoogleMap  --epoch_load_one=20 --epoch_load_two=20 --epoch_load_three=20 --epoch_num=20