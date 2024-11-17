model_name='dual_tower_high' # 'dual_tower', 'dual_tower_high', 'dual_tower_low'
save_name='/home/zhangyinan/ore/ore_classification/work_dirs/'$model_name

python ore_classification/train.py --modelName $model_name --saveName $save_name