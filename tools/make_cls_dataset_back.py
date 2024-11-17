import cv2
import numpy as np
import os
import json
import random

# 8:2
if __name__ == "__main__":
    
    sub_folder = ['ore', 'waste']
    dataset_path = '/home/zhangyinan/ore/data/dataset'
    save_path = '/home/zhangyinan/ore/data/dataset/train_val.json'

    json_data = {}
    json_data['train'] = []
    json_data['val'] = []
    with open(save_path, 'w') as f:
        for folder in sub_folder:
            samples_list = os.listdir(os.path.join(dataset_path, folder))
            random.shuffle(samples_list)
            new_samples_list = [os.path.join(dataset_path, folder, s) for s in samples_list]
            json_data['train'].extend(new_samples_list[:int(len(new_samples_list)*0.8)])
            print(folder)
            print(len(new_samples_list[:int(len(new_samples_list)*0.8)]))
            json_data['val'].extend(new_samples_list[int(len(new_samples_list)*0.8):])
            print(len(new_samples_list[int(len(new_samples_list)*0.8):]))
        
        json.dump(json_data, f, indent=4)