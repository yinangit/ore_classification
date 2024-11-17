import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models import create_model
from utils.config import Logger
from data import Oredataset_vis
import torch
from utils.WarmUpLR import WarmupLR
import numpy as np
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import cv2
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image



# set seed
GLOBAL_SEED = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_arguments():
    parser = argparse.ArgumentParser(description='Ore classification.')
    parser.add_argument("--Adiscription", type=str, default='Ore classification')
    parser.add_argument("--saveName", type=str, default='/home/zhangyinan/ore/data/vis_results')
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--init_lr", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--nclass", type=int, default=2)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--num_warmup",type=int,default=5)
    parser.add_argument("--in_dim",type=int,default=1)
    parser.add_argument("--lr_way",type=str,default='cos')
    parser.add_argument("--log_interval",type=int,default=100) # iteration
    parser.add_argument("--eval_interval",type=int,default=1) # epoch        
    parser.add_argument("--json_path",type=str,default='/home/zhangyinan/ore/data/dataset/train_val.json')
    parser.add_argument("--modelName",type=str,default='dual_tower_vis')
    parser.add_argument("--pretrained",type=str,default='/home/zhangyinan/ore/ore_classification/pretrained/model.pt')
    parser.add_argument("--model_weight",type=str,default='/home/zhangyinan/ore/ore_classification/work_dirs/try/model_47.pt')
    parser.add_argument("--hid_dim",type=int,default=128)

    return parser.parse_args()

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def vis_grad_cam(model, data_loaderTest):
    cam_low = GradCAM(model=model, target_layers=[model.model_low.layer1[-3]])
    cam_high = GradCAM(model=model, target_layers=[model.model_high.layer1[-3]])
    for data in data_loaderTest:
        input_low, input_high, labels, data_p = data
        input_low, input_high, labels = input_low.to(device), input_high.to(device), labels.to(device)

        input_tensor = torch.cat((input_low, input_high), dim=1)

        grayscale_cam_high = cam_high(input_tensor=input_tensor, targets=None) # 选定目标类别，如果不设置，targets = None, 则默认为分数最高的那一类
        grayscale_cam_high = grayscale_cam_high[0,:]
        grayscale_cam_low = cam_low(input_tensor=input_tensor, targets=None)
        grayscale_cam_low = grayscale_cam_low[0,:]

        img_low = cv2.imread(os.path.join(data_p[0], 'low.tiff'))
        img_low = cv2.resize(img_low, (30, 30))
        img_low = np.float32(img_low) / 255
        img_high = cv2.imread(os.path.join(data_p[0], 'high.tiff'))
        img_high = cv2.resize(img_high, (30, 30))
        img_high = np.float32(img_high) / 255

        cam_image_low = show_cam_on_image(img_low, grayscale_cam_low, use_rgb=True)
        cam_image_high = show_cam_on_image(img_high, grayscale_cam_high, use_rgb=True)

        now_save_path = data_p[0].replace('dataset', 'vis_results')
        if not os.path.exists(now_save_path):
            os.makedirs(now_save_path)
        print(now_save_path)

        cv2.imwrite(os.path.join(now_save_path, 'low.jpg'), cam_image_low)
        cv2.imwrite(os.path.join(now_save_path, 'high.jpg'), cam_image_high)



if __name__ == '__main__':
    args = get_arguments()

    worker_init_fn(args.seed)

    #测试数据和模型的保持位置
    saveName = args.saveName
    args.savePath = os.path.join('log', saveName)
    #日志保存位置
    train_log_path = os.path.join(args.savePath, 'train.log')

    #GPU
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # batch run
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu") # single run

    save_dir = args.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    Logger(train_log_path)
    
    print(args)

    model = create_model(args)

    print(model)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_weight))

    datasetTrain = Oredataset_vis(json_path=args.json_path, TrainValTest = 'train', transform=None) # train
    data_loaderTrain = DataLoader(datasetTrain, batch_size=args.batch_size,shuffle=True,num_workers=1)
    
    datasetVal = Oredataset_vis(json_path=args.json_path, TrainValTest = 'val', transform=None)
    data_loaderVal = DataLoader(datasetVal, batch_size=args.batch_size,shuffle=False,num_workers=1)


    vis_grad_cam(model, data_loaderVal)
