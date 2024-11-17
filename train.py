import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
from models import create_model
from utils.config import Logger
from data import Oredataset
import torch
from utils.WarmUpLR import WarmupLR
import numpy as np
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


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
    parser.add_argument("--saveName", type=str, default='/home/zhangyinan/ore/ore_classification/work_dirs/try')
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--init_lr", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=32)
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
    parser.add_argument("--modelName",type=str,default='dual_tower', choices=['dual_tower', 'dual_tower_high', 'dual_tower_low'])
    parser.add_argument("--pretrained",type=str,default='/home/zhangyinan/ore/ore_classification/pretrained/model.pt')
    parser.add_argument("--hid_dim",type=int,default=128)

    return parser.parse_args()

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(network, data_loaderTest):
    network.eval()

    y_true = list()
    y_pred = list()

    with torch.no_grad():
        for data in data_loaderTest:
            input_low, input_high, labels = data
            input_low, input_high, labels = input_low.to(device), input_high.to(device), labels.to(device)

            outputs = network(input_low, input_high)
            
            outputs = nn.Softmax(dim=1)(outputs)
            preds = torch.argmax(outputs, 1)

            y_true.extend(list(labels.cpu().numpy()))
            y_pred.extend(list(preds.cpu().numpy()))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算宏平均指标
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # 'None'：返回每个类别的精确率。在这种情况下，precision_score 函数会为每个类别分别计算精确率，并返回一个包含这些精确率的数组。
    # 'binary'：仅报告在被视为二分类问题时的精确率。这通常用于二分类问题，其中目标变量 y_true 和 y_pred 只有两个可能的类别。
    # 'micro'：通过计算总的真阳性、假阳性和假阴性的数量来全局计算精确率。这种计算方式不考虑类别的不平衡问题。
    # 'macro'：为每个类别计算精确率，然后取它们的未加权平均值。这种计算方式不考虑类别的不平衡问题。
    # 'weighted'：为每个类别计算精确率，然后取它们的加权平均值，权重是每个类别的实例数。这种计算方式考虑了类别的不平衡问题。
    # 'samples'：在多标签分类中，这将为每个样本计算精确率，然后取这些精确率的平均值。

    # 计算Top-1准确率
    correct = y_pred == y_true
    accuracy = np.mean(correct)

    # 计算混淆矩阵
    confusion_matrix1 = confusion_matrix(y_true, y_pred)

    # 计算AUC
    auc = roc_auc_score(y_true, y_pred)

    return auc, accuracy, precision, recall, f1, confusion_matrix1

def train(network: nn.Module, dataloader, dataloader_test, args):
    
    # Create Optimizer
    optimizer = optim.Adam(network.parameters(), lr = args.base_lr)
    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_epochs)
    schedulers = WarmupLR(scheduler_steplr, init_lr=args.init_lr, num_warmup=args.num_warmup, warmup_strategy=args.lr_way)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(torch.tensor([1.0,3.49]).cuda())
    # criterion = torch.nn.CrossEntropyLoss(torch.tensor([1.0,2.0]).cuda())

    best_f1 = -1

    # Train model on the dataset
    for epoch in range(args.num_epochs):

        schedulers.step()

        print('-' * 10)
        print('Train Epoch %d/%d' % (epoch, args.num_epochs - 1))
        network.train(mode=True)

        for i, data in enumerate(dataloader):
            input_low, input_high, labels = data
            input_low, input_high, labels = input_low.to(device), input_high.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = network(input_low, input_high)
            
            loss = criterion(outputs, labels)
                
            loss.backward()

            optimizer.step()

            if bool(i % args.log_interval) is False:
                print(f"Epoch {epoch}/{args.num_epochs-1}, Batch {i}/{(len(dataloader.dataset) - 1) // dataloader.batch_size + 1},Batch Loss={loss.item():.4f}")

        if bool(epoch % args.eval_interval) is False:
            auc, accuracy, precision, recall, f1, confusion_matrix1 = test(network,dataloader_test) # val
            print(f"Epoch {epoch}/{args.num_epochs-1}, AUC={auc:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            print(confusion_matrix1)
            save_path = os.path.join(save_dir, 'model_{}.pt'.format(epoch))

            if f1 >= best_f1:
                best_f1 = f1
                state_dict = network.state_dict()
                os.system('rm ' + save_dir + '/*.pt')
                torch.save(state_dict, save_path)

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
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable pytorch params:{} MB'.format(trainable_pytorch_total_params*1e-6))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total pytorch params:{} MB'.format(pytorch_total_params*1e-6))

    datasetTrain = Oredataset(json_path=args.json_path, TrainValTest = 'train', transform=None) # train
    data_loaderTrain = DataLoader(datasetTrain, batch_size=args.batch_size,shuffle=True,num_workers=4)
    
    datasetVal = Oredataset(json_path=args.json_path, TrainValTest = 'val', transform=None)
    data_loaderVal = DataLoader(datasetVal, batch_size=args.batch_size,shuffle=False,num_workers=4)

    train(model, data_loaderTrain, data_loaderVal, args)
