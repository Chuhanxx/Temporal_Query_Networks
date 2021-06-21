import argparse
import torch
import random
import json
import os
import sys
sys.path.append('../')

from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils.augmentation as A
import os.path as osp

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

from models.TQN import *
from utils.utils import make_dirs
from utils.plot_utils import *

from engine.engine import train_one_epoch, eval_one_epoch
from data.dataloader import TQN_dataloader,SUFB_collate



def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='initial', type=str)

    ## data setting 
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--img_dim', default=224, type=int)
    parser.add_argument('--clip_len', default=8, type=int, help='number of frames in each video block')
    parser.add_argument('--downsample', default=2, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--root', default='', type=str)
    parser.add_argument('--dataset_config', default='', type=str)
    parser.add_argument('--feature_file', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)

    # device params
    parser.add_argument("--gpus", dest="gpu", default="0", type=str)
    parser.add_argument('--num_workers', default=16, type=int) 

    ## model setting
    parser.add_argument("--model",default='s3d',type=str,help='i3d,s3d')
    parser.add_argument('--resume', default=-1, type=int)
    parser.add_argument('--dropout', default=0.8, type=float)

    parser.add_argument('--N', default=4, type=int,help='Number of layers in the temporal decoder')
    parser.add_argument('--H', default=4, type=int,help='Number of heads in the temporal decoder')
    parser.add_argument('--K', default=2, type=int,help='Number of clips updated per batch')

    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--pretrained_weights_path', default='', type=str)

    ## optim setting
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--optim', default='adam', type=str, help='sgd, adam, adadelta')
    parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--max_epoches', default=1000000, type=int)
    parser.add_argument('--best_acc', default=0, type=float)


    ## frequency setting
    parser.add_argument('--print_iter', default=5, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--save_epoch', default=5, type=int)

    parser.add_argument('--max_iter', default=20000000, type=int)
    parser.add_argument('--lr_steps', default=[10, 20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
    

    args = parser.parse_args()
    if args.dataset_config is not None:
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    make_dirs(args)
    vid2id = cp.load(open(osp.join(args.root,args.feature_file).replace('features','vid2id'),'rb'))

    id2vid = {}
    for vid in vid2id.keys(): 
        id2vid[vid2id[vid]]=vid

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.FloatTensor')

    ## Set Up Model

    net = TQN(args,SUFB=True).cuda()
    net = torch.nn.parallel.DataParallel(net)


    ## Load Model Weights

    if args.resume != -1:
        resume_file = osp.join(args.save_folder,str(args.resume)+'.pth')
        checkpoint = torch.load(resume_file)
        state_dict = checkpoint['model_state_dict']
        net.load_state_dict(state_dict,strict=True)
        net.module.queue = checkpoint['queue']
        args.best_acc = checkpoint['best_acc']
        resume_epoch = checkpoint['epoch']

    elif args.pretrained_weights_path!='':
        checkpoint = torch.load(osp.join(args.pretrained_weights_path))
        net.load_state_dict(checkpoint['model_state_dict'],strict=True)
        net.module.queue= cp.load(open(args.feature_file,'rb'))
        resume_epoch = -1
        print('=== resumed from checkpoint:', args.pretrained_weights_path,'===')


    ## Set Up Dataloader

    transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=True, clip_len=args.clip_len, h_ratio=0.6,p=0.8),
        A.RandomHorizontalFlip(consistent=True, clip_len=args.clip_len),
        A.ColorJitter(brightness=0.4, contrast=0.7, saturation=0.7, hue=0.25, 
                      p=1.0, consistent=False, clip_len=args.clip_len),
        A.ToTensor(),
        A.Normalize(args.dataset)])
    transform_test = transforms.Compose([
        A.CenterCrop(size=args.img_dim),
        A.ToTensor(),
        A.Normalize(args.dataset)])

    trainset = TQN_dataloader(args,
        transform=transform, mode='train',
        SUFB = True)
    testset = TQN_dataloader(args,mode='val',
        transform=transform_test,
        SUFB = True)


    train_loader = data.DataLoader(trainset, args.batch_size,num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=worker_init_fn, shuffle=True,
                                  drop_last=True,collate_fn = SUFB_collate, sampler= None)
    test_loader = data.DataLoader(testset, args.batch_size,num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=worker_init_fn, shuffle=False,
                                   collate_fn = SUFB_collate,drop_last=True,sampler = None)

    ## Set Up Optimizer

    parameters = net.parameters()
    params = []
    print('=> [optimizer] finetune TFM with smaller lr')
    for name, param in net.named_parameters():
        if ('attention' in name or 'decoder' in name) and int(args.resume)<10:
            params.append({'params': param, 'lr':args.lr/10})
        else:
            params.append({'params': param, 'lr':args.lr})

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)      


    ## Set Up Tensorboard

    writer_val = SummaryWriter(logdir=osp.join(args.tbx_dir,'val'))
    writer_train = SummaryWriter(logdir=osp.join(args.tbx_dir, 'train'))

    args.val_plotter = PlotterThread(writer_val)
    args.train_plotter = PlotterThread(writer_train)


    ## Start Training 

    cudnn.benchmark = True
    net.train()

    for epoch in range(args.max_epoches):        
        if epoch <= resume_epoch:
            continue
        adjust_learning_rate(args,optimizer, epoch)
        train_one_epoch(args,epoch,net,optimizer,trainset,train_loader,SUFB=True)

        if epoch % args.eval_epoch == 0:
            eval_one_epoch(args,epoch,net,testset,test_loader,SUFB=True)




def adjust_learning_rate(args,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 """
    epoch = epoch - 1
    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    print('current epoch:',epoch,'lr:',lr)
    if epoch >=10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
            param_group['weight_decay'] = decay 



if __name__ == '__main__':
    main()

