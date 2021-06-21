import os
import argparse
import random
import time
import torch
import torchvision
import sys
sys.path.append('../')
import numpy as np
import _pickle as cp
import os.path as osp
import torch.nn as nn
import utils.augmentation as A
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import json 
import glob

from tqdm import tqdm
from torchvision import transforms
from models.TQN import TQN

from data.dataloader import TQN_dataloader,SUFB_collate





def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='initial', type=str)

    ## data setting 
    parser.add_argument('--dataset', default='gym99', type=str)

    parser.add_argument('--img_dim', default=224, type=int)
    parser.add_argument('--clip_len', default=8, type=int, help='number of frames in each video block')
    parser.add_argument('--downsample', default=2, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resume_file', default='', type=str)
    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--dataset_config', default='', type=str)

    parser.add_argument('--all_frames', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='', type=str)

    # device params
    parser.add_argument("--gpus", dest="gpu", default="0", type=str)
    parser.add_argument('--num_workers', default=16, type=int) 

    ## model setting
    parser.add_argument("--model",default='s3d',type=str,help='')
    parser.add_argument('--resume', default=-1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    ## frequency setting
    parser.add_argument('--eval_epoch', default=5, type=int)
    parser.add_argument('--max_iter', default=20000000, type=int)
    
    
    args = parser.parse_args()
    if args.dataset_config is not None:
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    args.max_length = 1e6
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.FloatTensor')

    ## Set Up Model

    num_classes =int(''.join([s for s in args.dataset if s.isdigit()]))
    net = TQN(args,features_out=True)
    net = torch.nn.DataParallel(net).to(device)

    ## Load Model Weights

    assert args.resume_file!= ''
    checkpoint = torch.load(args.resume_file)
    state_dict = checkpoint['model_state_dict']
    net.load_state_dict(state_dict,strict=False)

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

    trainset = TQN_dataloader(args,transform=transform,mode='train')
    testset = TQN_dataloader(args,transform=transform_test,mode='val')

    for dataset in [trainset,testset]:
        data_loader = data.DataLoader(dataset, args.batch_size,num_workers=args.num_workers,
                collate_fn =SUFB_collate,pin_memory=True, worker_init_fn=worker_init_fn,drop_last=False)


        cudnn.benchmark = True
        net.eval()
        
        with torch.no_grad():
            for k, test_samples in tqdm(enumerate(data_loader),total=len(data_loader)):

                v_id, seq, target, _ = test_samples
                if v_id is None:
                    continue
                B, K, C, T, H, W  =seq.shape # [batch_size, num_clips, num_channels, clip_len, H, W]
                out_pkl = osp.join(args.out_dir,v_id[0]+'.pkl')

                if not osp.exists(osp.join(args.out_dir)):
                    os.mkdir(osp.join(args.out_dir))

                # Clip super long videos to fit it in one/two gpus
                if seq.shape[-3] >600:
                    seq = seq[:,:,int(0.2*K):-int(0.2*K):,:,:]

                # Forward
                feas = net((seq,None))
                feas = feas.squeeze().view(B,-1,feas.shape[-1])

                # Save individual feature files first 
                with open(out_pkl, 'wb') as f:
                    cp.dump(feas.cpu(),f)


    ##  Write All the Feature Files into One File 
    vid_to_id,features_dict = {}, {}
    pkls = glob.glob(osp.join(args.out_dir,'*.pkl'))

    for ind,pkl in enumerate(pkls):
      v_id = osp.basename(pkl).replace('.pkl','')
      vid_to_id[v_id] = ind
      features_dict[ind] = cp.load(open(pkl,'rb'))[0]

    with open(osp.join(args.out_dir,args.dataset+'_all_vid2id.pkl'), 'wb') as f:  
      cp.dump(vid_to_id,f)

    with open(osp.join(args.out_dir,args.dataset+'_all_features.pkl'), 'wb') as f:  
      cp.dump(features_dict,f)

    print('Saved featrues from ',len(features_dict),' video samples.')

if __name__ == '__main__':
    main()

