import argparse
import torch
import random
import json
import os
import sys
sys.path.append('../')

from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils.augmentation as A
import os.path as osp

from torch.utils.data import DataLoader
from torchvision import transforms

from models.TQN import *
from utils.utils import make_dirs,multihead_acc,calc_topk_accuracy
from utils.plot_utils import *

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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--root', default='', type=str)
    parser.add_argument('--dataset_config', default='', type=str)
    parser.add_argument('--feature_file', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)

    # device params
    parser.add_argument("--gpus", dest="gpu", default="0", type=str)
    parser.add_argument('--num_workers', default=16, type=int) 

    ## model setting
    parser.add_argument("--model",default='s3d',type=str,help='i3d,s3d')
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--N', default=4, type=int,help='Number of layers in the temporal decoder')
    parser.add_argument('--H', default=4, type=int,help='Number of heads in the temporal decoder')
    parser.add_argument('--K', default=2, type=int,help='Number of clips updated per batch')

    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--pretrained_weights_path', default='', type=str)


    args = parser.parse_args()

    if args.dataset_config is not None:
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    assert args.batch_size == 1

    make_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.FloatTensor')

    ## Set Up Model

    net = TQN(args).cuda()
    net = torch.nn.parallel.DataParallel(net)

    ## Load Model Weights

    resume_file = osp.join(args.save_folder,'best.pth')
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['model_state_dict'],strict=True)

    ## Set Up Dataloader

    transform_test = transforms.Compose([
        A.CenterCrop(size=args.img_dim),
        A.ToTensor(),
        A.Normalize(args.dataset)])
    testset = TQN_dataloader(args,mode='test',
        transform=transform_test,
        SUFB = False)
    test_loader = data.DataLoader(testset, args.batch_size,num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=worker_init_fn, shuffle=False,
                                   collate_fn = SUFB_collate,drop_last=True,sampler = None)


    net.eval()
    test_accuracy = [AverageMeter(),AverageMeter()]

    with torch.no_grad():

        for k, test_samples in tqdm(enumerate(test_loader),total=len(test_loader)):

            v_ids, seqs, cls_targets, att_targets = test_samples

            seqs = seqs[0]
            B, K, C, T, H, W = seqs.shape 
            cls_targets = cls_targets.cuda()
            att_targets = att_targets.view(-1).cuda()

            preds, cls_preds = net((seqs,None))
            preds = torch.softmax(preds, dim=-1).mean(0, keepdim=True)

            cls_preds = torch.softmax(cls_preds, dim=-1).mean(0, keepdim=True)
            match_acc = multihead_acc(preds, cls_targets, att_targets, \
                    testset.class_tokens, Q = args.num_queries)

            cls_acc = calc_topk_accuracy(cls_preds, cls_targets, (1,))[0]
            acc = [torch.stack([cls_acc, match_acc], 0).unsqueeze(0)]

            cls_acc, match_acc = torch.cat(acc, 0).mean(0)

            test_accuracy[0].update(cls_acc.item(), 1)
            test_accuracy[1].update(match_acc.item(), 1)

 
    test_acc = [i.avg for i in test_accuracy]
    print("attribute_match_acc:%.2f"% round(test_acc[1]*100, 2))
    print("class_token_acc:%.2f" % round(test_acc[0]*100, 2))


if __name__ == '__main__':
    main()


