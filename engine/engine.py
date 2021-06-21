import torch
import time
import random
from tqdm import tqdm
import numpy as np
from utils.utils import *
from utils.plot_utils import *
from torch import nn

def train_one_epoch(args,epoch,net,optimizer,trainset,train_loader,SUFB = False):
    np.random.seed(epoch)
    random.seed(epoch)
    net.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = [AverageMeter()]
    accuracy = [AverageMeter(),AverageMeter()]
    criterion = nn.CrossEntropyLoss(reduction='mean') 

    t0 = time.time()   

    for j, batch_samples in enumerate(train_loader):
        data_time.update(time.time() - t0)


        # cls_targets: action class labels 
        # att_targets: attribute labels 
        if not SUFB:
            v_ids, seq, cls_targets, n_clips_per_video, att_targets = batch_samples
            if seq is None:
                continue
            mask = tfm_mask(n_clips_per_video)
            preds,cls_preds = net((seq,mask))
        else:
            # ptrs: clip pointers, where the online sampled clips start
            v_ids, seq, cls_targets, ptrs, att_targets = batch_samples
            preds,cls_preds = net((seq,v_ids,ptrs))

        cls_targets = cls_targets.cuda()
        match_acc = multihead_acc(preds, cls_targets, att_targets, \
            trainset.class_tokens, Q = args.num_queries)

        preds = preds.reshape(-1, args.attribute_set_size)
        att_targets = att_targets.view(-1).cuda()
        cls_acc = calc_topk_accuracy(cls_preds, cls_targets, (1,))[0]

        acc = [torch.stack([cls_acc, match_acc], 0).unsqueeze(0)]
        cls_acc, match_acc = torch.cat(acc, 0).mean(0)

        loss = criterion(preds, att_targets)
        loss += criterion(cls_preds, cls_targets)

        accuracy[0].update(match_acc.item(), args.batch_size)
        accuracy[1].update(cls_acc.item(), args.batch_size)
        losses[0].update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_norm) 
        optimizer.step()

        torch.cuda.empty_cache()
        batch_time.update(time.time() - t0)
        t0 = time.time()

        if j % (args.print_iter) == 0:
            t1 = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss[0].val:.4f} Acc: {acc[0].val:.4f}\t'
              'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
              epoch, j, len(train_loader), 
              loss=losses, acc=accuracy, dt=data_time, bt=batch_time))

            args.train_plotter.add_data('local/loss', losses[0].local_avg, epoch*len(train_loader)+j)
            args.train_plotter.add_data('local/match_acc', accuracy[0].local_avg,epoch*len(train_loader)+j)
            args.train_plotter.add_data('local/cls_acc', accuracy[1].local_avg, epoch*len(train_loader)+j)
            torch.cuda.empty_cache()

    if epoch % args.save_epoch == 0:
        print('Saving state, epoch: %d iter:%d'%(epoch, j))
        save_ckpt(net,optimizer,args.best_acc,epoch,args.save_folder,str(epoch),SUFB)

    save_ckpt(net,optimizer,args.best_acc,epoch,args.save_folder,'latest',SUFB)

    train_acc = [i.avg for i in accuracy]
    args.train_plotter.add_data('global/loss', [i.avg for i in losses], epoch)        
    args.train_plotter.add_data('global/match_acc', accuracy[0].local_avg, epoch)
    args.train_plotter.add_data('global/cls_acc', accuracy[1].local_avg, epoch)




def eval_one_epoch(args,epoch,net,testset,test_loader,SUFB = False):
    net.eval()
    test_accuracy = [AverageMeter(),AverageMeter()]
    np.random.seed(epoch+1)
    random.seed(epoch+1)

    with torch.no_grad():
        for k, batch_samples in tqdm(enumerate(test_loader),total=len(test_loader)):

            # cls_targets: action class labels 
            # att_targets: attribute labels
            if not SUFB:
                v_ids,seq,cls_targets,n_clips_per_video,att_targets = batch_samples
                if seq is None:
                    continue
                mask = tfm_mask(n_clips_per_video)
                preds,cls_preds = net((seq,mask))
            else:
 
                # ptrs: clip pointers, where the online sampled clips start
                v_ids,seq,cls_targets,ptrs,att_targets = batch_samples
                preds,cls_preds = net((seq,v_ids,ptrs))

            cls_targets = cls_targets.cuda()
            match_acc  = multihead_acc(preds,cls_targets, att_targets, \
                testset.class_tokens, Q=args.num_queries)

            preds = preds.reshape(-1,args.attribute_set_size)
            att_targets = att_targets.view(-1).cuda()
            cls_acc = calc_topk_accuracy(cls_preds, cls_targets, (1,))[0]

            acc = [torch.stack([cls_acc, match_acc], 0).unsqueeze(0)]
            cls_acc, match_acc = torch.cat(acc, 0).mean(0)

            test_accuracy[0].update(cls_acc.item(), args.batch_size)
            test_accuracy[1].update(match_acc.item(), args.batch_size)

            torch.cuda.empty_cache()

        test_acc = [i.avg for i in test_accuracy]
        args.val_plotter.add_data('global/cls_acc',test_acc[0], epoch)
        args.val_plotter.add_data('global/match_acc',test_acc[1], epoch)


    if test_acc[1] > args.best_acc:
        args.best_acc = test_acc[1]
        torch.save({'model_state_dict': net.state_dict(),\
            'best_acc':test_acc[1]},\
            args.save_folder + '/'  +  'best.pth')



def save_ckpt(net,optimizer,best_acc,epoch,save_folder,name,SUFB):
    if SUFB:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'queue':net.module.queue,
            'best_acc':best_acc,
            'epoch':epoch},
            save_folder + '/'  +  name+'.pth')

    else:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc':best_acc,
            'epoch':epoch},
            save_folder + '/'  +  name+'.pth')
