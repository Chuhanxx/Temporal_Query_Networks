import torch
import os 
import os.path as osp

def tfm_mask(seg_per_video,temporal_mutliplier=1):
    """
    Attention mask for padded sequence in the Transformer
    True: not allowed to attend to 
    """
    B = len(seg_per_video)
    L = max(seg_per_video) * temporal_mutliplier
    mask = torch.ones(B,L,dtype=torch.bool)
    for ind,l in enumerate(seg_per_video):
        mask[ind,:(l*temporal_mutliplier)] = False

    return mask



def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res



def multihead_acc(preds,clabel,target,vocab,\
    Q=4,return_probs=False):
    """
    Args:
    preds: Predicted logits
    clabel: Class labels,
            List, [batch_size]
    target: Ground Truth attribute labels
            List, [batch_size,num_queries]
    vocab: The mapping between class index and attributes. 
           List, [num_classes,num_queries]
    Q: Number of queries, Int

    Output:
    prob_acc: match predicted attibutes to ground-truth attibutes of N classes,
              class with the highest similarity is the predicted class. 
    """

    # reshape the preds to (B,num_heads,num_classes)
    if len(preds.shape)==2:
        BQ,C = preds.shape
        B = BQ//Q
        preds = preds.view(-1,Q,C)
    elif len(preds.shape)==3:
        B,Q,C = preds.shape

    target = target.view(-1,Q)
    vocab_onehot = one_hot(vocab,C)

    cls_logits =torch.einsum('bhc,ahc->ba', preds, vocab_onehot.cuda())
    cls_pred = torch.argmax(cls_logits,dim=-1)
    prob_acc =  (cls_pred == clabel).sum()*1.0 /B

    if return_probs:
        return prob_acc,cls_logits
    else:
        return prob_acc



def one_hot(indices,depth):
    """
    make one hot vectors from indices
    """
    y = indices.unsqueeze(-1).long()
    y_onehot = torch.zeros(*indices.shape,depth)
    if indices.is_cuda:
        y_onehot = y_onehot.cuda()
    return y_onehot.scatter(-1,y,1)



def make_dirs(args):

    if osp.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)
    args.save_folder = osp.join(args.save_folder ,args.name)
    if osp.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)

    args.tbx_dir =osp.join(args.tbx_folder,args.name)
    if osp.exists(args.tbx_folder) == False:
        os.mkdir(args.tbx_folder)

    if osp.exists(args.tbx_dir) == False:
        os.mkdir(args.tbx_dir)

    result_dir = osp.join(args.tbx_dir,'results')
    if osp.exists(result_dir) == False:
        os.mkdir(result_dir)



def batch_denorm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1):
    """
    De-normalization the images for viusalization
    """
    shape = [1]*tensor.dim(); shape[channel] = 3
    dtype = tensor.dtype 
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).view(shape)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).view(shape)
    output = tensor.mul(std).add(mean)
    return output
