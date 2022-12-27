import math
import random

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F




def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def synmixup(input11, input21, input12, input22, alpha=1.0):
    with torch.no_grad():
        N = input11.shape[0]
        idx1 = list(range(N))
        random.shuffle(idx1)
        lam = np.random.beta(alpha, alpha, (N))
        lam = (lam>0.5)*lam + (1-lam>0.5)*(1-lam)
        lam = torch.from_numpy(lam).cuda().reshape([N,1,1,1]).float()

        output1 = input11 * lam + input21[idx1] * (1-lam)
        output2 = input12 * lam + input22[idx1] * (1-lam)
        return output1, output2




class MemoryBank:
    def __init__(self, maxlength = 32, dimension = 128, classnum = 200):
        self.maxlength = maxlength 
        self.dimension = dimension
        self.classnum = classnum
        self.ptr = np.zeros([classnum,2]).astype(int)
        self.feature = -torch.ones([classnum, 2, maxlength,dimension]).cuda(non_blocking=True)
        ###<add>
        self.gt = -torch.ones([classnum, 2, maxlength]).cuda(non_blocking=True)
        ###</add>
    def get(self):
        return self.feature

    def get_gt(self):
        return self.gt

    def innitial(self, feature, label, isclean, gt=None):
        N = feature.shape[0]
        print('MemoryBank Innitial: Feature shape = {}'.format(feature.shape))
        index = list(range(N))
        random.shuffle(index)
        index = torch.from_numpy(np.array(index)).int().cuda()
        torch.distributed.broadcast(index, 0)
        index = index.cpu().numpy().astype(int)
        for i in index:
            curlabel = int(label[i])
            curisclean = 1 if isclean[i] else 0
            if self.ptr[curlabel,curisclean] < self.maxlength:
                self.feature[curlabel, curisclean, self.ptr[curlabel,curisclean]] = feature[i]
                if not gt is None:
                    self.gt[curlabel,curisclean,self.ptr[curlabel,curisclean]] = gt[i]
                self.ptr[curlabel,curisclean] += 1
        if not (self.ptr >= self.maxlength).all(): 
            print('Not full in MB')
            for i in range(self.classnum):
                for j in range(2):
                    while self.ptr[i,j]<self.maxlength:
                        lack = min(int(self.maxlength - self.ptr[i,j]),self.ptr[i,j])
                        self.feature[i, j, self.ptr[i,j]:self.ptr[i,j]+lack] = self.feature[i, j, :lack]
                        if not gt is None:
                            self.gt[i, j, self.ptr[i,j]:self.ptr[i,j]+lack] = self.gt[i, j, :lack]
                        self.ptr[i,j] += lack
        self.ptr = np.zeros([self.classnum,2]).astype(int)
        
    
    def update(self, sub_feature, sub_label, isclean, sub_gt=None):
        with torch.no_grad():
            feature = [torch.zeros_like(sub_feature) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(feature, sub_feature)
            feature = torch.cat(feature,0)
            label = [torch.zeros_like(sub_label) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(label, sub_label)
            label = torch.cat(label,0)
            if not sub_gt is None:
                gt = [torch.zeros_like(sub_gt) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gt, sub_gt)
                gt = torch.cat(gt,0)
        curisclean = 1 if isclean else 0
        N = feature.shape[0]
        index = list(range(N))
        #print("{} {}".format(label.shape, feature.shape))
        for i in index:
            curlabel = int(label[i])    
            self.feature[curlabel, curisclean, self.ptr[curlabel,curisclean]] = feature[i]
            self.gt[curlabel, curisclean, self.ptr[curlabel,curisclean]] = gt[i]
            self.ptr[curlabel,curisclean] = (self.ptr[curlabel,curisclean]+1)%self.maxlength



class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=200):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
    


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
