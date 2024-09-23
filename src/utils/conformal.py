import numpy as np
import torch
import time
from tqdm import tqdm

def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate(val_loader, model, print_bool, topk=(1,3)):

    device = next(model.parameters()).device
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        topK = AverageMeter('topK')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # coverage_classes = AverageMeter('RAPS coverage')
        # size_classes = AverageMeter('RAPS size')

        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            x = x.to(device)
            target = target.to(device)
            # compute output
            output, S = model(x)
            # measure accuracy and record loss
            prec1, precK = accuracy(output, target, topk=topk)
            cvg, sz = coverage_size(S, target)
            # cvg_classes, sz_classes = class_coverage_size(S, target, classes)

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            topK.update(precK.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])
            # coverage_classes.update(cvg_classes, n=x.shape[0])
            # size_classes.update(sz_classes, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            # if print_bool:
            #     print(f'\rN: {N} | Avg Time: {batch_time.avg:.3f} | Avg Cvg@1: {top1.avg:.3f} | Avg Cvg@K: {topK.avg:.3f} | Avg Cvg@RAPS: {coverage.avg:.3f} | Avg Size@RAPS: {size.avg:.3f}', end='')
            #     print()
    if print_bool:
        print(f'\rN: {N} | Avg Time: {batch_time.avg:.3f} | Avg Cvg@1: {top1.avg:.3f} | Avg Cvg@K: {topK.avg:.3f} | Avg Cvg@RAPS: {coverage.avg:.3f} | Avg Size@RAPS: {size.avg:.3f}', end='')
        print()
        print('') #Endline

    return top1.avg, topK.avg, coverage.avg, size.avg 

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def class_coverage_size(S,targets,classes):
    covered = np.zeros(classes)
    size = np.zeros(classes)
    frequency = np.zeros(classes)
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered[targets[i].item()] += 1
        size[targets[i].item()] = size[targets[i].item()] + S[i].shape[0]
        frequency[targets[i].item()] += 1
    return float(covered)/frequency, size/frequency

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Computes logits and targets from a model and loader
def get_logits_targets(model, loader):
    device = next(model.parameters()).device
    num_classes = len(torch.unique(loader.dataset[:][1]))
    logits = torch.zeros((len(loader.dataset), num_classes)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            batch_logits = model(x.to(device)).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits

