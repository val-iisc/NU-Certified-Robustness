# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log

parser = argparse.ArgumentParser(description='Normal-Uniform Training with similarity regularizer')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=400, type=int, metavar='N',
                    help='batchsize (default: 400)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd_gauss', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--noise_sd_unif', default=0.0, type=float,
                    help="standard deviation of Uniform noise for data augmentation")
parser.add_argument('--sim_reg', default=1, type=int,
                    help="which similarity regularizer is used")
parser.add_argument('--beta', default=0, type=int,
                    help="hyperparam for similarity regularizer")
parser.add_argument('--log_file_name', default=None, type=str,
                    help="Name of log file")
parser.add_argument('--checkpoint_name', default=None, type=str,
                    help="checkpoint name")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers)

    model = get_architecture(args.arch, args.dataset)

    logfilename = os.path.join(args.outdir, args.log_file_name)
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss_l1\ttest acc_l1\ttestloss_l2\ttest acc_l2")

    criterion = CrossEntropyLoss(reduction='none').cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer,epoch, args.noise_sd_l2, args.noise_sd_l1, args.baseline,args.Stability, args.lbd, args.beta)
        test_loss_l1, test_acc_l1, test_loss_l2, test_acc_l2 = test(test_loader, model, criterion, args.noise_sd_l2, args.noise_sd_l1)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss_l1, test_acc_l1, test_loss_l2, test_acc_l2))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, args.checkpoint_name))


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd_gauss: float, noise_sd_unif: float, sim_reg: int, beta: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.cuda()
        targets = targets.cuda()
        
        
        if (sim_reg == 1):
            inputs_gauss = inputs.clone()
        elif (sim_reg == 2):
            inputs_unif = inputs.clone()
        elif (sim_reg == 3):
            inputs_gauss = inputs.clone()
            inputs_unif = inputs.clone()
         

        # augment inputs with noise
        
        for k in range(inputs.shape[0]):  #generating noise from Normal-Unifrom distribution
            noise_gauss=torch.randn_like(inputs[k], device='cuda') * noise_sd_gauss
            noise_unif=(torch.rand_like(inputs[k], device='cuda') - 0.5) * 2 * noise_sd_unif * np.sqrt(3)  
            inputs[k]=inputs[k] + noise_gauss + noise_unif
  
        
        # compute output
        
        if sim_reg == 0: 
            outputs=model(inputs)
            loss = criterion(outputs, targets).mean()
            
            
        elif sim_reg == 1:  # for only Gaussian KL regularizer
            outputs=model(inputs)
            for k in range(inputs_gauss.shape[0]):
                noise_normal=torch.randn_like(inputs_gauss[k], device='cuda') * noise_sd_gauss
                inputs_gauss[k] = inputs_gauss[k] + noise_normal
            output = model(inputs_gauss)
            
            loss = (criterion(outputs,targets) + beta*torch.distributions.kl_divergence(torch.distributions.Categorical(logits=outputs), torch.distributions.Categorical(logits=output))).mean()
        
        elif sim_reg == 2:  # for only unif KL regularizer
            outputs=model(inputs)
            for k in range(inputs_gauss.shape[0]):
                noise_unif=(torch.rand_like(inputs_unif[k], device='cuda') - 0.5) * 2 * noise_sd_unif * np.sqrt(3)
                inputs_unif[k] = inputs_unif[k] + noise_unif
            output = model(inputs_unif)
            
            loss = (criterion(outputs,targets) + beta*torch.distributions.kl_divergence(torch.distributions.Categorical(logits=outputs), torch.distributions.Categorical(logits=output))).mean()
        
        elif sim_reg == 3: # for similarity regularizer
            outputs = model(inputs)
            
            for k in range(inputs_gauss.shape[0]):
                noise_l2 = torch.randn_like(inputs_gauss[k], device='cuda') * noise_sd_gauss
                inputs_gauss[k] = inputs_gauss[k] + noise_l2
            
            for k in range(inputs_unif.shape[0]):
                noise_l1 = (torch.rand_like(inputs_unif[k], device='cuda') - 0.5) * 2 * noise_sd_unif * np.sqrt(3)
                inputs_unif[k] = inputs_unif[k] + noise_l1
            
            output_gauss = model(inputs_gauss)
            output_unif = model(inputs_unif)
            
            loss = ( criterion(outputs,targets) + beta*torch.distributions.kl_divergence(torch.distributions.Categorical(logits=outputs), torch.distributions.Categorical(logits=output_gauss)) + beta*torch.distributions.kl_divergence(torch.distributions.Categorical(logits=outputs), torch.distributions.Categorical(logits=output_unif)) ).mean()


        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd_gauss: float, noise_sd_unif: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_l1 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_l1 = AverageMeter()
    top5_l1 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd_gauss (torch.rand_like(inputs[k], device='cuda') - 0.5) * 2 * noise_sd_unif * np.sqrt(3)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
         

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
