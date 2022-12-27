# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import shutil
import time
import builtins
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
#from sklearn.mixture import GaussianMixture
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as torchmp
import torch.distributed as dist

from ops import dataset_config
from ops.dataset_scl import TSNDataSet
from ops.models_scl_ddp_bn_ST4 import TSN

from ops.temporal_shift import make_temporal_pool
from ops.transforms import *
from ops.utils import AverageMeter, MemoryBank, accuracy, SequentialDistributedSampler, distributed_concat
from opts import parser
from tools.select_by_clean_set import select_feature_var_no_clean, cos_selection_noclean, innitail_split


best_prec1 = 0


def main():
    

    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True  
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torchmp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    global best_prec1
    args.gpu = gpu

    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        check_rootfolders(args)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    input_size = model.scale_size if args.eval_full_res else model.input_size # for eval
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)



    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + args.world_size - 1) / args.world_size)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)


        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.eval_test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(input_size),
        ])
    elif args.eval_test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, int(scale_size), flip=False)
        ])
    elif args.eval_test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, int(scale_size), flip=False)
        ])
    elif args.eval_test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, int(scale_size))
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.eval_test_crops))
    
    train_dataset_clean = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, noise_type=args.noise_type, noise_rate=args.noise_rate)

    train_dataset_noisy = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample, noise_type=args.noise_type, noise_rate=args.noise_rate)
    
    test_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample)

    eval_dataset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                       new_length=1 if args.modality == "RGB" else 5,
                       modality=args.modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(input_mean, input_std),
                       ]), dense_sample=args.eval_dense_sample, twice_sample=args.eval_twice_sample,noise_type=args.noise_type, noise_rate=args.noise_rate)

    if args.distributed:
        train_sampler_clean = torch.utils.data.distributed.DistributedSampler(train_dataset_clean) 
        train_sampler_noisy = torch.utils.data.distributed.DistributedSampler(train_dataset_noisy) 
        test_sampler = None
        eval_sampler = SequentialDistributedSampler(eval_dataset, args.batch_size)
    else:
        train_sampler_clean = None
        train_sampler_noisy = None
        test_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset_clean,
        batch_size=args.batch_size, shuffle=(train_sampler_clean is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_clean,
        drop_last=True)  # prevent something not % n_GPU

    noisy_train_loader = torch.utils.data.DataLoader(
        train_dataset_noisy,
        batch_size=args.batch_size, shuffle=(train_sampler_noisy is None),
        num_workers=args.workers, pin_memory=True,sampler=train_sampler_noisy,
        drop_last=True) 

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size*10, shuffle=False,sampler=test_sampler,
        num_workers=args.workers, pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size, shuffle=(eval_sampler is None), sampler=eval_sampler,
            num_workers=args.workers, pin_memory=True, drop_last = False
    )



    actually_clean = np.zeros([50, num_class]) 
    believe_clean = np.zeros_like(actually_clean)
    

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(test_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log_{}.csv'.format(args.rank)), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))


    print('world size in torch.distributed {}'.format(torch.distributed.get_world_size()))
    print('Rank in dist {}'.format(torch.distributed.get_rank()))
    print('eval_sample len {}'.format(len(eval_sampler.dataset)))

    isclean = None
    MB = MemoryBank(maxlength = 8, dimension = 128, classnum = num_class)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler_clean.set_epoch(epoch)
            train_sampler_noisy.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer, noisy_train_loader, MB, args=args)
        
        log_training.write('train end' + '\n')
        log_training.flush()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log_training.write('validate start' + '\n')
            log_training.flush()
            prec1 = 0
            #if (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            prec1 = validate(test_loader, model, criterion, epoch,args, log_training, tf_writer)
            log_training.write('validate end' + '\n')
            log_training.flush()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            #if (args.multiprocessing_distributed):
            log_training.write(output_best + '\n')
            log_training.flush()
            if ( epoch +1 ) % 5 == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0)):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, args)
            
            if epoch + 1 > args.eval_start_epoch:  
                feature_record_sub, gt_record_sub, low_dim_record_sub = eval(eval_loader, model, args, log=log_training)
                log_training.write('eval end' + '\n')
                log_training.flush()

                feature_record = distributed_concat(feature_record_sub, len(eval_sampler.dataset))
                gt_record = distributed_concat(gt_record_sub, len(eval_sampler.dataset))       
                low_dim_record = distributed_concat(low_dim_record_sub, len(eval_sampler.dataset))     
                        
                isclean, prob = split_dataset(feature_record, gt_record, epoch, actually_clean, believe_clean, log_training, isclean, MB, args)
                isclean = torch.from_numpy(isclean).float().cuda(args.gpu, non_blocking=True)
                
                prob = torch.from_numpy(prob).float().cuda(args.gpu, non_blocking=True)
                log_training.write('split end' + '\n')
                log_training.flush()
                torch.distributed.barrier() 
                torch.distributed.broadcast(isclean, 0)
                torch.distributed.broadcast(prob, 0)
                MB.innitial(low_dim_record, gt_record[:,1], isclean, gt_record[:,0])
                del feature_record
                
                if epoch < args.epochs - 1:
                    isclean = isclean.data.cpu().numpy().astype(bool)
                    prob = prob.data.cpu().numpy()
                    split_time = time.time()
                    train_dataset_clean.adjust_split(isclean, prob)
                    train_dataset_noisy.adjust_split(~isclean, prob)
                    # ddp need a new sampler here 
                    train_sampler_clean = torch.utils.data.distributed.DistributedSampler(train_dataset_clean) 
                    train_sampler_noisy = torch.utils.data.distributed.DistributedSampler(train_dataset_noisy)

                    train_loader = torch.utils.data.DataLoader(
                        train_dataset_clean,
                        batch_size=args.batch_size, shuffle=(train_sampler_clean is None),
                        num_workers=args.workers, pin_memory=True, sampler=train_sampler_clean,
                        drop_last=True)  # prevent something not % n_GPU

                    noisy_train_loader = torch.utils.data.DataLoader(
                        train_dataset_noisy,
                        batch_size=args.batch_size, shuffle=(train_sampler_noisy is None),
                        num_workers=args.workers, pin_memory=True,sampler=train_sampler_noisy,
                        drop_last=True) 
                    print('change dataset cost {}'.format(time.time()-split_time))




    np.save("./log/actually_clean.npy",actually_clean)
    np.save("./log/believe_clean.npy", believe_clean)

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, noisy_train_loader, MB, args):  # MB memory bank
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    knn_right = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    print(args)
    # switch to train mode
    model.train()

    end = time.time()
    noisy_iter =  iter(noisy_train_loader)
    sample_seed = epoch

    for i, (input, target,clean_gt,_,prob) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        input_1 = input.reshape([-1,16,3,224,224])[:,:8].reshape([-1,24,224,224]).cuda(args.gpu, non_blocking=True)
        
        input_2 = input.reshape([-1,16,3,224,224])[:,8:].reshape([-1,24,224,224]).cuda(args.gpu, non_blocking=True)
        prob = prob.cuda(args.gpu, non_blocking=True)
        clean_gt = clean_gt.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        input_var_1 = torch.autograd.Variable(input_1)
        input_var_2 = torch.autograd.Variable(input_2)
        target_var = torch.autograd.Variable(target)


        # compute output and celoss
        output_1,_,low_dim_feature_1 = model(input_var_1)
        output_2,_,low_dim_feature_2 = model(input_var_2)
        loss = (criterion(output_1, target_var) + criterion(output_2, target_var))/2
        if True:
            fakeloss = low_dim_feature_1.mean() + low_dim_feature_2.mean() - low_dim_feature_1.mean() - low_dim_feature_2.mean() # always = 0, ddp bug, fuck
            loss = loss + fakeloss * 0 
        # contrastive loss

        if epoch > args.eval_start_epoch and len(noisy_iter) > 0:
        
            try:
                noisy_input, noisy_target,noisy_gt,_,noisy_prob = noisy_iter.next()
            except:
                sample_seed += 50
                noisy_train_loader.sampler.set_epoch(sample_seed)
                noisy_iter =  iter(noisy_train_loader)
                noisy_input, noisy_target,noisy_gt,_,noisy_prob = noisy_iter.next()

                

            noisy_input_1 = noisy_input.reshape([-1,16,3,224,224])[:,:8].reshape([-1,24,224,224]).cuda(args.gpu, non_blocking=True)
            noisy_input_2 = noisy_input.reshape([-1,16,3,224,224])[:,8:].reshape([-1,24,224,224]).cuda(args.gpu, non_blocking=True)
            noisy_prob = noisy_prob.cuda(args.gpu)
            noisy_target = noisy_target.cuda(args.gpu)
            noisy_gt = noisy_gt.cuda(args.gpu, non_blocking=True)
            noisy_input_var_1 = torch.autograd.Variable(noisy_input_1)
            noisy_input_var_2 = torch.autograd.Variable(noisy_input_2)
            noisy_target_var = torch.autograd.Variable(noisy_target)
            _,_,noisy_low_dim_feature_1 = model(noisy_input_var_1, stop_grad = True)
            _,_,noisy_low_dim_feature_2 = model(noisy_input_var_2, stop_grad = True)



            
            MB_feature= MB.get()
            MB_gt = MB.get_gt().reshape(-1)
            MB_feature = MB_feature.reshape(-1, MB.dimension)
            MB_target = torch.arange(MB.classnum).reshape([MB.classnum,1]).repeat(1,2*MB.maxlength).reshape(-1).cuda()

            
            l1 = low_dim_feature_1.shape[0]
            l2 = MB_feature.shape[0]
            assert l2 == MB_target.shape[0], '{} {}'.format(l2, MB_target.shape[0])
            batch_low_dim = torch.cat((low_dim_feature_1, low_dim_feature_2, noisy_low_dim_feature_1, noisy_low_dim_feature_2), 0) # shape 4*l1 by 128
            batch_target = torch.cat((target_var, target_var, noisy_target_var, noisy_target_var)) #shape 4*l1
            dict_low_dim = MB_feature # shape: 4*l2 by 128
            dict_target = MB_target #shape 4*l2
            
            # batch
            #stable_mask = (torch.abs(dict_prob - 0.5)>0.2).float().reshape(1,-1)

            same_class_mask = (batch_target.reshape([-1,1]) == dict_target.reshape([1,-1])).float().cuda() #4*l2 by l2
            batch_same_class_mask = torch.ones(4*l1,1).cuda() #4*l2 by 1
            same_class_mask = torch.cat([batch_same_class_mask, same_class_mask], 1) 
        

            clean_sample_mask = torch.ones(MB.classnum,2,MB.maxlength).cuda()
            clean_sample_mask[:,0] = 0
            clean_sample_mask = clean_sample_mask.reshape([1,-1]).repeat(l1*4,1) #4*l1 by l2
            batch_clean_sample_mask = torch.ones(4*l1,1).cuda()
            batch_clean_sample_mask[2*l1:] = 0
            clean_sample_mask = torch.cat([batch_clean_sample_mask, clean_sample_mask], 1)

            
            
            pairwise_mult = torch.matmul(batch_low_dim, dict_low_dim.t().contiguous()) # 4*l1 by l2
            batch_mult = torch.sum(batch_low_dim * torch.cat((low_dim_feature_2, low_dim_feature_1, noisy_low_dim_feature_2, noisy_low_dim_feature_1), 0), dim=1, keepdim=True) # 4*l1 by 1
            pairwise_mult = torch.cat((batch_mult, pairwise_mult), 1) # 4*l1 by l2 + 1
            pairwise_mult_mask = pairwise_mult[2*l1:].clone().detach()
            pairwise_mult_mask[:,0] = -1

            
            noisy_pos_index = torch.argsort(pairwise_mult_mask,dim=1)[:,-int(2*MB.maxlength):]
            ###<change>
            #noisy_pos_index = torch.argsort(pairwise_mult_mask,dim=1)[:,-8:]
            BP_gt = MB_gt[noisy_pos_index-1]
            noisy_gt = torch.cat((noisy_gt,noisy_gt),0)
            TP = (noisy_gt.reshape([-1,1]) == BP_gt).sum(dim=1).float().mean() 
            knn_right.update(TP.item(), input.size(0))
            ###</change>
            noisy_pos_mask = torch.zeros([2*l1,l2+1]).cuda()
            noisy_pos_mask[:,0] = 1
            for j in range(noisy_pos_mask.shape[0]):
                noisy_pos_mask[j][noisy_pos_index[j]] = 1
            logits = torch.div(pairwise_mult, 0.1) 
            exp_logits = torch.exp(logits) 

            
            #clean - batch
            log_prob_clean = torch.log(torch.exp(logits[:2*l1]) + 1e-10) - torch.log((exp_logits[:2*l1]*(1-(1-same_class_mask[:2*l1])*(1-clean_sample_mask[:2*l1]))).sum(1, keepdim=True) + 1e-10) # 2 * 4
            log_prob_clean = log_prob_clean *  clean_sample_mask[:2*l1]* same_class_mask[:2*l1] #2*4
            log_prob_clean = torch.sum(log_prob_clean / torch.sum( clean_sample_mask[:2*l1] * same_class_mask[:2*l1], dim=1, keepdim=True),dim=1)
            contrastiveloss = -log_prob_clean.mean() 

            # noisy - batch
            log_prob_noisy =  torch.log(torch.exp(logits[2*l1:]) + 1e-10) - torch.log((exp_logits[2*l1:]).sum(1, keepdim=True) + 1e-10) # 2 * 4
            log_prob_noisy = log_prob_noisy * noisy_pos_mask #2*4
            log_prob_noisy = torch.sum(log_prob_noisy / torch.sum( noisy_pos_mask, dim=1, keepdim=True),dim=1)
            contrastiveloss = contrastiveloss - log_prob_noisy.mean()

            MB.update(low_dim_feature_1, target, True, clean_gt)

            MB.update(low_dim_feature_2, target, True, clean_gt)
            MB.update(noisy_low_dim_feature_1, noisy_target, False, noisy_gt)
            MB.update(noisy_low_dim_feature_2, noisy_target, False, noisy_gt)



        else:
            contrastiveloss = 0
        #assert torch.distributed.get_world_size() % 8 == 0, torch.distributed.get_world_size()
        #print( torch.distributed.get_world_size())
        loss = loss + contrastiveloss * 0.5

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_1.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        


        # compute gradient and do SGD step
        loss.backward()


        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'TP {knn_right.val:.4f} ({knn_right.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, knn_right=knn_right, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('tp/train', knn_right.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(test_loader, model, criterion, epoch, args, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target,_,_,_) in enumerate(test_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output,_,_= model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0):
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg

def eval(eval_loader, model, args, log=None):
    # parameter setting from test_model.py
    num_crop = args.eval_test_crops
    
    if args.eval_dense_sample:
        num_crop *= 10  # 10 clips for testing when using dense sample

    if args.eval_twice_sample:
        num_crop *= 2

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+ args.modality)
    


    feature_record = torch.ones([eval_loader.sampler.num_samples,2,2048]).cuda(args.gpu, non_blocking=True) # mean/var in axis1
    gt_record = torch.zeros([eval_loader.sampler.num_samples, 2]).long().cuda(args.gpu, non_blocking=True) -1
    low_dim_record = torch.ones([eval_loader.sampler.num_samples,128]).cuda(args.gpu, non_blocking=True)

    # get all feature record
    model.eval()

    with torch.no_grad():
        for i, (x, label, gt, index,_) in enumerate(eval_loader):
            # compute output
            
            if log is not None and i%20==0:
                output = "Eval Training: {}/{}\n".format(i, len(eval_loader))
                log.write(output + '\n')
                log.flush()
                print(output)   
            x = x.cuda(args.gpu, non_blocking=True)
            x_in = x.view(-1, length, x.size(2), x.size(3))
            if args.shift:
                x_in = x_in.view(-1, args.num_segments, length, x_in.size(2), x_in.size(3))   
            index = index% eval_loader.sampler.num_samples # ddp
           # print(x_in.shape)
            
            _,feature,low_dim = model(x_in)
            #feature = feature.reshape([-1,1,2048]).mean(dim=1)



            feature_record[index,0] = feature.data.mean(axis=1)
            feature_record[index,1] = feature.data.var(axis=1)
            low_dim_record[index] = low_dim.data
            gt_record[index, 0] = gt.cuda().data
            gt_record[index, 1] = label.cuda().data
    
    return feature_record, gt_record, low_dim_record

def split_dataset(feature_record, gt_record, epoch, actually_clean, believe_clean, log,isclean, MB, args):

    all_feature = feature_record.cpu().numpy()
    gtLabel = gt_record.cpu().numpy().astype(int)
    if isclean is None:
        if 'pair' in args.noise_type and args.noise_rate == 0.4 :
            isclean = innitail_split(all_feature[:,0], gtLabel) #mean
        else:
            isclean = gtLabel[:,1] > -100
    regular_feature = select_feature_var_no_clean(all_feature[isclean,1], gtLabel[isclean,1], k=args.vec_len, K=args.vec_len) 
    
    prob = cos_selection_noclean(regular_feature, all_feature[:,0], gtLabel, isclean, tau = 0.5)
    result = prob > 0.5
    
    num_class = int(gtLabel[:,1].max()+1)

    for c in range(num_class):
        believe_clean[epoch][c] = np.sum(result * (gtLabel[:,1] == c))
        actually_clean[epoch][c] = np.sum(result * (gtLabel[:,1] == c) * (gtLabel[:,0] == c))
    if log is not None:
        output = ("============Eval Result At Epoch {}=============\n".format(epoch) 
          + "believe_clean : {}\n".format(np.sum(believe_clean[epoch]))
          +"actually_clean : {}\n".format(np.sum(actually_clean[epoch])))
        log.write(output + '\n')
        log.flush()
        print(output)

    return result, prob

def save_checkpoint(state, is_best,args):
    filename = '%s/%s/%d_ckpt.pth.tar' % (args.root_model, args.store_name,state['epoch'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
