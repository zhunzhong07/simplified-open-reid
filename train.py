from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
sys.path.append("..")
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import pdb

from reid import datasets
from reid.models import ResNet
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss_biu
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = ResNet(args.depth, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)
    model = nn.DataParallel(model).cuda()
    # Load from checkpoint
    start_epoch = best_map = 0
    if args.if_resume:
        print(args.resume)
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        prior_best_map = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, prior_best_map))
    # model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        return

    # Criterion
    alpha= args.alpha
    beta = args.beta
    gamma = args.gamma
    criterion = TripletLoss_biu(margin = args.margin, num_instances=args.num_instances, 
                                    alpha = alpha, beta =beta , gamma =gamma).cuda()

    # Optimizer
    if args.optimizer == 'sgd':
            # base_param_ids = set(map(id, model.module.base.parameters()))
            # new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
            # param_groups = [
            #     {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            #     {'params': new_params, 'lr_mult': 1.0}]
        param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 40))
        else :
            lr = args.lr if epoch <= 80 else \
                 args.lr * (0.1 ** ((epoch - 100) / 60.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        if epoch % 3 ==0:
            metric.train(model,train_loader)
            top_map = evaluator.evaluate(test_loader, dataset.query, dataset.gallery) 
            is_best = top_map > prior_best_map
            prior_best_map = max(top_map, prior_best_map)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_map': top_map,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        # print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
        #       format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=['cuhk03', 'market1501', 'viper', 'dukemtmc'])
    parser.add_argument('-b', '--batch-size', type=int, default=72)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', type=bool,default=True,
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--depth',  type=int, default='50',
                         choices=[50,101,152])
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--alpha', type=float, default= 1.0)
    parser.add_argument('--beta', type=float, default= 0.0)
    parser.add_argument('--gamma', type=float, default= 0.0)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.0006,
                        help="learning rate of all parameters")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--if_resume', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='/home/mit/biu/simplified_open_reid/xentropy3.tar', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = '/home/mit/biu/simplified_open_reid'
    dataset_dir = '/home/mit/biu/dataset'#TitanX-602
    #dataset_dir = '/mount/jmy/dataset'
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=dataset_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs','TripLet_biu_no_last_relu'))
    main(parser.parse_args())
