import collections
import os

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.resnet20_qp import resnet20_qp
from models.synpase_pruning import adder2d_qp as my_adderConv2d_layer


def get_dataloader_model(args, augment=True):
    if args.task == 'cifar10':
        n_class = 10
        if augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])
        data_train = CIFAR10(args.data, train=True, transform=transform_train, download=True)
        data_test = CIFAR10(args.data, train=False, transform=transform_test, download=True)
        data_train_loader = DataLoader(
            data_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        data_test_loader = DataLoader(
            data_test,
            batch_size=250,
            num_workers=4,
            pin_memory=True,
        )
    else:
        raise NotImplementedError(f"{args.model} is not supported for now")

    if args.model == 'resnet20':
        model = resnet20_qp(args=args, num_classes=n_class)
    else:
        raise NotImplementedError(f"{args.model} is not supported for now")

    return model, data_train_loader, data_test_loader


def apply_nm_config(net, nm_config):
    prefix_module = 'module' in list(nm_config.keys())[0]
    for n, m in net.named_modules():
        if isinstance(m, my_adderConv2d_layer):
            if ('module' in n) and not prefix_module:
                m.prune_N = nm_config[n.replace('module.', '')]
            elif not ('module' in n) and prefix_module:
                m.prune_N = nm_config['module.' + n]
            else:
                m.prune_N = nm_config[n]


def get_density(model):
    n_tot = 0
    n_param = 0
    for name, m in model.named_modules():
        if isinstance(m, my_adderConv2d_layer):
            n_tot += m.adder.numel()
            n_param += m.adder.numel() * m.prune_N / m.prune_M
        elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            n_tot += m.weight.numel()
            n_param += m.weight.numel()
    return n_param / n_tot


class ScoreOptimizer(Optimizer):
    def __init__(self, params, update_m=0.99):
        # Define the parameter groups and initial hyperparameters
        defaults = dict(update_m=update_m)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data = p.data * group['update_m'] + grad * (1. - group['update_m'])


def my_lr_scheduler(
    optimizer,
    base_lr,
    current_step,
    current_epoch,
    num_steps_per_epoch,
    milestone,
    num_warmup_epoch,
):
    if current_epoch < num_warmup_epoch and num_warmup_epoch > 0:
        lr = base_lr * (current_step / (num_steps_per_epoch * num_warmup_epoch))
    else:
        cnt = 0
        for m in milestone:
            if current_epoch >= m:
                cnt += 1
        lr = base_lr * (0.1 ** cnt)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def apply_prune(net: torch.nn.Module, logging):
    for n, m in net.named_modules():
        if isinstance(m, my_adderConv2d_layer):
            m.prune_apply = True
            print(f"{n} warmup finishes. start applying pruning")
            if logging:
                logging.info(f"{n} warmup finishes. start applying pruning")


# separate the parameter groups for the mask_score
def separate_params(net):
    group_special = []
    group_normal = []
    for m in net.modules():
        if isinstance(m, my_adderConv2d_layer):
            group_normal.append(m.adder)
            group_special.append(m.score_nm_mask)
            if m.bias is not None:
                group_normal.append(m.bias)
        elif isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            group_normal.append(m.weight)
            if m.bias is not None:
                group_normal.append(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if m.bias is not None:
                group_normal.append(m.weight)
            if m.bias is not None:
                group_normal.append(m.bias)
    return group_normal, group_special


def backup_source_files(args):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists(f'outputs/{args.exp_name}'):
        os.makedirs(f'outputs/{args.exp_name}')
    if not os.path.exists(f'outputs/{args.exp_name}/src'):
        os.makedirs(f'outputs/{args.exp_name}/src')
    os.system(f'cp models/* outputs/{args.exp_name}/src/')

    # Get the current directory
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    # List all files in the current directory
    files_in_directory = os.listdir(current_directory)
    # Filter out files that end with .py (Python files)
    python_files = [file for file in files_in_directory if file.endswith('.py')]
    for file in python_files:
        os.system(f'cp {file} outputs/{args.exp_name}/src/{file}')


def resume_model(args, net):
    net_ref_dict = torch.load(args.resume_path)
    if not isinstance(net_ref_dict, collections.OrderedDict):
        net_ref_dict = net_ref_dict['state_dict']
    net_dict = net.state_dict()
    tem_dict = collections.OrderedDict()
    for key, value in net_dict.items():
        tem_dict[key] = net_ref_dict['module.' + key]
    net.load_state_dict(tem_dict)
    print(f'Resumed from {args.resume_path}!')


def train_pipeline(
    args,
    current_epoch,
    current_step,
    model,
    data_train_loader,
    optimizer,
    optimizer_special,
    criterion,
):
    model.train()
    _num_steps_per_epoch = len(data_train_loader)
    for i, (images, labels) in enumerate(data_train_loader):
        my_lr_scheduler(
            optimizer=optimizer,
            base_lr=args.lr,
            current_step=current_step,
            current_epoch=current_epoch,
            milestone=args.lr_milestones,
            num_warmup_epoch=args.lr_warmup_epoch,
            num_steps_per_epoch=_num_steps_per_epoch,
        )
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        output = model(images)
        loss_net = criterion(output, labels)
        loss_net.backward()
        optimizer.step()
        optimizer_special.step()
        current_step += 1
    return current_step


def test_pipeline(model, current_epoch, data_test_loader, criterion, logging):
    model.eval()
    top1 = torch.tensor(0.).cuda()
    top5 = torch.tensor(0.).cuda()
    avg_loss = torch.tensor(0.).cuda()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            if criterion:
                avg_loss += criterion(output, labels).item() * images.size(0)

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            top1 += correct[:1].reshape(-1).float().sum(0)
            top5 += correct[:5].reshape(-1).float().sum(0)
    avg_loss /= len(data_test_loader.dataset)
    acc1 = float(top1) / len(data_test_loader.dataset)
    acc5 = float(top5) / len(data_test_loader.dataset)
    s = 'EPOCH %d Test Avg. Loss: %f, Accuracy   TOP1 %f   TOP5 %f' % (
        current_epoch,
        avg_loss,
        acc1,
        acc5,
    )
    if logging:
        logging.info(s)
    return acc1


def learning_pipeline(
    args,
    model,
    data_train_loader,
    data_test_loader,
    optimizer,
    optimizer_special,
    criterion,
    logging,
):
    step_cnt = 0
    # apply_prune(model, logging)
    test_pipeline(model, -1, data_test_loader, criterion, logging)
    if args.do_train == 1:
        best_acc = 0
        for e in range(0, args.epoch):
            if e == args.prune_warmup_epoch:
                apply_prune(model, logging)
            # train and test
            step_cnt = train_pipeline(
                args=args,
                current_epoch=e,
                current_step=step_cnt,
                model=model,
                data_train_loader=data_train_loader,
                optimizer=optimizer,
                optimizer_special=optimizer_special,
                criterion=criterion,
            )
            acc = test_pipeline(model, e, data_test_loader, criterion, logging)
            # save
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            if is_best:
                torch.save(model.state_dict(), f'./outputs/{args.exp_name}/best_model.pth')
            logging.info(f"end of epoch {e} learning rate: {optimizer.param_groups[0]['lr']} best acc: {best_acc}")
        torch.save(model.state_dict(), f'./outputs/{args.exp_name}/last_model.pth')
