import logging
import argparse
import glob
import pickle

import torch

from train_utils import (
    ScoreOptimizer,
    apply_nm_config,
    backup_source_files,
    get_dataloader_model,
    get_density,
    learning_pipeline,
    resume_model,
    separate_params,
)

parser = argparse.ArgumentParser(description='prune')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--task', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet20')
# training settings
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--resume_path', type=str)
parser.add_argument('--resume', type=int, default=1)
parser.add_argument('--lr_milestones', type=list, default=[40, 60, 70])
parser.add_argument('--lr_warmup_epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--do_train', type=int, default=0)
parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--update_score', type=int, default=1)
# prune and quant
parser.add_argument('--prune_wgt_method', type=str, default='UNIFORM')
parser.add_argument('--quant_alpha', type=float, default=0.0, help='clipping threshold')
parser.add_argument('--quant_bits', type=int, default=0, help='0 means using floating-point')
parser.add_argument('--target_density', type=int, default=375, help='target density in permillage')
parser.add_argument('--prune_M', type=int, default=8)
parser.add_argument('--prune_N', type=int, default=3, help='only unstructured synpase pruning will use this')
parser.add_argument('--prune_unstructured', type=int, default=0, help='use unstructured pruning')
parser.add_argument('--prune_warmup_epoch', type=int, default=0,
                    help='#epochs of training before applying pruning')
parser.add_argument('--mask_score_momentum', type=float, default=0.99)
# adder function
parser.add_argument('--use_transform', type=int, default=0)

args = parser.parse_args()

backup_source_files(args)

def train():
    logging.basicConfig(filename=f'./outputs/{args.exp_name}/run.log', level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(str(args))

    # get data loader and model
    model, data_train_loader, data_test_loader = get_dataloader_model(args)
    if args.resume == 1:
        resume_model(args, model)
    # apply nm_config
    nm_config_file_name = (
        f"./nm_config/{args.prune_wgt_method}/"
        f"{args.task}_{args.model}_nmConfig_M{args.prune_M}_tar{args.target_density}_*.plk"
    )
    print(nm_config_file_name)
    logging.info(nm_config_file_name)
    file_name = glob.glob(nm_config_file_name)[0]
    with open(file_name, 'rb') as pkl_file:
        nm_dict = pickle.load(pkl_file)
    logging.info(str(nm_dict))
    apply_nm_config(model, nm_config=nm_dict)
    density = get_density(model)
    logging.info(f"actual density: {density}")
    print(f"actual density: {density}")

    model = torch.nn.DataParallel(model).cuda()
    # loss and opt
    criterion = torch.nn.CrossEntropyLoss().cuda()

    group_normal, group_special = separate_params(model)
    optimizer = torch.optim.SGD(group_normal, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_special = ScoreOptimizer(group_special, update_m=args.mask_score_momentum)
    learning_pipeline(
        args,
        model,
        data_train_loader,
        data_test_loader,
        optimizer,
        optimizer_special,
        criterion,
        logging,
    )


def main():
    train()


if __name__ == '__main__':
    main()
