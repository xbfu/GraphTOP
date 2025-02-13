import logging
import torch

from logger import Logger
from utils import set_random_seed, get_args
from task import NodeTask


if __name__ == '__main__':
    args = get_args()
    pretrain_task = args.pretrain_task
    assert pretrain_task in ['GraphCL', 'SimGRACE', 'EdgePredGPPT', 'EdgePredGraphPrompt']
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    for seed in range(5):
        print('Dataset_name:', args.dataset_name,
              ' | pretrain_task:', pretrain_task,
              ' | shots:', args.shots,
              ' | lambda_e:', args.lambda_e,
              ' | lambda_s:', args.lambda_s,
              ' | seed:', seed)

        filename = f'log/{args.dataset_name}_{pretrain_task}_{args.shots}_{args.lr}_{args.lambda_e}_{args.lambda_s}_{seed}.log'
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        logger = Logger(filename, formatter)

        set_random_seed(seed)

        task = NodeTask(args.dataset_name, args.shots, args.hidden_dim, device, pretrain_task, logger)
        task.train(args.batch_size, lr=args.lr, epochs=args.epochs, lambda_e=args.lambda_e, lambda_s=args.lambda_s)
