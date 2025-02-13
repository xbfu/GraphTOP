import numpy as np
import random
import argparse

import torch

def compute_entropy(prob, eps = 1e-10):
    return -torch.mean(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Downstream task: node classification')
    parser.add_argument('--dataset_name', type=str, default='PubMed', help='Dataset name')
    parser.add_argument('--pretrain_task', type=str, default='EdgePredGraphPrompt', help='pre-training task')
    parser.add_argument('--shots', type=int, default=5, help='number of shots (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim (default: 128)')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for training (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
    parser.add_argument('--lambda_e', type=int, default=2, help='lambda_1 (default: 2)')
    parser.add_argument('--lambda_s', type=int, default=10, help='lambda_2 (default: 10)')
    parser.add_argument('--epochs', type=int, default=500, help='hidden_dim (default: 500)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()
    return args
