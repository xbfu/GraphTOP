import random

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, HeterophilousGraphDataset
from torch_geometric.utils import k_hop_subgraph


def get_subgraphs(data, node_list, num_hops=2):
    graph_list = []
    for subgraph_idx, node in enumerate(node_list):
        subset, edge_index, mapping, _ = k_hop_subgraph(node_idx=node,
                                                        num_hops=num_hops,
                                                        edge_index=data.edge_index,
                                                        relabel_nodes=True)

        edge_index = edge_index.T.tolist()
        target_node = mapping.item()

        neighbor_idx = subset.tolist()
        neighbor_idx.pop(target_node)
        neighbor_mask = [True] * len(subset)
        neighbor_mask[target_node] = False

        adjusted_edge_index = []
        edge_weight = []
        trainable_edge = []
        degree = 0

        for neighbor in range(len(subset)):
            if neighbor != target_node:
                adjusted_edge_index.extend([[target_node, neighbor], [neighbor, target_node]])
                if [target_node, neighbor] in edge_index:
                    degree += 1
                    edge_index.remove([target_node, neighbor])
                    edge_index.remove([neighbor, target_node])
                    edge_weight.extend([1, 1])
                else:
                    edge_weight.extend([0, 0])
                trainable_edge.extend([True, True])

        if len(edge_index) > 0:
            adjusted_edge_index.extend(edge_index)
            edge_weight.extend([1] * len(edge_index))
            trainable_edge.extend([False] * len(edge_index))

        subgraph_data = Data(x=data.x[subset],
                             edge_index=torch.tensor(adjusted_edge_index).T,
                             edge_weight=torch.tensor(edge_weight).float(),
                             y=data.y[node],
                             neighbor_idx=torch.tensor(neighbor_idx),
                             neighbor_mask=torch.tensor(neighbor_mask),
                             target_node=torch.tensor(target_node),
                             target_node_index=torch.tensor(target_node),
                             trainable_edge=torch.tensor(trainable_edge),
                             original_idx=node,
                             node_idx=subset,
                             subgraph_idx=subgraph_idx,
                             degree=degree,
                             )

        graph_list.append(subgraph_data)
    return graph_list


def load_node_data(dataset_name, data_folder):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'{data_folder}/Planetoid', name=dataset_name)
    elif dataset_name in ['Amazon-ratings', 'Minesweeper']:
        dataset = HeterophilousGraphDataset(root=f'{data_folder}/HeterophilousGraphDataset', name=dataset_name)
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=f'{data_folder}/Flickr')
    else:
        return None, -1, -1

    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes

    return data, input_dim, output_dim


def NodeDownstream(data, shots=5, test_node_num=1000):
    num_classes = data.y.max().item() + 1
    node_list = []
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            node_list.extend(indices)
        else:
            node_list.extend(random.sample(indices, k=shots))
    random_node_list = random.sample(range(data.num_nodes), k=data.num_nodes)
    for node in node_list:
        random_node_list.remove(node)
    train_node_list = node_list
    if test_node_num > 1:
        test_node_list = random_node_list[:test_node_num]
    else:
        test_node_list = random_node_list[:int(test_node_num * data.num_nodes)]

    return train_node_list, test_node_list
