import community as community_louvain
from torch_geometric.utils import to_networkx, subgraph
import numpy as np
from torch_geometric.data import Data
import torch
import pandas as pd
import os
import scipy.sparse as sp


def louvain_graph_cut(whole_graph, num_owners, delta, verbose=False):
    edges = whole_graph.edge_index
    G = to_networkx(whole_graph, to_undirected=True)
    partition = community_louvain.best_partition(G, random_state=0)
    groups = []
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    if verbose:
        print(groups)
    partition_groups = {group_i: [] for group_i in groups}
    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    if delta <= 0 or delta >= 1:
        exit("Error: delta should belong to (0,1).")
    group_len_max = int(len(list(G.nodes())) // num_owners * (1 - delta))
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    if verbose:
        print(groups)
    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id: [] for owner_id in range(num_owners)}
    owner_nodes_len = len(list(G.nodes())) // num_owners
    owner_list = [i for i in range(num_owners)]
    owner_ind = 0
    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        count = 0
        while len(owner_node_ids[owner_list[owner_ind]]) + len(
                partition_groups[group_i]) >= owner_nodes_len * (1 + delta):
            owner_ind = (owner_ind + 1) % len(owner_list)
            count += 1
            if count % num_owners == 0:
                exit("Error: fail to create a federated graph dataset.\n"
                     "You can relax the requirement of data size in the data owner, \n"
                     "or increase the number of data owners.")
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    if verbose:
        for owner_i in owner_node_ids.keys():
            print('nodes len for ' + str(owner_i) + ' = ' + str(len(owner_node_ids[owner_i])))
    local_G = []
    local_nodes_ids = []
    subj_set = list(set(whole_graph.y.detach().tolist()))
    local_node_subj_0 = []
    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = torch.tensor(partition_i)
        sbj_i = whole_graph.y[locs_i].detach().numpy()
        local_node_subj_0.append(sbj_i)
    count = []
    for owner_i in range(num_owners):
        count_i = {k: [] for k in subj_set}
        sbj_i = local_node_subj_0[owner_i]
        for i in range(len(sbj_i)):
            count_i[sbj_i[i]].append(owner_node_ids[owner_i][i])
        count.append(count_i)
    for k in subj_set:
        for owner_i in range(num_owners):
            if len(count[owner_i][k]) < 2:
                for j in range(num_owners):
                    if len(count[j][k]) > 2:
                        id = count[j][k][-1]
                        count[j][k].remove(id)
                        count[owner_i][k].append(id)
                        owner_node_ids[owner_i].append(id)
                        owner_node_ids[j].remove(id)
                        break
    for owner_i in range(num_owners):
        partition_i = owner_node_ids[owner_i]
        locs_i = torch.tensor(partition_i)
        sbj_i = whole_graph.y[locs_i]
        local_nodes_ids.append(partition_i)
        feats_i = whole_graph.x[locs_i]
        local_edges, local_weights = subgraph(subset=torch.tensor(partition_i), edge_index=edges, relabel_nodes=True,
                                              edge_attr=whole_graph.edge_attr)
        graph_i = Data(x=feats_i, y=sbj_i, edge_index=local_edges, edge_attr=local_weights)
        if "sens" in whole_graph.keys:
            graph_i["sens"] = whole_graph.sens[locs_i]
        local_G.append(graph_i)
    return local_G, local_nodes_ids


def f1_score(nb_classes, preds, classes, device):
    confusion_matrix = torch.zeros(nb_classes, nb_classes).to(device)
    for t, p in zip(classes, preds):
        confusion_matrix[t.long(), p.long()] += 1
    precision = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)
    f1 = precision * recall / (precision + recall)
    f1 = torch.nan_to_num(f1)
    return f1.mean()


def reward1(embeddings, edges, device):
    edges = edges.T
    n = embeddings.shape[0]
    r = torch.zeros((n, n)).to(device)
    for e in edges:
        r[e] = embeddings[e[0]] * embeddings[e[1]].T
    return r


def reward2(embeddings, edges, device):
    n = embeddings.shape[0]
    r = torch.zeros((n, n)).to(device)
    edges_sparse = torch.zeros((n, n), dtype=torch.bool).to(device)
    edges_sparse[edges.T[0], edges.T[1]] = 1
    for i in range(n):
        if sum(edges_sparse[i]) > 0:
            central_e = torch.sum(embeddings[edges_sparse[i]], dim=0, keepdim=True) / sum(edges_sparse[i])
            r[i][edges_sparse[i]] = torch.relu(2 * torch.matmul(embeddings[edges_sparse[i]], central_e.T) -
                                               torch.norm(embeddings[edges_sparse[i]], dim=1, keepdim=False) ** 2)
    return r


def load_pokec(index: str, sens_attr: str = "body_type_indicator") -> Data:
    dir = "./pokec-" + index
    print('Loading dataset from {}'.format(dir))
    idx_features_labels = pd.read_csv(os.path.join(dir, "region_job.csv"))  # 67796*279
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove("completion_percentage")
    header.remove("AGE")
    header.remove(sens_attr)
    predict_attr = "I_am_working_in_field"
    header.remove(predict_attr)  # remove predictable feature
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
    labels = idx_features_labels[predict_attr].values  # array([-1,  0,  1,  2,  3, 4], dtype=int64)
    label_idx = np.where(labels < 0)[0]
    labels[label_idx] = np.max(labels) + 1  # convert negative label to positive
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(dir, "region_job_relationship.txt"), dtype=int)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    #   adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                      shape=(labels.shape[0], labels.shape[0]),
    #                     dtype=np.float32)
    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # adj = adj + sp.eye(adj.shape[0])
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)  # -1,0,1,2,3,4
    data = Data(x=features, edge_index=torch.LongTensor(edges.T), y=labels)
    data["sens"] = torch.LongTensor(idx_features_labels[sens_attr].values)
    return data


def load_bail(sens_attr="WHITE"):
    def build_relationship(x, thresh=0.25):
        from scipy.spatial import distance_matrix
        df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
            import random
            random.seed(912)
            random.shuffle(neig_id)
            for neig in neig_id[:200]:
                if neig != ind:
                    idx_map.append([ind, neig])
        # print('building edge relationship complete')
        idx_map = np.array(idx_map)

        return idx_map


    dataset = "bail"
    path = "./bail"
    predict_attr = "RECID"
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))  # 67796*279
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)  # non-sensitive
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    data = Data(x=features, edge_index=torch.LongTensor(edges.T), y=labels)
    data["sens"] = sens
    return data
