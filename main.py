import argparse
from copy import deepcopy
import torch.nn.functional as F
import torch
from torch import autograd
from torch_geometric.transforms import RandomNodeSplit
import os
from torch_geometric.utils import sort_edge_index, degree, subgraph
import time
from aggregate import FedAvg, FedCluster
from utils import louvain_graph_cut, f1_score, reward2 as reward, load_pokec, load_bail
from model import GCN, FC, Bandits

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pokec-n')
parser.add_argument('-n', type=int, default=10)
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dimension', type=int, default=16)
parser.add_argument('--d_dimension', type=int, default=32)
parser.add_argument('--val_ratio', type=float, default=0.3)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lrd', type=float, default=0.01)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--locald_ep', type=int, default=1)
parser.add_argument('--localg_ep', type=int, default=5)
parser.add_argument('--aggr', type=str, default='FedAvg')
parser.add_argument('--aggrd', type=str, default='FedAvg')
parser.add_argument('--sens_index', type=int, default=3)
parser.add_argument('--lamb', type=float, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--sampler', type=str, default="no")
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('-k', type=float, default=100)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.aggr == "FedAvg":
    aggr = FedAvg
else:
    exit("Error: unrecognized aggregation method " + args.aggr)
    aggr = None
if args.aggrd == "FedCluster":
    aggrd = FedCluster
elif args.aggr == "FedAvg":
    aggrd = FedAvg
else:
    exit("Error: unrecognized aggregation method " + args.aggrd)
    aggrd = None
dataset_path = "./" + args.dataset
if args.dataset == "cora":
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(dataset_path, name="Cora")
    data = dataset[0]
elif args.dataset == "lastfm":
    from torch_geometric.datasets import LastFMAsia

    dataset = LastFMAsia("./lastfm")
    data = dataset[0]
elif args.dataset == "github":
    from torch_geometric.datasets import GitHub

    dataset = GitHub("./github")
    data = dataset[0]
elif args.dataset == "facebook":
    from torch_geometric.datasets import SNAPDataset

    dataset = SNAPDataset("./facebook", name="ego-facebook")
    print(dataset)
    data = dataset
    exit(0)
elif args.dataset == "pokec":
    from torch_geometric.datasets import SNAPDataset
    from torch_geometric.data import download_url, extract_gz
    import os.path as osp
    import os

    dataset = SNAPDataset("./pokec", name="soc-pokec")
    print(dataset)
    print(dataset[0])
    data = dataset[0]
    if osp.isdir(dataset.raw_dir) and len(os.listdir(dataset.raw_dir)) <= 1:
        path = download_url("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", dataset.raw_dir)
        extract_gz(path, dataset.raw_dir)
    exit(0)
elif args.dataset == "pokec-n":
    data = load_pokec("n")
    dataset = [data]
elif args.dataset == "pokec-z":
    data = load_pokec("z")
    dataset = [data]
elif args.dataset == "bail":
    data = load_bail("WHITE")
    dataset = [data]
else:
    data = None
    dataset = None
    exit('Error: unrecognized dataset ' + args.dataset)
data.edge_index = sort_edge_index(data.edge_index)
if len(dataset) < args.n:
    filename = os.path.join(args.dataset, "n-{}-seed-{}-val-{}-test-{}.pt".format(args.n, args.seed, args.val_ratio,
                                                                                  args.test_ratio))
    if not os.path.exists(filename):
        local_data, local_id = louvain_graph_cut(data, args.n, 0.9)
        for i, G in enumerate(local_data):
            print("# of nodes: :", G.num_nodes)
            print("# of edges: :", G.num_edges)
            split = RandomNodeSplit(num_val=args.val_ratio, num_test=args.test_ratio)
            local_data[i] = split(G)
            local_data[i] = local_data[i].to(device)
            # local_data[i]["unsens_x"] = deepcopy(local_data[i].x)
            # local_data[i]["unsens_x"][:, args.sens_index] = 0
            if "sens" not in local_data[i].keys:
                local_data[i]["sens"] = local_data[i].x[:, args.sens_index].to(torch.int64)
        torch.save(local_data, filename)
        local_data = torch.load(filename)
    else:
        print("Load {}".format(filename))
        local_data = torch.load(filename)
else:
    local_data = dataset
if args.sampler == "bandit":
    bandits = []
    edge_mask = []
    for i in range(args.n):
        print(local_data[i].edge_index.shape)
        deg = degree(local_data[i].edge_index[0], local_data[i].num_nodes)
        local_data[i]["sample_mask"] = deg > args.k
        edge_mask.append(torch.zeros(local_data[i].num_edges, dtype=torch.bool).to(device))
        if sum(local_data[i]["sample_mask"]) > 0:
            sample_edges = []
            edge_index = local_data[i].edge_index.clone().detach()
            for j, edge in enumerate(edge_index.T):
                if local_data[i].sample_mask[edge[0]]:
                    sample_edges.append([edge[0].item(), edge[1].item()])
                else:
                    edge_mask[i][j] = True
            bandit = Bandits(sum(local_data[i]["sample_mask"]), deg[local_data[i]["sample_mask"]], args.gamma, args.k,
                             args.epochs * args.localg_ep, device, torch.LongTensor(sample_edges).T.to(device))
            bandits.append(bandit)
            print(local_data[i].edge_index.shape)
        else:
            bandits.append(None)
elif args.sampler == "random":
    for i in range(args.n):
        deg = degree(local_data[i].edge_index[0])
        row, col = local_data[i].edge_index
        p = torch.zeros(row.shape[0], device=device)
        for j, u in enumerate(row):
            p[j] = args.k / deg[u]
        local_data[i]["prob"] = p
else:
    bandits = None
    edge_mask = []
if args.model == "GCN":
    net_g = GCN(data.num_features, args.dimension, args.dimension, int(data.y.max()) + 1).to(device)
else:
    net_g = None
    exit('Error: unrecognized model ' + args.model)
print(net_g)
wg_locals = [net_g.state_dict()] * args.n
# net_d = Discriminator(args.d_dimension, args.dimension).to(device)
net_d = FC(args.dimension, int(data.sens.max()) + 1, 2).to(device)
print(net_d)
# net_d.weights_init()
wd_locals = [net_d.state_dict()] * args.n
min_loss = 0
loss_train = []
loss_test = []
lr = args.lr
lrd = args.lrd
sens_acc = []
one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1
mone = mone.to(device)
timeg, timed = [], []
for iter_i in range(args.epochs):
    loss_locals = []
    val_acc_locals = []
    test_acc_locals = []
    # wd_locals = [net_d.state_dict()] * args.n
    for li in range(args.local_ep):
        tic = time.perf_counter()
        for i in range(args.n):
            local_g = local_data[i]
            local_netg = deepcopy(net_g).to(device)
            local_netd = deepcopy(net_d).to(device)
            local_netg.load_state_dict(wg_locals[i])
            local_netd.load_state_dict(wd_locals[i])
            optimizerD = torch.optim.Adam(local_netd.parameters(), lr=lrd)
            local_netd.requires_grad_(True)
            local_netg.requires_grad_(False)
            #  real_data = local_netg(local_g.x, local_g.edge_index, local_g.edge_attr, True)
            fake_data = local_netg(local_g.x, local_g.edge_index, local_g.edge_attr, True)
            for iter_d in range(args.locald_ep):
                local_netd.zero_grad()
                fake = autograd.Variable(fake_data)
                out = local_netd(fake)
                loss = F.cross_entropy(out, local_g.sens)
                loss.backward()
                optimizerD.step()
            wd_locals[i] = local_netd.state_dict()
        wd_locals = aggrd(wd_locals, int(data.sens.max()) + 1)
        toc = time.perf_counter()
        timed.append(toc - tic)
        # net_d.load_state_dict(wd_global)
        tic = time.perf_counter()
        for i in range(args.n):
            local_g = local_data[i]
            local_netg = deepcopy(net_g).to(device)
            local_netg.load_state_dict(wg_locals[i])
            local_netd = deepcopy(net_d).to(device)
            local_netd.load_state_dict(wd_locals[i])
            local_netd.requires_grad_(False)
            local_netg.requires_grad_(True)
            optimizer = torch.optim.Adam(local_netg.parameters(), lr=lr)
            local_loss = 0
            for iter_g in range(args.localg_ep):
                local_netg.zero_grad()
                if args.sampler == "bandit" and bandits[i] is not None:
                    bandits[i].play()
                    print("Play finish")
                    edge_index = torch.cat((bandits[i].sample(args.k), local_g.edge_index[:, edge_mask[i]]), 1)
                    print("Sample finish")
                elif args.sampler == "random":
                    row, col = local_g.edge_index
                    sample_mask = torch.rand(row.size(0), device=device) <= local_g.prob
                    edge_index = local_g.edge_index[:, sample_mask]
                    if local_g.edge_attr is not None:
                        edge_attr = local_g.edge_attr[sample_mask]
                    else:
                        edge_attr = None
                else:
                    edge_index = local_g.edge_index
                    edge_attr = local_g.edge_attr
                fake_data = local_netg(local_g.x, edge_index, edge_attr, True)
                if args.sampler == "bandit" and bandits[i] is not None:
                    bandits[i].update(reward(fake_data, edge_index, device))
                fake_data = local_netd(fake_data[local_g.sens != 0])
                loss_g = F.cross_entropy(fake_data, torch.zeros(fake_data.shape[0], dtype=torch.int64).to(device))
                loss_g.backward()
                fake_data = local_netg(local_g.x, edge_index, edge_attr)
                loss_gnn = F.cross_entropy(fake_data[local_g.train_mask], local_g.y[local_g.train_mask]) * args.alpha
                loss_gnn.backward()
                optimizer.step()
                loss = loss_g
                local_loss += float(loss)
            wg_locals[i] = local_netg.state_dict()
            loss_locals.append(local_loss / args.localg_ep)
        toc = time.perf_counter()
        timeg.append(toc - tic)
    wg_locals = aggr(wg_locals)
    net_g.load_state_dict(wg_locals[0])
    for i in range(args.n):
        local_g = local_data[i]
        net_g.eval()
        pred = net_g(local_g.x, local_g.edge_index, local_g.edge_attr).argmax(dim=-1)
        accs = []
        for mask in [local_g.val_mask, local_g.test_mask]:
            accs.append(int((pred[mask] == local_g.y[mask]).sum()) / int(mask.sum()))
        val_acc_locals.append(accs[0])
        test_acc_locals.append(accs[1])
    loss_avg = sum(loss_locals) / len(loss_locals)
    val_acc_avg = sum(val_acc_locals) / args.n
    test_acc_avg = sum(test_acc_locals) / args.n
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(iter_i, loss_avg, val_acc_avg, test_acc_avg))
    embeds = []
    test_acc = 0
    f1 = 0
    for i in range(args.n):
        local_g = local_data[i]
        net_g.requires_grad_(False)
        embeds.append(net_g(local_g.x, local_g.edge_index, local_g.edge_attr, embed=True))
    for i in range(args.n):
        local_g = local_data[i]
        sens = local_g.sens.to(torch.int64)
        model_sens = FC(args.dimension, int(sens.max()) + 1, 2).to(device)
        optimizer = torch.optim.Adam(model_sens.parameters(), lr=0.01)
        for epoch in range(1, 5):
            model_sens.train()
            optimizer.zero_grad()
            out = model_sens(embeds[i])
            loss = F.cross_entropy(out, sens)
            loss.backward()
            optimizer.step()
        preds = []
        senss = []
        for j in range(args.n):
            if j != i:
                model_sens.eval()
                out = model_sens(embeds[j])
                pred = out.argmax(dim=-1)
                preds.append(pred)
                senss.append(local_data[j].sens)
        preds = torch.cat(preds)
        senss = torch.cat(senss)
        #      f1 += float(f1_score(int(data.sens.max()) + 1, preds, senss, device))
        test_acc += int((preds == senss).sum()) / len(preds)
    print("Sensitivity inference accuracy {:.4f}".format(test_acc / args.n))
    #  print("F1 score {:.4f}".format(f1 / args.n))
    sens_acc.append(test_acc / args.n)
    lr *= 0.993
    lrd *= 0.993
print("Time of generator: {:.4f}".format(sum(timeg)))
print("Time of discriminator: {:.4f}".format(sum(timed)))
print("Mean sensitivity inference accuracy {:.4f}".format(sum(sens_acc) / len(sens_acc)))
print("75th percentile {:.4f}".format(sorted(sens_acc)[int(len(sens_acc) * 0.75)]))
print("error {:.4f}, {:.4f}".format(max(sens_acc) - sum(sens_acc) / len(sens_acc),
                                    sum(sens_acc) / len(sens_acc) - min(sens_acc)))
with open(os.path.join(args.dataset,
                       "n-{}-seed-{}-val-{}-test-{}-embed-alpha-{}.pck".format(args.n, args.seed, args.val_ratio,
                                                                               args.test_ratio, args.alpha)),
          "wb") as file:
    import pickle as pck
    import numpy as np

    pck.dump(np.array(sens_acc), file)
