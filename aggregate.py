import copy
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def kmeans(data, k, max_time=100):
    n, m = data.shape
    ini = torch.randint(n, (k,)).to(device)  # 只有一维需要逗号
    midpoint = data[ini]  # 随机选择k个起始点
    time = 0
    last_label = 0
    while time < max_time:
        d = data.unsqueeze(0).repeat(k, 1, 1)  # shape k*n*m
        mid_ = midpoint.unsqueeze(1).repeat(1, n, 1)  # shape k*n*m
        dis = torch.sum((d - mid_) ** 2, 2)  # 计算距离
        label = dis.argmin(0)  # 依据最近距离标记label
        if torch.sum(label != last_label) == 0:  # label没有变化,跳出循环
            return label
        last_label = label
        for i in range(k):  # 更新类别中心点，作为下轮迭代起始
            kpoint = data[label == i]
            if i == 0:
                midpoint = kpoint.mean(0).unsqueeze(0)
            else:
                midpoint = torch.cat([midpoint, kpoint.mean(0).unsqueeze(0)], 0)
        time += 1
    return label


def FedAvg(w, k=None):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return [w_avg] * len(w)


def FedCluster(w, k=2):
    w_cluster = copy.deepcopy(w)
    for key in w[0].keys():
        features = []
        shape = w[0][key].shape
        for i in range(len(w)):
            features.append(w[i][key].reshape(-1))
        features = torch.stack(features, dim=0)
        labels = kmeans(features, k).float().squeeze()
        for label in range(k):
            indices = labels == label
            for j, index in enumerate(indices):
                if index:
                    w_cluster[j][key] = torch.mean(features[indices], dim=0).reshape(shape)
    return w_cluster


def local(w, k=None):
    return w
