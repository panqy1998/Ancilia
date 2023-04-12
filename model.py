import torch
from torch import Tensor, autograd
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, dim, num_classes):
        super(Discriminator, self).__init__()
        main = torch.nn.Sequential(
            torch.nn.Linear(num_classes, dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(dim, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Linear') != -1:
            self.weight.data.normal_(0.0, 0.02)
            self.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            self.weight.data.normal_(1.0, 0.02)
            self.bias.data.fill_(0)

    def calc_gradient_penalty(self, real_data, fake_data, device, lamb):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.main(interpolates).view(-1)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
        return gradient_penalty


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, nclass: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, nclass)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, embed=False) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        if not embed:
            x = self.fc(x)
        return x


class FC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layer=1):
        super(FC, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, out_channels)
        layers = []
        for i in range(layer - 1):
            layers.append(torch.nn.Linear(in_channels, in_channels))
            layers.append(torch.nn.ReLU(True))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.fc1(x)
        return x


class Bandits:
    def __init__(self, nbandits, narms, gamma, k, epoch, device, edges):
        self.weights = torch.zeros((nbandits, nbandits)).to(device)
        self.gamma = gamma
        self.nbandits = nbandits
        self.narms = narms
        self.k = k
        self.delta = torch.sqrt((1 - gamma) * gamma ** 4 * k ** 5 * torch.log(narms / k) / (epoch * narms ** 4))
        #        self.U = torch.zeros((nbandits, narms), dtype=torch.bool).to(device)
        self.device = device
        self.weights[edges[0], edges[1]] = 1
        self.st = None
        self.p = None
        self.ut = None

    def playm(self):
        rhs = (1 / self.k - self.gamma / self.narms) / (1 - self.gamma)
        sum_weights = torch.sum(self.weights, dim=1, keepdim=False)
        ut = torch.zeros((self.nbandits, self.nbandits), dtype=torch.bool, device=self.device)
        ut[self.narms <= self.k] = self.weights[sum_weights <= self.k] > 0
        self.p = self.weights.clone().detach()
        for js in torch.nonzero(self.narms > self.k).detach().tolist():
            j = js[0]
            if max(self.weights[j]) > rhs[j]:
                w_sorted = sorted(self.weights[j], reverse=True)
                for i in range(len(w_sorted)):
                    alpha = (rhs[j] * sum_weights[j]) / (1 - i * rhs[j])
                    curr = w_sorted[i]
                    if alpha > curr:
                        ut[j] = self.weights[j] >= alpha
                        self.weights[j][ut[j]] = alpha
                        break
                    sum_weights[j] = sum_weights[j] - curr
                    if i + 1 == len(w_sorted):
                        raise Exception("Alpha does not found.")
            p = ((1 - self.gamma) * self.weights[j] / sum_weights[j] + self.gamma / self.narms[j]) * self.k
            self.p[j] = p
        self.ut = ut

    def updatem(self, rewards):
        rbar = rewards / self.p
        self.weights[not self.ut] *= torch.exp(self.delta * rbar)

    def samplem(self, k):
        p = self.p.clone().detach()
        while torch.sum(torch.logical_and(p > 0, p < 1)) > 0:
            bandits = torch.nonzero(torch.sum(torch.logical_and(p > 0, p < 1), dim=1, keepdim=False))
            for bandit in bandits:
                beta = min(1 - p[bandit, 0], p[bandit, 0])
                zeta = min(1 - p[bandit, 1], p[bandit, 1])
                if torch.rand(1, device=self.device)[0] < zeta / (zeta + beta):
                    p[bandit, 0] += beta
                    p[bandit, 1] -= beta
                else:
                    p[bandit, 0] -= zeta
                    p[bandit, 1] += zeta
        self.st = p.to(torch.bool)
        return torch.nonzero(p >= 1).T

    def sample(self, k):
        bandits = self.narms >= k

        r = torch.rand((self.nbandits, self.nbandits), device=self.device)
        self.st = torch.nonzero(self.p > r).T
        return self.st

    def play(self):
        self.p = (1-self.gamma)*self.weights/torch.sum(self.weights, dim=1)+self.gamma/self.narms
        self.p[self.narms == 0] = 0

    def update(self, rewards):
        rbar = rewards / self.p
        self.weights *= torch.exp(self.delta * rbar)
