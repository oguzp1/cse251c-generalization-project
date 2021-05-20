import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, in_size, out_size, width, depth):
        super().__init__()
        self.width = width
        self.depth = depth

        layers = [
            nn.Flatten(),
            nn.Linear(in_size, width),
            nn.ReLU(),
        ]

        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, out_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        # function that gives you a representation you can save the model with
        return f'MLPw{self.width}d{self.depth}'

# ------------------------------------------------------------------------------------

diffs = []
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(12, 6)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.2)

def update(info, max_diff, min_diff):
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    diff, seed, dataset, depth, cor, epoch, data, stride, layer_str, title = info

    x = torch.arange(1, len(data['incorrects']['regular']) + 1)
    y_reg = torch.tensor(data['incorrects']['regular'])
    y_cor = torch.tensor(data['incorrects']['corrupt'])

    y_reg_dif = y_reg[1:] - y_reg[:-1]
    y_cor_dif = y_cor[1:] - y_cor[:-1]

    plt.suptitle(f'{dataset.upper()} MLP {depth}x512 ({cor}% Corruption)')

    ax2.bar(x[1:], y_reg_dif)
    ax2.bar(x[1:], y_cor_dif)
    ax2.plot([epoch + stride, epoch + stride], ax2.get_ylim(), color='r', label='_nolegend_')
    if stride > 1:
        ax2.plot([epoch, epoch], ax2.get_ylim(), color='g', label='_nolegend_')
    ax2.set_title('Incorrect Prediction Difference')
    ax2.legend(['Pristine', 'Corrupt'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Difference of Number of Incorrect Guesses')

    # ax1.imshow(diff, cmap='seismic', interpolation='nearest')
    with sns.axes_style('white'):
        sb = sns.heatmap(diff, vmax=max_diff, vmin=min_diff, ax=ax1, center=0, cmap='seismic')
    ax1.set_title(title)

def plot_matrices(seed, dataset, depth, cor, epoch, data, stride=1):
    checkpoint1 = torch.load(f'{dataset}_results/seed{seed}_MLPw512d{depth}_cor{cor}_epoch{epoch}.tar')
    checkpoint2 = torch.load(f'{dataset}_results/seed{seed}_MLPw512d{depth}_cor{cor}_epoch{epoch + stride}.tar')

    in_size = (28, 28) if dataset == 'mnist' else (32, 96)
    in_size_int = in_size[0] * in_size[1]

    mlp1 = MLP(in_size_int, 10, 512, depth)
    mlp2 = MLP(in_size_int, 10, 512, depth)

    mlp1.load_state_dict(checkpoint1['model_state_dict'])
    mlp2.load_state_dict(checkpoint2['model_state_dict'])

    for idx, (param1, param2) in enumerate(zip(mlp1.parameters(), mlp2.parameters())):
        param1_numpy = (param1.detach() if param1.dim() == 2 else param1.detach().unsqueeze(1)).numpy()
        param2_numpy = (param2.detach() if param2.dim() == 2 else param2.detach().unsqueeze(1)).numpy()
        diff = param2_numpy - param1_numpy

        layer_str = ''

        if idx == 0:  # 512 x ins_ize input layer
            diff_reshaped = diff.reshape(512, *in_size)
            diff = np.block([[diff_reshaped[16*r + t] for t in range(16)] for r in range(32)])
            layer_str = 'Input Layer Weights'
        if idx == 1:  # 512 x 1 bias input layer
            diff = diff#.reshape(32, 16)
            layer_str = 'Input Layer Bias'
        if idx == 2:  # 512 x 512 1st hidden layer
            layer_str = 'Hidden Layer 1 Weights'
        if idx == 3:  # 512 x 1 bias 1st hidden layer
            diff = diff#.reshape(32, 16)
            layer_str = 'Hidden Layer 1 Bias'

        if depth == 1:
            if idx == 4:  # 10 x 512 output layer
                layer_str = 'Output Layer Weights'
            if idx == 5:  # 10 x 1 bias output layer
                layer_str = 'Output Layer Bias'
        else:
            if idx == 4:  # 512 x 512 2nd hidden layer
                layer_str = 'Hidden Layer 2 Weights'
            if idx == 5:  # 512 x 1 bias 2nd hidden layer
                diff = diff#.reshape(32, 16)
                layer_str = 'Hidden Layer 2 Bias'
            if idx == 6:  # 512 x 512 3rd hidden layer
                layer_str = 'Hidden Layer 3 Weights'
            if idx == 7:  # 512 x 1 bias 3rd hidden layer
                diff = diff#.reshape(32, 16)
                layer_str = 'Hidden Layer 3 Bias'
            if idx == 8:  # 10 x 512 output layer
                layer_str = 'Output Layer Weights'
            if idx == 9:  # 10 x 1 bias output layer
                layer_str = 'Output Layer Bias'

        diffs.append((diff, seed, dataset, depth, cor, epoch, data, stride, layer_str,
                      f'{layer_str} Difference (Epoch {epoch + stride} - {epoch})'))

for SEED, dataset in [(123, 'mnist'), (123, 'cifar10')]:
    for d in [1, 3]:
        for cor in [25, 50, 75]:
            with open(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_info.json') as r:
                data = json.load(r)

            for i in tqdm(range(1, 1000), desc='Epochs'):
                try:
                    plot_matrices(SEED, dataset, d, cor, i, data)
                except IOError:
                    break

            for layer_str, diff_group in tqdm(groupby(sorted(diffs, key=itemgetter(8)), key=itemgetter(8)), desc='GIF'):
                list_diff_group = list(diff_group)#[0:1]

                max_diff = max([t[0].max() for t in list_diff_group])
                min_diff = min([t[0].min() for t in list_diff_group])

                ani = FuncAnimation(fig, update, list_diff_group, fargs=(max_diff, min_diff))
                writer = PillowWriter(fps=3)
                ani.save(f'{dataset}_heatmaps/seed{SEED}_MLPw512d{d}_cor{cor}_{layer_str.lower().replace(" ", "")}.gif', writer=writer)

            diffs = []
