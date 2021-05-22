import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from itertools import groupby
from operator import itemgetter
from functools import cmp_to_key
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

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data = np.mean(data, axis=-1, keepdims=True)
    std_data = np.std(data, ddof=1, axis=-1, keepdims=True)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data - mean_data) / std_data


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return np.sum(norm_data(data0) * norm_data(data1), axis=-1) / (data0.shape[-1] - 1)

diffs = []
logs = []
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(12, 6)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(wspace=0.2)

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
                diffs_stacked = np.stack(list(map(itemgetter(0), diff_group)), axis=-1)

                y_reg = np.array(data['incorrects']['regular'])
                y_cor = np.array(data['incorrects']['corrupt'])

                y_reg_dif = np.tile(y_reg[1:] - y_reg[:-1], reps=(*diffs_stacked.shape[:-1], 1))
                y_cor_dif = np.tile(y_cor[1:] - y_cor[:-1], reps=(*diffs_stacked.shape[:-1], 1))

                ncc_reg = ncc(diffs_stacked, y_reg_dif)
                ncc_cor = ncc(diffs_stacked, y_cor_dif)

                plt.clf()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                with sns.axes_style('white'):
                    sns.heatmap(ncc_reg, vmax=1.0, vmin=-1.0, ax=ax1, center=0, cmap='seismic')
                    sns.heatmap(ncc_cor, vmax=1.0, vmin=-1.0, ax=ax2, center=0, cmap='seismic')

                plt.suptitle(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption)')
                ax1.set_title(f'{layer_str} NCC with Pristine Incorrect Difference')
                ax2.set_title(f'{layer_str} NCC with Corrupt Incorrect Difference')

                plt.savefig(f'{dataset}_heatmaps/seed{SEED}_MLPw512d{d}_cor{cor}_{layer_str.lower().replace(" ", "")}_ncc.png')

                plt.clf()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                reg_truth = (ncc_reg < 0.6) & (ncc_reg > -0.6)
                cor_truth = (ncc_cor < 0.6) & (ncc_cor > -0.6)

                num_reg = (~reg_truth).sum()
                num_cor = (~cor_truth).sum()

                if num_cor == num_reg:
                    cat = 'Mixed'
                elif num_cor >= 2 * num_reg:
                    cat = 'Corrupt'
                elif num_cor > num_reg:
                    cat = 'Corrupt?'
                elif 2 * num_cor > num_reg:
                    cat = 'Pristine?'
                else:
                    cat = 'Pristine'

                logs.append(('{:7} MLP {}x512 ({}% Corruption) | {:25} | # Pristine Correlated: {:4} (Fraction: {:.6f}) | # Corrupt Correlated: {:4} (Fraction: {:.6f}) | Overlap: {:3} | {:10}\n',
                             dataset.upper(), d, cor, layer_str, num_reg, (~reg_truth).mean(), num_cor, (~cor_truth).mean(), (~reg_truth & ~cor_truth).sum(), cat))

                ncc_reg[reg_truth] = 0.0
                ncc_cor[cor_truth] = 0.0

                with sns.axes_style('white'):
                    sns.heatmap(ncc_reg, vmax=1.0, vmin=-1.0, ax=ax1, center=0, cmap='seismic')
                    sns.heatmap(ncc_cor, vmax=1.0, vmin=-1.0, ax=ax2, center=0, cmap='seismic')

                plt.suptitle(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption)')
                ax1.set_title(f'{layer_str} NCC with Pristine Incorrect Difference')
                ax2.set_title(f'{layer_str} NCC with Corrupt Incorrect Difference')

                plt.savefig(f'{dataset}_heatmaps/seed{SEED}_MLPw512d{d}_cor{cor}_{layer_str.lower().replace(" ", "")}_ncc_filtered.png')

            diffs = []

def compare(t1, t2):
    cmp = t1[-4] - t2[-4]
    return cmp if cmp != 0 else t1[-6] - t2[-6]

with open('logs.txt', 'w') as log_file:
    for log, *args in sorted(logs, key=cmp_to_key(compare), reverse=True):
        log_file.write(log.format(*args))
