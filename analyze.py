import torch
from torch import nn
import json
import matplotlib.pyplot as plt
from itertools import groupby

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

for SEED, dataset in [(123, 'mnist'), (123, 'cifar10')]:
    for d in [1, 3]:
        for cor in [25, 50, 75]:

            track = {'regular': {}, 'corrupt': {}}
            incorrects = {'regular': [], 'corrupt': []}
            test_accs = []
            losses = None

            for i in range(1, 1000):
                try:
                    checkpoint = torch.load(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_epoch{i}.tar')

                    for k, v in checkpoint['last_regular_idx_dict'].items():
                        if k not in track['regular']:
                            track['regular'][k] = [-1] * (i - 1)
                        track['regular'][k].append(v)

                    for k, lst in track['regular'].items():
                        if k not in checkpoint['last_regular_idx_dict']:
                            lst.append(-1)

                    for k, v in checkpoint['last_corrupt_idx_dict'].items():
                        if k not in track['corrupt']:
                            track['corrupt'][k] = [-1] * (i - 1)
                        track['corrupt'][k].append(v)

                    for k, lst in track['corrupt'].items():
                        if k not in checkpoint['last_corrupt_idx_dict']:
                            lst.append(-1)

                    incorrects['regular'].append(len(checkpoint['last_regular_idx_dict']))
                    incorrects['corrupt'].append(len(checkpoint['last_corrupt_idx_dict']))

                    test_accs.append(checkpoint['test_accuracy'])

                    losses = checkpoint['losses']
                except IOError:
                    break

            with open(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_info.json', 'w') as w:
                json.dump({
                    'track': track,
                    'test_accuracies': test_accs,
                    'losses': losses,
                    'incorrects': incorrects,
                }, w)

            # plots
            x = torch.arange(1, len(incorrects['regular']) + 1)
            y_reg = torch.tensor(incorrects['regular'])
            y_cor = torch.tensor(incorrects['corrupt'])

            # p vs c incorrect counts
            plt.plot(x, y_reg)
            plt.plot(x, y_cor)
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Incorrect Predictions')
            plt.legend(['Pristine', 'Corrupt'])
            plt.xlabel('Epochs')
            plt.ylabel('Number of Incorrect Guesses')
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_incorrects.png')
            plt.clf()

            # p vs c incorrect percentages
            plt.plot(x, y_reg / y_reg[0].float())
            plt.plot(x, y_cor / y_cor[0].float())
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Incorrect Predictions')
            plt.legend(['Pristine', 'Corrupt'])
            plt.xlabel('Epochs')
            plt.ylabel('Fraction of Incorrect Guesses')
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_incorrects_frac.png')
            plt.clf()

            # p vs c incorrect count diffs
            y_reg_dif = y_reg[1:] - y_reg[:-1]
            y_cor_dif = y_cor[1:] - y_cor[:-1]
            plt.bar(x[1:], y_reg_dif)
            plt.bar(x[1:], y_cor_dif)
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Incorrect Prediction Difference')
            plt.legend(['Pristine', 'Corrupt'])
            plt.xlabel('Epochs')
            plt.ylabel('Difference of Number of Incorrect Guesses')
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_incorrect_diffs.png')
            plt.clf()

            regular_sign_changes = {k: len(list(groupby(lst, key=lambda x: x < 0)))
                                    for k, lst in track['regular'].items()}
            corrupt_sign_changes = {k: len(list(groupby(lst, key=lambda x: x < 0)))
                                    for k, lst in track['corrupt'].items()}

            # sign change hist
            reg_hist = list(regular_sign_changes.values())
            cor_hist = list(corrupt_sign_changes.values())
            plt.hist(reg_hist, bins=(None if not reg_hist else range(max(reg_hist) + 1)))
            plt.hist(cor_hist, bins=(None if not cor_hist else range(max(cor_hist) + 1)))
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Number of Correct/Incorrect Changes')
            plt.legend(['Pristine', 'Corrupt'])
            plt.xlabel('Number of Correct/Incorrect Changes')
            plt.ylabel('Frequency')
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_sign_changes.png')
            plt.clf()

            # loss plot
            plt.plot(x, losses['train'])
            plt.plot(x, losses['test'])
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Losses')
            plt.legend(['Train', 'Test'])
            plt.xlabel('Epochs')
            plt.ylabel('Cross Entropy Loss')
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_loss.png')
            plt.clf()

            # test accuracy plot
            plt.plot(x, test_accs)
            plt.title(f'{dataset.upper()} MLP {d}x512 ({cor}% Corruption) Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy')
            ax = plt.gca()
            ax.set_ylim([0.0, 1.0])
            plt.savefig(f'{dataset}_results/seed{SEED}_MLPw512d{d}_cor{cor}_test_accuracy.png')
            plt.clf()
