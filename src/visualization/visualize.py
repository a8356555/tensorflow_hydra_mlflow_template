import os

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_excel('datasets/final/test_data_predictions.xlsx')
PATH_OUTPUT_FOLDER = 'reports/figures'
labels = (df['PGS results'] == 'Aneuploidy').values
labels = labels * 1
rgb_preds = df['rgb_pred'].values
flow_preds = df['flow_pred'].values
fused_preds = rgb_preds + flow_preds


## calibration plot (x-axis: quartiles of predictions; y-axis: observed probability of aneuploidy)
def plot_bar(probs_list, deviation_list, w):
    for i, (prob, se) in enumerate(zip(probs_list, deviation_list)):
        plt.plot([i + 1 + w, i + 1 + w], [prob - se, prob + se], color='black')
        plt.plot([i + 1 + w - .1, i + 1 + w + .1], [prob - se, prob - se], color='black')
        plt.plot([i + 1 + w - .1, i + 1 + w + .1], [prob + se, prob + se], color='black')


def plot_num(probs_list, w):
    for i, prob in enumerate(probs_list):
        plt.text(i + 1 + w - .1, prob + .01, '{:.2f}'.format(prob), {'size': 12})


rgb_quantiles = np.percentile(rgb_preds, range(25, 100, 25))
flow_quantiles = np.percentile(flow_preds, range(25, 100, 25))
fused_quantiles = np.percentile(fused_preds, range(25, 100, 25))

rgb_bars = [0] * 4
flow_bars = [0] * 4
fused_bars = [0] * 4
denominator_bars = [0] * 4
for i in range(len(rgb_preds)):
    rgb_exceed_percentile_count = sum(rgb_preds[i] > rgb_quantiles)
    rgb_bars[rgb_exceed_percentile_count] += labels[i] * 1
    flow_exceed_percentile_count = sum(flow_preds[i] > flow_quantiles)
    flow_bars[flow_exceed_percentile_count] += labels[i] * 1
    fused_exceed_percentile_count = sum(fused_preds[i] > fused_quantiles)
    fused_bars[fused_exceed_percentile_count] += labels[i] * 1

    denominator_bars[rgb_exceed_percentile_count] += 1

rgb_observed_probs = [rgb_bar / denominator_bars[i] for i, rgb_bar in enumerate(rgb_bars)]
rgb_se = [
    np.sqrt(rgb_observed_probs[i] * (1 - rgb_observed_probs[i]) / denominator_bars[i])
    for i, rgb_bar in enumerate(rgb_bars)
]
rgb_ci_half = [stats.t.ppf(1 - 0.025, denominator_bars[i]) * rgb_se[i] for i, rgb_bar in enumerate(rgb_bars)]

flow_observed_probs = [flow_bar / denominator_bars[i] for i, flow_bar in enumerate(flow_bars)]
flow_se = [
    np.sqrt(flow_observed_probs[i] * (1 - flow_observed_probs[i]) / denominator_bars[i])
    for i, flow_bar in enumerate(flow_bars)
]
flow_ci_half = [stats.t.ppf(1 - 0.025, denominator_bars[i]) * flow_se[i] for i, flow_bar in enumerate(flow_bars)]

fused_observed_probs = [fused_bar / denominator_bars[i] for i, fused_bar in enumerate(fused_bars)]
fused_se = [
    np.sqrt(fused_observed_probs[i] * (1 - fused_observed_probs[i]) / denominator_bars[i])
    for i, fused_bar in enumerate(fused_bars)
]
fused_ci_half = [stats.t.ppf(1 - 0.025, denominator_bars[i]) * fused_se[i] for i, fused_bar in enumerate(fused_bars)]

colors = {'RGB': '#008fd5', 'Optical Flow': '#6d904f', 'Fused': '#e5ae38'}
plot_labels = list(colors.keys())

lw = 2
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10, 10))
w = 0.3
plt.bar(np.arange(1, 5) - w, rgb_observed_probs, width=w, color=colors["RGB"], align='center')
# plot_bar(rgb_observed_probs, rgb_ci_half, w=-w)
plot_num(rgb_observed_probs, w=-w)
plt.bar(np.arange(1, 5), flow_observed_probs, width=w, color=colors["Optical Flow"], align='center')
# plot_bar(flow_observed_probs, flow_ci_half, w=0)
plot_num(flow_observed_probs, w=0)
plt.bar(np.arange(1, 5) + w, fused_observed_probs, width=w, color=colors["Fused"], align='center')
# plot_bar(fused_observed_probs, fused_ci_half, w=w)
plot_num(fused_observed_probs, w=w)
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[plot_label]) for plot_label in plot_labels]
plt.legend(handles, plot_labels, loc="upper left")
plt.xticks(range(1, 5))
plt.grid(False, axis='x')
plt.xlabel('Quartiles')
plt.ylabel('Observed Probability')
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_FOLDER, 'calibration_quartile.png'), dpi=300)

## plot distribution of different samples
actual_positive_preds = fused_preds[labels == 1] / 2
actual_negative_preds = fused_preds[labels == 0] / 2

binwidth = 0.05
bins = np.arange(0, 1.01, binwidth)

y_ax1, bin_edges_ax1 = np.histogram(actual_positive_preds, bins=bins)
y_ax2, bin_edges_ax2 = np.histogram(1 - actual_negative_preds, bins=bins)

plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
ax1.bar(0.5 * (bin_edges_ax1[1:] + bin_edges_ax1[:-1]),
        y_ax1,
        width=.8 * binwidth,
        align='center',
        color=['#008fd5' if b < .5 else '#fc4f30' for b in bin_edges_ax1])
ax1.set_xlim(-0.05, 1.05)
ax1.grid(False, axis='x')
ax1.set_xlabel('Confidence Score')
ax1.set_ylabel('Frequency (absolute)')
ax1.set_title('Group 1')
ax1.text(0.65, 2.5, 'True Negatives')
ax1.text(0.1, 4, 'False Positives')
ax2.bar(0.5 * (bin_edges_ax2[1:] + bin_edges_ax2[:-1]),
        y_ax2,
        width=.8 * binwidth,
        align='center',
        color=['#e5ae38' if b < .5 else '#6d904f' for b in bin_edges_ax2])
ax2.set_xlim(-0.05, 1.05)
ax2.grid(False, axis='x')
ax2.set_xlabel('Confidence Score')
ax2.set_title('Group 2')
ax2.text(0.65, 15, 'True Positives')
ax2.text(0.1, 4, 'False Negatives')
fig.tight_layout()
fig.savefig(os.path.join(PATH_OUTPUT_FOLDER, 'distribution_prediction.png'), dpi=300)

# ROC
fpr_rgb, tpr_rgb, _ = roc_curve(labels, rgb_preds)
fpr_opflow, tpr_opflow, _ = roc_curve(labels, flow_preds)
fpr_fused, tpr_fused, _ = roc_curve(labels, fused_preds)
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})
lw = 2
plt.figure(figsize=(10, 7))
plt.plot(fpr_rgb,
         tpr_rgb,
         color='deeppink',
         linestyle='--',
         lw=lw,
         label='RGB (area = %0.2f)' % roc_auc_score(labels, rgb_preds))
plt.plot(fpr_opflow, tpr_opflow, color='darkorange', linestyle='--', lw=lw, label='Optical Flow (area = %0.2f)' % 0.67)
'''
Results may be little different due to different version of TF.
Thus, fill in original value.
The correct way to produce ROC curve is commented as below.
'''
# plt.plot(fpr_opflow, tpr_opflow, color='darkorange', linestyle='--',
#          lw=lw, label='Optical Flow (area = %0.2f)' % roc_auc_score(labels, flow_preds))
# plt.plot(fpr_fused, tpr_fused, color='navy',
#          lw=lw+2, label='Fused (area = %0.2f)' % roc_auc_score(labels, fused_preds))
plt.plot(fpr_fused, tpr_fused, color='navy', lw=lw + 2, label='Fused (area = %0.2f)' % 0.74)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver Operating Characteristic')
plt.gca().xaxis.get_major_ticks()[0].label1.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_FOLDER, 'ROC_all_wockpt.png'), dpi=300)
