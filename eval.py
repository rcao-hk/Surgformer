import os
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse
from pandas import read_csv

# 1.3.2 F1-score bug
print('sklearn version:', sklearn.__version__)

def read_labels(file_path):
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.txt':
        labels = []
        first_line = True
        with open(file_path, 'r') as file:
            for line in file:
                if first_line:
                    first_line = False
                    continue
                labels.append(int(line.strip().split()[1]))
            # labels = [int(line.strip().split()[1]) for line in file]
    elif file_ext == '.csv':
        truth_data = read_csv(file_path, header=0, skipinitialspace=True, dtype='Int64').squeeze('columns')
        labels = truth_data["Steps"].tolist()
    return labels


# cholec80
# method = 'Memba_v0.1.5_'
# epoch_num = 60
# Video-level Accuracy: 92.76970931720015 ± 6.388062765771166
# Phase-level Precision: 86.42617703735596
# Phase-level Recall: 85.64331828534313
# Phase-level Jaccard Index: 76.09277527531343

# method = 'Memba_v0.1.5.1_'
# epoch_num = 67
# Video-level Accuracy: 92.79215457546006 ± 5.80628063507965
# Phase-level Precision: 86.56807069206415
# Phase-level Recall: 86.5162122893151
# Phase-level Jaccard Index: 76.07554480509594

# method = 'Memba_v0.1.5.2_'
# epoch_num = 50
# Video-level Accuracy: 92.60485887756991 ± 6.721705122671121
# Phase-level Precision: 84.76582763929585
# Phase-level Recall: 87.7550129846815
# Phase-level Jaccard Index: 75.86197981981016

# method = 'Memba_v0.1.6'
# ------------------epoch number: 43 -------------------------
# Video-level Accuracy: 87.19473721615778 ± 7.129344642663641
# Phase-level Precision: 81.24817634112692
# Phase-level Recall: 78.29734157573237
# Phase-level Jaccard Index: 66.33513255480425

# method = 'Memba_v0.1.7'
# ------------------epoch number: 69 -------------------------
# Video-level Accuracy: 91.42790066362976 ± 6.3472033993297705
# Phase-level Precision: 82.50721998455049
# Phase-level Recall: 87.27583840622934
# Phase-level Jaccard Index: 73.63533629787608

# method = 'Mamba_v0.1.9'
# ------------------epoch number: 174 -----------------------
# Video-level Accuracy: 91.73614240567 ± 5.428220569249169
# Phase-level Precision: 85.0554145930665
# Phase-level Recall: 86.61883885211914
# Phase-level Jaccard Index: 75.5017442597058

# method = 'Mamba_v0.2.1'
# ------------------epoch number: 152 -------------------------
# Video-level Accuracy: 92.6019874546899 ± 4.8063885287891965
# Phase-level Precision: 86.84056504683622
# Phase-level Recall: 87.30366243170667
# Phase-level Jaccard Index: 77.34145352763029

# method = 'UMamba_v0.1'
# ------------------epoch number: 74 -------------------------
# Video-level Accuracy: 90.56948384509431 ± 7.223739440004349
# Phase-level Precision: 82.05027216708147
# Phase-level Recall: 86.28984749077489
# Phase-level Jaccard Index: 72.74808642968785

# method = 'UMamba_v0.1.1'
# ------------------epoch number: 74 -------------------------
# Video-level Accuracy: 90.56948384509431 ± 7.223739440004349
# Phase-level Precision: 82.05027216708147
# Phase-level Recall: 86.28984749077489
# Phase-level Jaccard Index: 72.74808642968785

# method = 'SMamba_v0.2.3'
# ------------------epoch number: 199 -------------------------
# Video-level Accuracy: 92.01175259865495 ± 5.705331775402388
# Phase-level Precision: 86.38022970599059
# Phase-level Recall: 87.18322136929669
# Phase-level Jaccard Index: 76.92885562328739

# method = 'UMamba_v0.1.6'
# ------------------epoch number: 169 -------------------------
# Video-level Accuracy: 92.0418656813986 ± 7.734857890557254
# Phase-level Precision: 85.96154283209623
# Phase-level Recall: 89.01385806082757
# Phase-level Jaccard Index: 77.83024112402919

# method = 'UMamba_v0.1.6.1'
# ------------------epoch number: 89 -------------------------
# Video-level Accuracy: 92.16871801268181 ± 6.186106703117174
# Phase-level Precision: 84.34444541726494
# Phase-level Recall: 88.55732854303831
# Phase-level Jaccard Index: 76.32306616137116

# autolaparo
# method = 'SMamba_v0.2.3'
# ------------------epoch number: 7 -------------------------
# Video-level Accuracy: 75.88490120799771 ± 8.345051757075291
# Phase-level Precision: 68.68173880114554
# Phase-level Recall: 60.727416521898625
# Phase-level Jaccard Index: 49.20150774985429

# method = 'SMamba_v0.2.4'
# ------------------epoch number: 153 -------------------------
# Video-level Accuracy: 78.29201020421634 ± 8.29339631753122
# Phase-level Precision: 73.32191040607458
# Phase-level Recall: 64.03569996699895
# Phase-level Jaccard Index: 52.20205858645349

# method = 'SMamba_v0.2.5'
# ------------------epoch number: 167 -------------------------
# Video-level Accuracy: 79.1676395357446 ± 8.27089513733491
# Phase-level Precision: 71.4578604746732
# Phase-level Recall: 65.31612308003865
# Phase-level Jaccard Index: 53.04395323488775

# method = 'SMamba_v0.2.6'
# ------------------epoch number: 153 -------------------------
# Video-level Accuracy: 78.92591561686402 ± 8.715897873510633
# Phase-level Precision: 74.74056402073882
# Phase-level Recall: 64.58695517340524
# Phase-level Jaccard Index: 52.78786657668909

# method = 'SMamba_v0.2.7'
# ------------------epoch number: 379 -------------------------
# Video-level Accuracy: 80.30281531327536 ± 7.79122753599813
# Phase-level Precision: 75.23071410959491
# Phase-level Recall: 66.46540608031214
# Phase-level Jaccard Index: 55.02895230115685

# method = 'SMamba_v0.2.8'
# ------------------epoch number: 359 -------------------------
# Video-level Accuracy: 80.20572587198355 ± 8.268387511291394
# Phase-level Precision: 76.99505667954469
# Phase-level Recall: 66.14400189449893
# Phase-level Jaccard Index: 54.99754898090789

# method = 'SMamba_v0.1.7'
# ------------------epoch number: 5 -------------------------
# Video-level Accuracy: 75.91584924556389 ± 8.744658241687592
# Phase-level Precision: 67.64260579766301
# Phase-level Recall: 60.56108680758244
# Phase-level Jaccard Index: 48.45831953014093

# method = 'UMamba_v0.1.6'
# ------------------epoch number: 174 -------------------------
# Video-level Accuracy: 78.15364010547134 ± 7.574324156395639
# Phase-level Precision: 68.59932311197153
# Phase-level Recall: 62.615077124884635
# Phase-level Jaccard Index: 52.24809102150465

# method = 'UMamba_v0.1.6.2'
# ------------------epoch number: 174 -------------------------
# Video-level Accuracy: 78.15364010547134 ± 7.574324156395639
# Phase-level Precision: 68.59932311197153
# Phase-level Recall: 62.615077124884635
# Phase-level Jaccard Index: 52.24809102150465

# method = 'tecno_v0.1'
# ------------------epoch number: 93 -------------------------
# Video-level Accuracy: 80.61272736249997 ± 7.143918122881357
# Phase-level Precision: 73.00634813941053
# Phase-level Recall: 64.93230354255107
# Phase-level Jaccard Index: 53.60710923722532

def get_args_parser():
    parser = argparse.ArgumentParser('Sequence Mamba training and evaluation script', add_help=False)
    parser.add_argument('--result_root', type=str, default='/media/gpuadmin/rcao/result')
    parser.add_argument('--dataset', default='cholec80', choices=['cholec80', 'autolaparo'])
    parser.add_argument('--seed', default=0, type=int)
    return parser

parser = argparse.ArgumentParser('SR-Mamba evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

dataset = args.dataset  # 'cholec80' 'autolaparo'
result_root = args.result_root
method_list = ['surgformer_HTA_KCA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4']
report_epoch = [99, 199]
# vis_acc_per_video = False

# 路径设置
if dataset == 'cholec80':
    eval_list = range(41, 81)
    pred_filename_holder = 'video-{:02d}.txt'
    gt_filename_holder = 'video-{:02d}.txt'

acc_runs = []
prec_runs = []
rec_runs = []
jac_runs = []
for method_id in method_list:

    # phase-level
    all_gt_data = []
    all_pred_data = []

    # video-level
    accuracy_list = []
    # precision_list = []
    # recall_list = []
    # jaccard_list = []

    gt_folder = os.path.join(result_root, dataset, method_id, 'phase_annotations')
    pred_folder = os.path.join(result_root, dataset, method_id, 'prediction')
    length_list = []
    for video_idx in eval_list:
        pred_filename = pred_filename_holder.format(video_idx)
        gt_filename = gt_filename_holder.format(video_idx)
        gt_path = os.path.join(gt_folder, gt_filename)
        pred_path = os.path.join(pred_folder, pred_filename)
        
        if os.path.exists(pred_path):
            # 读取 ground truth 和预测数据
            gt_data = read_labels(gt_path)
            pred_data = read_labels(pred_path)

            # 追加数据
            all_gt_data.extend(gt_data)
            all_pred_data.extend(pred_data)

            length_list.append(len(gt_data))
            accuracy_list.append(accuracy_score(gt_data, pred_data) * 100)
            # precision_list.append(precision_score(gt_data, pred_data, average='macro', labels=list(range(7))) * 100)
            # recall_list.append(recall_score(gt_data, pred_data, average='macro', labels=list(range(7))) * 100)
            # jaccard_list.append(jaccard_score(gt_data, pred_data, average='macro', labels=list(range(7))) * 100)

    # 计算平均指标
    # mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    # mean_precision = sum(precision_list) / len(precision_list)
    # mean_recall = sum(recall_list) / len(recall_list)
    # mean_jaccard = sum(jaccard_list) / len(jaccard_list)

    mean_acc = np.mean(accuracy_list)
    std_acc = np.std(accuracy_list)

    # print(f"Video-level Precision: {mean_precision}")
    # print(f"Video-level Recall: {mean_recall}")
    # print(f"Video-level Jaccard Index: {mean_jaccard}")

    # 计算整体指标
    accuracy = accuracy_score(all_gt_data, all_pred_data)
    precision = precision_score(all_gt_data, all_pred_data, average='macro') * 100
    recall = recall_score(all_gt_data, all_pred_data, average='macro') * 100
    jaccard = jaccard_score(all_gt_data, all_pred_data, average='macro') * 100

    # print(f"Phase-level Accuracy: {accuracy}")
    print(f"------------------------------------------")
    print(f"Video-level Accuracy: {mean_acc} ± {std_acc}")
    print(f"Phase-level Precision: {precision}")
    print(f"Phase-level Recall: {recall}")
    print(f"Phase-level Jaccard Index: {jaccard}")
    
    acc_runs.append(mean_acc)
    prec_runs.append(precision)
    rec_runs.append(recall)
    jac_runs.append(jaccard)
    
    
print(f"------------------epoch number: " + str(report_epoch) + " -------------------------")
print(f"Accuracy: {np.mean(acc_runs)} ± {np.std(acc_runs)}")
print(f"Precision: {np.mean(prec_runs)} ± {np.std(prec_runs)}")
print(f"Recall: {np.mean(rec_runs)} ± {np.std(rec_runs)}")
print(f"Jaccard Index: {np.mean(jac_runs)} ± {np.std(jac_runs)}")