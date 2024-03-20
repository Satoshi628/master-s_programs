import numpy as np
import os
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from viz import *

import config as c
from model import *
from utils import *
from metrics import compute_pro

def localize(image, depth, st_pixel, labels, fg, mask, batch_ind):
    for i in range(fg.shape[0]):
        if labels[i] > 0:
            fg_i = t2np(fg[i, 0])
            depth_viz = t2np(depth[i, 0])
            depth_viz[fg_i == 0] = np.nan
            viz_maps(t2np(image[i]), depth_viz, t2np(mask[i, 0]), t2np(st_pixel[i]), fg_i,
                     str(batch_ind) + '_' + str(i), norm=True)


def evaluate(test_loader):
    student = Model(nf=not c.asymmetric_student, channels_hidden=c.channels_hidden_student)
    student = load_weights(student, 'student')

    teacher = Model()
    teacher = load_weights(teacher, 'teacher')

    teacher.eval()
    student.eval()

    up = torch.nn.Upsample(size=None, scale_factor=c.depth_len[0] // c.map_len[0], mode='bicubic',
                           align_corners=False)

    test_labels = list()
    mean_st = list()
    max_st = list()

    score_maps = list()
    gt_masks = list()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            depth, fg, labels, image, features, mask = data
            mask_size = mask.shape[-2:]
            depth, fg, image, features, mask = to_device([depth, fg, image, features, mask])
            fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

            img_in = features if c.pre_extracted else image
            fg_down = downsampling(fg, c.map_len, bin=False)

            z_t, jac_t = teacher(img_in, depth)
            z, jac = student(img_in, depth)

            st_loss = get_st_loss(z_t, z, fg_down, per_sample=True)
            st_pixel = get_st_loss(z_t, z, fg_down, per_pixel=True)

            if c.eval_mask:
                st_pixel = st_pixel * fg_down[:, 0]
            st_pixel = up(st_pixel[:, None])[:, 0]

            mean_st.append(t2np(st_loss))
            max_st.append(np.max(t2np(st_pixel), axis=(1, 2)))
            test_labels.append(labels)
            gt_masks.append(t2np(mask).flatten())
            score_maps.append(t2np(st_pixel).flatten())

            if c.localize:
                localize(image, depth, st_pixel, labels, fg, mask, i)

    mean_st = np.concatenate(mean_st)
    max_st = np.concatenate(max_st)

    gt_masks = np.concatenate(gt_masks)
    score_maps = np.concatenate(score_maps)

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    mean_st_auc = roc_auc_score(is_anomaly, mean_st)
    max_st_auc = roc_auc_score(is_anomaly, max_st)
    pixel_auc = roc_auc_score(gt_masks, score_maps)
    PRO = compute_pro(gt_masks.reshape(-1,*mask_size), score_maps.reshape(-1,*mask_size))

    print('AUROC %\tmean over maps: {:.2f} \t max over maps: {:.2f} \t pixel: {:.2f} \t PRO: {:.2%}'.format(mean_st_auc * 100,
                                                                                             max_st_auc * 100,
                                                                                             pixel_auc * 100,
                                                                                             PRO))

    viz_roc(mean_st, is_anomaly, name='mean')
    viz_roc(max_st, is_anomaly, name='max')
    viz_roc(score_maps, gt_masks, name='pixel')

    compare_histogram(mean_st, is_anomaly, log=True, name='mean')
    compare_histogram(max_st, is_anomaly, log=True, name='max')

    return mean_st_auc, max_st_auc, pixel_auc, PRO


if __name__ == "__main__":
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(join(c.dataset_dir, d))]
    max_scores = list()
    mean_scores = list()
    pixel_scores = list()
    PRO_scores = list()
    for i_c, cn in enumerate(all_classes):
        c.class_name = cn
        print('\nEvaluate class ' + c.class_name)
        train_set, test_set = load_datasets(get_mask=True)
        _, test_loader = make_dataloaders(train_set, test_set)
        mean_sc, max_sc, pixel_sc, PRO = evaluate(test_loader)
        mean_scores.append(mean_sc)
        max_scores.append(max_sc)
        pixel_scores.append(pixel_sc)
        PRO_scores.append(PRO)
    
    with open(f"viz/{c.modelname}_result.txt", "w") as f:
        f.write("category\tmean I-AUROC\tmax I-AUROC\tP-AUROC\tPRO\n")
        for idx, cn in enumerate(all_classes):
            f.write(f"{cn}\t{mean_scores[idx]:.2%}\t{max_scores[idx]:.2%}\t{pixel_scores[idx]:.2%}\t{PRO_scores[idx]:.2%}\n")
        f.write(f"mean\t{np.mean(mean_scores):.2%}\t{np.mean(max_scores):.2%}\t{np.mean(pixel_scores):.2%}\t{np.mean(PRO_scores):.2%}\n")

    mean_scores = np.mean(mean_scores) * 100
    max_scores = np.mean(max_scores) * 100
    pixel_scores = np.mean(pixel_scores) * 100
    PRO_scores = np.mean(PRO_scores) * 100
    print('\nmean AUROC % over all classes\n\tmean over maps: {:.2f} \t max over maps: {:.2f} \t pixel: {:.2f} \t PRO: {:.2f}'.format(mean_scores,
                                                                                                      max_scores, pixel_scores, PRO_scores))
