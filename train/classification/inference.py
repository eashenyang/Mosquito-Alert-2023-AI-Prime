import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
import shutil
from os import path as osp
from pathlib import Path
import json
from timm_predictor import TimmPredictor

predictor = TimmPredictor(model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', 
                          model_weights='/public/191-aiprime/jiawei.dong/projects/timm/output/train/20230720-204958-convnextv2_large_fcmae_ft_in22k_in1k_384-384/cls_mos_7881_fold0.pth',
                          device=7, 
                          new_shape=384, 
                          num_classes=6)

img_dir = '/public/share/challenge_dataset/MosquitoAlert2023/test_sub_images_yzj_best'
output_path = '/public/191-aiprime/jiawei.dong/projects/mosquito/cls_result_v6_convnextl_extra_v3_in_tta_yzj_det_last.json'

img_list = glob.glob(img_dir+ '/*.jpeg')

n_img = len(img_list)
batch_size = 2

pred_tta = []
for degree in [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
# for degree in [0, cv2.ROTATE_90_CLOCKWISE]:
    pred_dict = {}
    for img_idx in tqdm(range(0, len(img_list), batch_size)):
        imgs = []
        _batch = min(n_img - img_idx, batch_size)
        for j in range(img_idx, img_idx + _batch):
            img_path = img_list[j]
            img_name = osp.basename(img_path)
            img_date = osp.basename(osp.dirname(img_path))
            try:
                # img = cv2.imread(img_path)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (384, 384))
                if degree != 0:
                    img = cv2.rotate(img, degree)
                imgs.append(img)
            except:
                print(f'error at {img_path}')
                continue
        
        preds = predictor.inference(imgs)

        
        for j, p in enumerate(preds):
            img_path = img_list[img_idx + j]
            img_name = osp.basename(img_path)
            # if p[2] > 0.8:
            #     img_path = img_list[img_idx + j]
            #     img_name = osp.basename(img_path)
            #     new_path = osp.join(output_dir, img_name)
            #     shutil.copy(img_path, new_path)

            # label = str(int(np.argmax(p)))
            score = p
            pred_dict[img_name] = score
    pred_tta.append(pred_dict)


new_pred = {}
for pred in pred_tta:
    for img_name, score in pred.items():
        if img_name not in new_pred:
            new_pred[img_name] = score
        else:
            new_pred[img_name] += score

for img_name, score in new_pred.items():
    new_pred[img_name] = str(int(np.argmax(score)))


with open(output_path, 'w') as f:
    json.dump(new_pred, f)