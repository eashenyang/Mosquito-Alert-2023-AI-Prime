import timm
#############################################
# @File    :   yolov7_predictor.py
# @Version :   1.0
# @Author  :   JiaweiDong
# @Time    :   2022/10/10 Mon  
# @Desc    :   
#############################################
import time
import os
import cv2
from copy import deepcopy
import sys
sys.path.append("prime_models")
from collections import OrderedDict, defaultdict
import torch
import timm
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
# from train_hybird import ModelMultiHead
# from memory_profiler import profile
from tqdm import tqdm
class TimmPredictor:
    def __init__(self, model_name,
                 model_weights, 
                 device='cuda:0', 
                 new_shape=1024, 
                 num_classes=2, 
                 mode='regression',
                 multi_head=False):
        self.new_shape = new_shape
        self.multi_head = multi_head
        self.mode = mode

        if self.mode == 'ordinal':
            assert num_classes != 1

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"[TIMM CLS Model] --------- Load from: {model_weights} ---------")
        logger.info(f"[TIMM CLS Model] --------- DEVICES: {device} ---------")
        logger.info(f"[TIMM CLS Model] --------- SHAPE: {self.new_shape} ---------")
        # logger.info('-------------weights: ',weights)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        # if self.mode == 'ordinal' or self.mode == 'corn' or self.mode == 'features':
        #     self.model = ModelMultiHead(model, mode=self.mode, cross_attention=False, multi_head=multi_head)
        # if self.mode == 'features':
            # self.model.head = None

        state = torch.load(model_weights, map_location='cpu')
        if 'state_dict_ema' in state:
            logger.debug('EMA Model Found, Using EMA...')
            state = state['state_dict_ema']
        elif 'state_dict' in state:
            logger.debug('EMA Model Not Found, Using Normal...')
            state = state['state_dict']
        else:
            logger.error('No State Dict Found...')
        state = self.clean_state_dict(state)
        self.model.load_state_dict(state, strict=False)
        # self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("[TIMM CLS Model] --------- Load Success ---------")

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize([self.new_shape,self.new_shape]),
            transforms.Resize(self.new_shape),
            # transforms.CenterCrop(320), 
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def clean_state_dict(self, state_dict):
        # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[12:] if k.startswith('module.base.') else k
            # name = name.replace('fc.', 'model.head.')
            # if 'model' not in name:
            #     name = 'model.' + name
            cleaned_state_dict[name] = v
        return cleaned_state_dict

    def preprocess_single(self, img):
        img = img[:, :, ::-1]
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        return img.to(self.device)

    #@profile
    def preprocess_multi(self, imgs):
        transformed_imgs = []
        for img in imgs:
            img = img[:, :, ::-1]
            ti = self.transform(img)
            # ti = ti.type(torch.cuda.FloatTensor)
            transformed_imgs.append(ti)
        transformed_imgs = torch.stack(transformed_imgs)
        return transformed_imgs.to(self.device)

    def preprocess_pair(self, imgs, imgs_head):
        transformed_imgs = []
        for img, img_head in zip(imgs, imgs_head):
            img = img[:, :, ::-1]
            ti = self.transform(img)
            # ti = ti.type(torch.cuda.FloatTensor)
            img_head = img_head[:, :, ::-1]
            tih = self.transform(img_head)

            img_6c = torch.cat((ti, tih), dim=0)

            transformed_imgs.append(img_6c)

        transformed_imgs = torch.stack(transformed_imgs)
        return transformed_imgs.to(self.device)

    #@profile
    def inference(self, imgs):
        imgs = [imgs for _ in range(16)]
        if isinstance(imgs, list):
            imgs = self.preprocess_multi(imgs)
        elif isinstance(imgs, np.ndarray):
            imgs = self.preprocess_single(imgs)
        else:
            raise NotImplementedError('This dimension is not implemented.')
        durations = 0
        print('starting inference.....')
        for i in range(10):
            with torch.no_grad():
                # print(imgs.size())
                start_time = time.time()
                preds = self.model(imgs)
                preds = torch.softmax(preds, dim=1)
                end_time = time.time()
                duration = (end_time-start_time) / 16 * 1000
                durations += duration
                print(f'{i}th inference time is : {duration} ms')
        print(f"average inference time is {durations/10} ms")
        # return duration
        preds = self.postprocess(preds)
        # # print(preds)
        return preds
    
    def inference_regression(self, imgs, preprocess=True):
        if preprocess:
            if isinstance(imgs, list):
                imgs = self.preprocess_multi(imgs)
            elif isinstance(imgs, np.ndarray):
                imgs = self.preprocess_single(imgs)
            else:
                raise NotImplementedError('This dimension is not implemented.')
        with torch.no_grad():
            # print(imgs.size())
            
            if self.mode == 'regression':
                preds = self.model(imgs)
                preds = torch.sigmoid(preds) * 191
                # preds = (preds * 190) + 56.61900934391641
                # preds = preds * 37.43000203163268 + 56.61900934391641
            elif self.mode == 'ordinal':
                if not self.multi_head:
                    logits, probas = self.model(imgs)
                else:
                    logits, logits2, probas = self.model(imgs)
                predict_levels = probas > 0.5
                preds = torch.sum(predict_levels, dim=1)

                if self.multi_head:
                    m = nn.Softmax(dim=1)
                    p = m(logits2)
                    a = torch.arange(1, 192, dtype=torch.float32).to(self.device)
                    preds2 = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
                    preds = torch.stack([preds, preds2]).mean(dim=0)
            elif self.mode == 'corn':
                logits, probas = self.model(imgs)
                probas = torch.cumprod(probas, dim=1)
                predict_levels = probas > 0.5
                preds = torch.sum(predict_levels, dim=1)

        preds = self.postprocess(preds)
        # print(preds)
        return preds

    def inference_features(self, imgs):
        if isinstance(imgs, list):
            imgs = self.preprocess_multi(imgs)
        elif isinstance(imgs, np.ndarray):
            imgs = self.preprocess_single(imgs)
        else:
            raise NotImplementedError('This dimension is not implemented.')
        with torch.no_grad():
            # preds = self.model.forward_features(imgs)
            # preds = preds[:, self.model.model.num_prefix_tokens:].mean(dim=1)
            # preds = self.model.model.fc_norm(preds)
            # preds = self.model.forward_head(preds, pre_logits=True)
            preds = self.model(imgs)
        preds = self.postprocess(preds)
        # print(preds)
        return preds

    def inference_multi_input(self, imgs, imgs_head):
        img_all = self.preprocess_pair(imgs, imgs_head)
        with torch.no_grad():
            logits, probas = self.model(img_all)
            predict_levels = probas > 0.5
            preds = torch.sum(predict_levels, dim=1)
        preds = self.postprocess(preds)
        return preds
    
    def inference_nohead(self, imgs):
        if isinstance(imgs, list):
            imgs = self.preprocess_multi(imgs)
        elif isinstance(imgs, np.ndarray):
            imgs = self.preprocess_single(imgs)
        else:
            raise NotImplementedError('This dimension is not implemented.')
        with torch.no_grad():
            # print(imgs.size())
            preds = self.model.forward_features(imgs)
            preds = self.pool(preds).squeeze()
        #     preds = torch.softmax(preds, dim=1)
        # preds = self.postprocess(preds)
        # print(preds)
        return preds

    def inference_sigmoid(self, imgs):
        if isinstance(imgs, list):
            imgs = self.preprocess_multi(imgs)
        elif isinstance(imgs, np.ndarray):
            imgs = self.preprocess_single(imgs)
        else:
            raise NotImplementedError('This dimension is not implemented.')
        with torch.no_grad():
            # print(imgs.size())
            preds = self.model(imgs)
            preds = torch.sigmoid(preds)
        preds = self.postprocess(preds)
        # print(preds)
        return preds

    def postprocess(self, preds):   
        preds = preds.detach().cpu().numpy()
        if preds.shape[0] > 1:
            preds = np.mean(preds, axis=0)
        return preds

if __name__ == '__main__':
    image = cv2.imread('/public/191-aiprime/jiawei.dong/dataset/lifejacket/20230202_nc4_shfb/labeled/0/fubao_cam792_20230201_152047_535522.jpg')
    model_name = 'ecaresnet50t'
    model_weight = 'output/train/20230613-160146-ecaresnet50t-128/model_best.pth.tar'

    # base_line_name = 
    # baseline_weight = '/public/191-aiprime/jiawei.dong/projects/timm/output/train/20230613-160146-ecaresnet50t-128/model_best.pth-2ae8a615.pth'
    shape = (256,128)
    num_class = 4
    predictor = TimmPredictor(model_name=model_name, model_weights=model_weight, new_shape=shape, num_classes=num_class)
    predictor.inference(image)