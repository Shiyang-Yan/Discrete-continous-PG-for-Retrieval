from vocab import Vocabulary
#import evaluation
#evaluation.evalrank('runs/runX_f30_discrete/model_best.pth.tar', data_path='data/data/data', split="test", fold5=False)
import torch
from vocab import Vocabulary
import evaluation_models

# for coco
#print('Evaluation on COCO:')
#evaluation_models.evalrank('runs/runX_f30_rank/model_best.pth.tar', "runs/runX_f30_500_20/model_best.pth.tar", data_path='data/data/data/', split="test", fold5=False)

# for flickr
#print('Evaluation on Flickr30K:')
#evaluation_models.evalrank("runs/flickSRN/pretrain_model/pretrain_model/flickr/model_fliker_2.pth.tar", "runs/flickr_VSRN/pretrain_model/pretrain_model/flickr/model_fliker_2.pth.tar", data_path='data/data/data/', split="test", fold5=False)