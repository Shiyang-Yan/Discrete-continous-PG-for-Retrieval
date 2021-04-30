from vocab import Vocabulary
import visual_att_2
#visual_att_2.evalrank('runs/runX_coco_test/model_best.pth.tar', data_path='data/data/data', split="testall", fold5=False)
import torch
from vocab import Vocabulary
import evaluation_models
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# for coco
print('Evaluation on COCO:')
evaluation_models.evalrank('runs/runX_f30_rebuttal2/model_best.pth.tar', "runs/runX_30k_post_rebuttal/model_best.pth.tar", data_path='data/data/data/', split="test", fold5=False)# for flickr
#print('Evaluation on Flickr30K:')
#evaluation_models.evalrank("runs/runX_f30_rebuttal2/model_best.pth.tar", "runs/runX_f30_rebuttal_glove/model_best.pth.tar", data_path='data/data/data/', split="test", fold5=False)