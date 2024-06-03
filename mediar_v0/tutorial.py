import torch
import glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor, EnsemblePredictor

project_path = '/Users/glue/work/ImagInProVision/ImagInVisionPro_glue/mediar_v0'

# MEDIAR uses two pretrained models to conduct ensemble prediction.
pretrained_model_path1 = project_path +'/weights/pretrained/phase1.pth'
pretrained_model_path2 = project_path +'/weights/pretrained/phase1.pth'

preterained_weights1 = torch.load(pretrained_model_path1, map_location='cpu')
preterained_weights2 = torch.load(pretrained_model_path2, map_location='cpu')


# Load weights on the our MEDIARFormer model.
# MEDIARFormer predicts 3-dimensional outputs (3 classes), where each corresponds to:
# Cell Recognition: Predicts whether a pixel belongs to cell (1) or background (0)
# Cell Distinction: Horizontal Vector & Vertical Vector to the cell center.

model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
}

model1 = MEDIARFormer(**model_args)
model1.load_state_dict(preterained_weights1, strict=False)

model2 = MEDIARFormer(**model_args)
model2.load_state_dict(preterained_weights2, strict=False)

input_img_path = project_path + '/input_examples'
img1 = io.imread(input_img_path +'/img1.tiff')
io.imshow(img1)

img2 = io.imread(input_img_path +'/img2.tif')
io.imshow(img2)

# Let's detect the cell instances in the image using a single MEDIARFormer model.
# The Predictor conduct its pediction on all images in the input_path and save the results in the output_path.
# In this example, we do not use test-time Augmentation by setting use_tta as False.
output_path = project_path + '/results'

predictor = Predictor(model1, 'cpu', input_img_path, output_path, algo_params={"use_tta": False})
_ = predictor.conduct_prediction()

# Show the results
pred1 = io.imread(output_path + '/img1_label.tiff')
io.imshow(pred1, cmap="cividis")

cell_count = len(np.unique(pred1))-1 # exclude the background
print(f"\n{cell_count} Cells detected!")

pred2 = io.imread(output_path + '/img1_label.tiff')
io.imshow(pred2, cmap="cividis")

cell_count = len(np.unique(pred2))-1 # exclude the background
print(f"\n{cell_count} Cells detected!")

# Let's use the ensemble models with TTA.
# In this example, we use test-time Augmentation by setting use_tta as True.
# It takes much longer, as it need to conduct multiple forward paths.

predictor = EnsemblePredictor(model1, model2, 'cpu', input_img_path, output_path, algo_params={"use_tta": True})
_ = predictor.conduct_prediction()

# Show the results
pred1 = io.imread(output_path + '/img1_label.tiff')
io.imshow(pred1, cmap="cividis")

pred1.shape

cell_count = len(np.unique(pred1))-1 # exclude the background
print(f"\n{cell_count} Cells detected!")

pred2 = io.imread(output_path + '/img2_label.tiff')
io.imshow(pred2, cmap="cividis")

cell_count = len(np.unique(pred2))-1 # exclude the background
print(f"\n{cell_count} Cells detected!")
