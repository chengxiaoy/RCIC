import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import models
from train import ImagesDS

path_data = 'data'
ds_test = ImagesDS(df_test, path_data, mode='test')

model = models.resnet34(pretrained=True)
model.load_state_dict(torch.load('models/Model_ResNet34_45.pth'))

model.eval()
with torch.no_grad():
    preds = np.empty(0)
    for x, _ in tqdm(tloader):
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        preds = np.append(preds, idx, axis=0)
submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code', 'sirna'])