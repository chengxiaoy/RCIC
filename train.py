import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR

from torchvision import models, transforms as T
import torchvision

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
import sys
from rcic_data import *
from model import get_basic_model
import joblib

# sys.path.append('rxrx1-utils')
# import rxrx.io as rio

warnings.filterwarnings('ignore')

path_data = 'data'
# device = 'cuda'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 48
torch.manual_seed(0)
use_rgb = False
model_name = 'densenet201'
classes = 1108
pic_size = 512
experiment_name = str(use_rgb) + "_" + str(batch_size) + "_" + str(
    pic_size) + "_" + model_name + "_" + datetime.now().strftime('%b%d_%H-%M')

ds, ds_val, ds_test = get_dataset(use_rgb, size=pic_size)

model = get_basic_model(model_name, use_rgb)

model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# model.load_state_dict(torch.load('models/Model_False_24_512_densenet201_Aug07_13-01_35_val_acc=0.526287.pth'))
# model = model.module

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=16)
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=16)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

val_epoch = {}


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    val_epoch[epoch] = metrics
    print("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch,
                  metrics['loss'],
                  metrics['accuracy']))


# lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)


#
#
# lr_scheduler = MultiStepLR(optimizer, [15, 25, 100], 0.1)


@trainer.on(Events.EPOCH_COMPLETED)
def update_lr_scheduler(engine):
    # lr_scheduler.step()
    lr_scheduler.step(val_epoch[engine.state.epoch]['accuracy'])
    # lr_scheduler.step(engine.state.metrics['accuracy'])
    lr = float(optimizer.param_groups[0]['lr'])
    print("Learning rate: {}".format(lr))


# handler = EarlyStopping(patience=6, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
# val_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_STARTED)
def turn_on_layers(engine):
    epoch = engine.state.epoch
    # if epoch == 1:
    #     for name, child in model.named_children():
    #         if name == 'fc':
    #             pbar.log_message(name + ' is unfrozen')
    #             for param in child.parameters():
    #                 param.requires_grad = True
    #         else:
    #             pbar.log_message(name + ' is frozen')
    #             for param in child.parameters():
    #                 param.requires_grad = False
    if epoch == 1:
        pbar.log_message("Turn on all the layers")
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True


# checkpoints = ModelCheckpoint('models', 'Model', save_interval=3, n_saved=3, create_dir=True, require_empty=False)
checkpoints = ModelCheckpoint('models', 'Model', score_function=lambda engine: engine.state.metrics['accuracy'],
                              score_name='val_acc', n_saved=3, create_dir=True, require_empty=False)
val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {experiment_name: model})

pbar = ProgressBar(bar_format='')
# pbar.attach(trainer, output_transform=lambda x: {'loss': x})

import os

if not 'KAGGLE_WORKING_DIR' in os.environ:  # If we are not on kaggle server
    from ignite.contrib.handlers.tensorboard_logger import *

    tb_logger = TensorboardLogger("board/" + experiment_name)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag="validation", metric_names=["accuracy", "loss"],
                                                              another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)

    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.close()

trainer.run(loader, max_epochs=40)

model.eval()

preds = np.empty(0)
confi = np.empty(0)
with torch.no_grad():
    for x, _ in tqdm(tloader):
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        confidence = output.max(dim=-1)[0].cpu().numpy()
        preds = np.append(preds, idx, axis=0)
        confi = np.append(confi, confidence, axis=0)

joblib.dump([confi, preds], "res.pkl")

true_idx = np.empty(0)
for i in range(19897):
    if confi[i] > confi[i + 19897]:
        true_idx = np.append(true_idx, preds[i])
    else:
        true_idx = np.append(true_idx, preds[i + 19897])

submission = pd.read_csv('data/test.csv')
submission['sirna'] = true_idx.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code', 'sirna'])

# submission = pd.read_csv(path_data + '/test.csv')
# submission['sirna'] = preds.astype(int)
# submission.to_csv('submission.csv', index=False, columns=['id_code', 'sirna'])
