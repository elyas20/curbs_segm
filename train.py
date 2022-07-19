import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.dataset import BasicDataset, CurbsDataset
from utils import  VisdomLinePlotter, set_bn_momentum
from loss import FocalLoss, dice_loss

from evaluate import evaluate
from models.unet_model import UNet
from models.DUNET.deform_unet import DUNetV1V2
from models import network


dir_img = Path('./data/rgb_images/rgb_images/')
dir_img_test = Path('./data/rgb_images/rgb_images_test/')

dir_mask = Path('./data/curb_annotations/curb_annotations/gtLabels/')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              env:str ="training_unet"):

    plotter = VisdomLinePlotter(env_name=env)

    # 1. Create dataset
    try:
        dataset = CurbsDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


    scheduler = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=1e-4, amsgrad=False)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_mse = nn.MSELoss(reduction='sum')
    focal_crit = FocalLoss()
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # loss = loss_mse(masks_pred, F.one_hot(
                    #     true_masks, net.n_classes).permute(0, 3, 1, 2).float())
                    # loss = focal_crit(masks_pred, true_masks)
                    loss = criterion(masks_pred, true_masks)  + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        val_score, val_epoch_loss = evaluate(net, val_loader, device)
        scheduler.step()
        plotter.plot('loss', 'train', 'Class Loss', epoch, epoch_loss/len(train_loader))
        plotter.plot('loss', 'val', 'Class Loss', epoch,
                     val_epoch_loss/len(val_loader))
        plotter.plot('acc', 'val', 'Class Accuracy', epoch, val_score.cpu())
        plotter.plot('learning rate', 'learning rate',
                     'learning rate', epoch, optimizer.param_groups[0]['lr'])


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--env_name', default="segmentation training", help='Visdom env name')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16])
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")


    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models+["unet", "dunet"], help='model name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    if (args.model=="unet"):
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif(args.model=="dunet"):
        net = DUNetV1V2(n_channels=3, n_classes=args.classes,
                        downsize_nb_filters_factor=4)
    else:
        net = network.modeling.__dict__[args.model](num_classes=args.classes, output_stride=args.output_stride)
        if args.separable_conv and 'plus' in args.model:
            network.convert_to_separable_conv(net.classifier)
        set_bn_momentum(net.backbone, momentum=0.01)
    logging.info(f'Network:\n'
                f'\t{args.model} model type\n'
                 f'\t{3} input channels\n'
                 f'\t{args.classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  env=args.env_name)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
