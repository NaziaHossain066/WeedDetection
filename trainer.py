import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.cropandweed_dataset_2 import CustomDataset, RandomGenerator  # Import your CustomDataset


# Custom collate function to handle None values in the batch
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Worker initialization function to ensure reproducibility
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


# Validation function to evaluate model on the validation set
'''def validate(model, val_loader, ce_loss, dice_loss, num_classes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            total_loss += 0.4 * loss_ce + 0.6 * loss_dice
    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss.item()'''

def validate(model, val_loader, ce_loss, dice_loss, num_classes, writer, epoch_num):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass
            outputs = model(image_batch)

            # Calculate loss
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            total_loss += 0.4 * loss_ce + 0.6 * loss_dice

            # Log images once per epoch (after completing the entire validation loop)
            if epoch_num % 1 == 0:  # This ensures that images are logged at the end of every epoch
                # Normalize the image batch (adjust normalization as needed)
                image = (image_batch[0, ...] - image_batch[0, ...].min()) / (image_batch[0, ...].max() - image_batch[0, ...].min())

                # Add the image to TensorBoard
                writer.add_image('validation/Image', image, epoch_num, dataformats='CHW')

                # Process outputs (get predictions)
                outputs_pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # Get predicted class indices
                outputs_pred = (outputs_pred[0, ...] * 50).unsqueeze(0)  # Scale for visibility and add channel dimension (1, H, W)

                # Process ground truth labels
                labs = label_batch[0, ...].unsqueeze(0) * 50  # Add channel dimension and scale for visibility

                # Add predictions and ground truth to TensorBoard
                writer.add_image('validation/Prediction', outputs_pred, epoch_num, dataformats='CHW')
                writer.add_image('validation/GroundTruth', labs, epoch_num, dataformats='CHW')

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss.item()


def trainer_custom(args, model, snapshot_path, val_path=None):
    # Logging setup
    logging.basicConfig(filename=snapshot_path + f"/{args.config_file}_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Hyperparameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Dataset and Dataloader setup
    db_train = CustomDataset(root_dir=args.root_path, transform=transforms.Compose(
                             [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)

    if val_path:
        db_val = CustomDataset(root_dir=val_path, transform=transforms.Compose(
                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                               worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)

    # Resume training if a checkpoint is provided
    start_epoch = 0
    iter_num = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resumed training from checkpoint: {args.resume} at epoch {start_epoch}")

    # Multi-GPU handling
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Loss functions and optimizer
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # TensorBoard Writer
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    # Training loop
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass and loss calculation
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # Logging losses
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(f"Iteration {iter_num}: Loss: {loss.item()}, Loss_CE: {loss_ce.item()}")

        # Periodic validation
        if epoch_num % 2 == 0 and val_path:
            val_loss = validate(model, valloader, ce_loss, dice_loss, num_classes, writer, epoch_num)
            logging.info(f"Validation loss at epoch {epoch_num}: {val_loss}")
            writer.add_scalar('validation/loss', val_loss, iter_num)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(snapshot_path, f"{args.config_file}_best_model.pth")
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, best_model_path)
                logging.info(f"Saved new best model to {best_model_path}")

        # Save periodic checkpoints
        if (epoch_num + 1) % 10 == 0:
            checkpoint_path = os.path.join(snapshot_path, f"{args.config_file}_epoch_{epoch_num}.pth")
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            logging.info(f"Saved model checkpoint to {checkpoint_path}")

        # End training after the final epoch
        if epoch_num >= max_epoch - 1:
            final_model_path = os.path.join(snapshot_path, f"{args.config_file}_final_model.pth")
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }, final_model_path)
            logging.info(f"Saved final model to {final_model_path}")
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
    
    

'''import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.cropandweed_dataset_2 import CustomDataset, RandomGenerator  # Import your CustomDataset


# Custom collate function to handle None values in the batch
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Worker initialization function to ensure reproducibility
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


# Validation function to evaluate model on the validation set
def validate(model, val_loader, ce_loss, dice_loss, num_classes):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            total_loss += 0.4 * loss_ce + 0.6 * loss_dice
    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss.item()

def validate(model, val_loader, ce_loss, dice_loss, num_classes, writer, epoch_num):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sampled_batch in val_loader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass
            outputs = model(image_batch)

            # Calculate loss
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            total_loss += 0.4 * loss_ce + 0.6 * loss_dice

            # Log images once per epoch (after completing the entire validation loop)
            if epoch_num % 1 == 0:  # This ensures that images are logged at the end of every epoch
                # Normalize the image batch (adjust normalization as needed)
                image = (image_batch[0, ...] - image_batch[0, ...].min()) / (image_batch[0, ...].max() - image_batch[0, ...].min())

                # Add the image to TensorBoard
                writer.add_image('validation/Image', image, epoch_num, dataformats='CHW')

                # Process outputs (get predictions)
                outputs_pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # Get predicted class indices
                outputs_pred = (outputs_pred[0, ...] * 50).unsqueeze(0)  # Scale for visibility and add channel dimension (1, H, W)

                # Process ground truth labels
                labs = label_batch[0, ...].unsqueeze(0) * 50  # Add channel dimension and scale for visibility

                # Add predictions and ground truth to TensorBoard
                writer.add_image('validation/Prediction', outputs_pred, epoch_num, dataformats='CHW')
                writer.add_image('validation/GroundTruth', labs, epoch_num, dataformats='CHW')

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss.item()


# Main trainer function
def trainer_custom(args, model, snapshot_path, val_path=None):
    # Logging setup
    logging.basicConfig(filename=snapshot_path + f"/{args.config_file}_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Hyperparameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Dataset and Dataloader setup
    db_train = CustomDataset(root_dir=args.root_path, transform=transforms.Compose(
                             [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)

    if val_path:
        db_val = CustomDataset(root_dir=val_path, transform=transforms.Compose(
                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                               worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)

    # Multi-GPU handling
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Loss functions and optimizer
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # TensorBoard Writer
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # Training loop
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass and loss calculation
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # Logging losses
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(f"Iteration {iter_num}: Loss: {loss.item()}, Loss_CE: {loss_ce.item()}")

            # Visualizations in TensorBoard
            if iter_num % 20 == 0:
                image = (image_batch[0, 0:1, :, :] - image_batch.min()) / (image_batch.max() - image_batch.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num % 20 == 0:
                # Normalize the image batch (adjust normalization as needed)
                image = (image_batch[0, ...] - image_batch[0, ...].min()) / (image_batch[0, ...].max() - image_batch[0, ...].min())

                # Add the image to TensorBoard
                writer.add_image('train/Image', image, iter_num, dataformats='CHW')

                # Process outputs (get predictions)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # Get predicted class indices
                outputs = (outputs[0, ...] * 50).unsqueeze(0)  # Scale for visibility and add channel dimension (1, H, W)

                # Process ground truth labels
                labs = label_batch[0, ...].unsqueeze(0) * 50  # Add channel dimension and scale for visibility

                # Add predictions and ground truth to TensorBoard
                writer.add_image('train/Prediction', outputs, iter_num, dataformats='CHW')
                writer.add_image('train/GroundTruth', labs, iter_num, dataformats='CHW')

        # Periodic validation
        if epoch_num % 2 == 0 and val_path:
            val_loss = validate(model, valloader, ce_loss, dice_loss, num_classes, writer, epoch_num)
            logging.info(f"Validation loss at epoch {epoch_num}: {val_loss}")
            writer.add_scalar('validation/loss', val_loss, iter_num)

    

        # Save checkpoints
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, f"{args.config_file}_epoch_{epoch_num}.pth")
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model checkpoint to {save_mode_path}")

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f"{args.config_file}_epoch_{epoch_num}.pth")
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved final model to {save_mode_path}")
            iterator.close()
            break

    writer.close()
    return "Training Finished!"'''