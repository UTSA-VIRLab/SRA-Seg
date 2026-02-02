import argparse
import logging
import os
import random
import shutil
import sys
import time
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from torch.nn.modules.loss import CrossEntropyLoss

from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    TwoStreamBatchSampler,
    ThreeStreamBatchSampler,
)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

import timm


def load_dino_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    model.cuda()
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/FIVES', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum iteration number to pre-train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to self-train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether to use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
# Labeled and unlabeled settings:
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled batch size per gpu')
parser.add_argument('--labelnum', type=int, default=5, help='number of labeled patients')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency weight")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency rampup period")
parser.add_argument("--magnitude", type=float, default=6.0, help="magnitude")
parser.add_argument("--s_param", type=int, default=6, help="multiplier of random masks")
parser.add_argument("--use_sa", action="store_true", help="whether to use SA-based optimization loss")
parser.add_argument("--sa_weight", type=float, default=0.1, help="weight for the SA loss term")

args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=2)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state["net"])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state["net"])
    optimizer.load_state_dict(state["opt"])


def save_net_opt(net, optimizer, path):
    state = {"net": net.state_dict(), "opt": optimizer.state_dict()}
    torch.save(state, str(path))


def get_FIVES_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_FIVES_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()


def get_FIVES_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_FIVES_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {key: alpha * model_ema_state[key] + (1 - alpha) * model_state[key] for key in model_state}
    ema_model.load_state_dict(new_dict)


def soft_seg_loss(logits, gt_soft, eps=1e-6):
    pred = F.softmax(logits, dim=1).clamp(eps, 1 - eps)
    gt = gt_soft.clamp(eps, 1 - eps)
    # dice
    num = 2 * torch.sum(pred * gt, dim=(2, 3))
    den = torch.sum(pred + gt, dim=(2, 3)) + eps
    dice = 1 - torch.mean(num / den)
    # KL as CE with soft targets
    logp = pred.log()
    ce = -torch.sum(gt * logp, dim=1).mean()
    return dice + ce


# def generate_mask(img):
#     B, _, H, W = img.shape
#     m = torch.ones(H, W, device=img.device)
#     px, py = int(H * 2 / 3), int(W * 2 / 3)
#     w = np.random.randint(0, H - px)
#     h = np.random.randint(0, W - py)
#     m[w:w + px, h:h + py] = 0
#     alpha = F.avg_pool2d(m[None, None], 3, stride=1, padding=1).squeeze()
#     inv = 1 - m[None, None]
#     er = F.max_pool2d(inv, 3, stride=1, padding=1).squeeze()
#     loss_mask = (1 - er).long().unsqueeze(0).expand(B, H, W)
#     return m.long(), alpha, loss_mask, (w, h)

def generate_mask(img, hole_frac=2/3, kernel_size=7):
    B, _, H, W = img.shape
    m = torch.ones((H, W), device=img.device, dtype=torch.float32)
    hole_h, hole_w = int(H * hole_frac), int(W * hole_frac)
    w = np.random.randint(0, H - hole_h)
    h = np.random.randint(0, W - hole_w)
    m[w:w+hole_h, h:h+hole_w] = 0
    pad = kernel_size // 2
    alpha = F.avg_pool2d(
        m.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    ).squeeze(0).squeeze(0)
    inv = 1.0 - m
    er = F.max_pool2d(
        inv.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    ).squeeze(0).squeeze(0)
    loss_mask = (1.0 - er).long().unsqueeze(0).expand(B, H, W)
    return m.long(), alpha, loss_mask, (w, h)


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "FIVES" in dataset:
        ref_dict = {"5": 28, "10": 56, "100": 760}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def compute_sa_loss(real_imgs, unlabeled_imgs, feature_extractor):
    if hasattr(feature_extractor.patch_embed, "img_size"):
        target_size = feature_extractor.patch_embed.img_size
    else:
        target_size = (224, 224)
    real_small = F.interpolate(real_imgs, size=target_size, mode='bilinear', align_corners=False)
    unl_small = F.interpolate(unlabeled_imgs, size=target_size, mode='bilinear', align_corners=False)
    if real_small.shape[1] == 1:
        real_small = real_small.repeat(1, 3, 1, 1)
        unl_small = unl_small.repeat(1, 3, 1, 1)
    with torch.no_grad():
        f_real = feature_extractor(real_small)
    f_unlabeled = feature_extractor(unl_small)
    dist_matrix = torch.cdist(f_unlabeled, f_real, p=2)
    min_dists, _ = torch.min(dist_matrix, dim=1)
    return torch.mean(min_dists)


def pre_train(args, snapshot_path, feature_extractor=None):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, f"{args.model}_best_model.pth")
    labeled_sub_bs = int(args.labeled_bs / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print(f"Total slices is: {total_slices}, labeled slices is: {labeled_slice}")
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("Start pre_training")
    logging.info(f"{len(trainloader)} iterations per epoch")

    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs: args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs: args.labeled_bs]

            bin_mask, alpha, loss_mask, _ = generate_mask(img_a)
            oh_a = F.one_hot(lab_a.long(), num_classes).permute(0, 3, 1, 2).float()
            oh_b = F.one_hot(lab_b.long(), num_classes).permute(0, 3, 1, 2).float()

            α_img = alpha.unsqueeze(0).unsqueeze(0).expand_as(img_a)
            α_lbl = alpha.unsqueeze(0).unsqueeze(0).expand(-1, num_classes, -1, -1)

            net_input = img_a * α_img + img_b * (1 - α_img)
            gt_soft = oh_a * α_lbl + oh_b * (1 - α_lbl)

            out_mixl = model(net_input)
            loss = soft_seg_loss(out_mixl, gt_soft)

            if feature_extractor is not None and args.use_sa:
                sa_loss = compute_sa_loss(img_a, img_a, feature_extractor)
                loss += args.sa_weight * sa_loss
                writer.add_scalar("info/sa_loss", sa_loss.item(), iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar("info/total_loss", loss, iter_num)
            logging.info(f"iteration {iter_num}: loss: {loss}")

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image("pre_train/Mixed_Image", image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image("pre_train/Mixed_Prediction", outputs[1] * 50, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(f"info/val_{class_i + 1}_dice", metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f"info/val_{class_i + 1}_hd95", metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, f"iter_{iter_num}_dice_{round(best_performance, 4)}.pth"
                    )
                    save_best_path = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                logging.info(f"iteration {iter_num}: mean_dice: {performance}")
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path, feature_extractor=None):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, f"{args.model}_best_model.pth")
    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print(f"Total slices is: {total_slices}, labeled slices is: {labeled_slice}")
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info(f"Loaded from {pre_trained_model}")

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("Start self_training")
    logging.info(f"{len(trainloader)} iterations per epoch")

    model.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs: args.labeled_bs]
            uimg_a, uimg_b = (
                volume_batch[args.labeled_bs: args.labeled_bs + unlabeled_sub_bs],
                volume_batch[args.labeled_bs + unlabeled_sub_bs:],
            )
            ulab_a, ulab_b = (
                label_batch[args.labeled_bs: args.labeled_bs + unlabeled_sub_bs],
                label_batch[args.labeled_bs + unlabeled_sub_bs:],
            )
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs: args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_FIVES_masks(pre_a, nms=1)
                plab_b = get_FIVES_masks(pre_b, nms=1)
                bin_mask, alpha, loss_mask, _ = generate_mask(img_a)
                oh_la = F.one_hot(lab_a.long(), num_classes).permute(0, 3, 1, 2).float()
                oh_lb = F.one_hot(lab_b.long(), num_classes).permute(0, 3, 1, 2).float()
                pl_a = F.one_hot(plab_a.long(), num_classes).permute(0, 3, 1, 2).float()
                pl_b = F.one_hot(plab_b.long(), num_classes).permute(0, 3, 1, 2).float()

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            α_img = alpha.unsqueeze(0).unsqueeze(0).expand_as(img_a)
            α_lbl = alpha.unsqueeze(0).unsqueeze(0).expand(-1, num_classes, -1, -1)

            net_unl = uimg_a * α_img + img_a * (1 - α_img)
            gt_unl = pl_a * α_lbl + oh_la * (1 - α_lbl)
            net_lab = img_b * α_img + uimg_b * (1 - α_img)
            gt_lab = oh_lb * α_lbl + pl_b * (1 - α_lbl)

            out_unl = model(net_unl)
            out_lab = model(net_lab)
            loss = soft_seg_loss(out_unl, gt_unl) + soft_seg_loss(out_lab, gt_lab)

            if feature_extractor is not None and args.use_sa:
                sa_loss = compute_sa_loss(img_a, uimg_a, feature_extractor)
                loss += args.sa_weight * sa_loss
                writer.add_scalar("info/sa_loss", sa_loss.item(), iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            logging.info(f"iteration {iter_num}: loss: {loss}")

            if iter_num % 20 == 0:
                image = net_unl[1, 0:1, :, :]
                writer.add_image("train/Un_Image", image, iter_num)
                outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
                writer.add_image("train/Un_Prediction", outputs[1, ...] * 50, iter_num)
                gt_vis = torch.argmax(gt_unl, dim=1, keepdim=True) * 50  # B×1×H×W
                writer.add_image("train/Un_GroundTruth", gt_vis[1], iter_num)  # (1×H×W) OK for CHW

                image_l = net_lab[1, 0:1, :, :]
                writer.add_image("train/L_Image", image_l, iter_num)
                outputs_l = torch.argmax(torch.softmax(out_lab, dim=1), dim=1, keepdim=True)
                writer.add_image("train/L_Prediction", outputs_l[1, ...] * 50, iter_num)
                gt_vis_l = torch.argmax(gt_lab, dim=1, keepdim=True) * 50
                writer.add_image("train/L_GroundTruth", gt_vis_l[1], iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(f"info/val_{class_i + 1}_dice", metric_list[class_i, 0], iter_num)
                    writer.add_scalar(f"info/val_{class_i + 1}_hd95", metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path, f"iter_{iter_num}_dice_{round(best_performance, 4)}.pth"
                    )
                    save_best_path = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                logging.info(f"iteration {iter_num} : mean_dice : {performance}")
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = f"./model/BCP/FIVES_{args.exp}_{args.labelnum}_labeled/pre_train"
    self_snapshot_path = f"./model/BCP/FIVES_{args.exp}_{args.labelnum}_labeled/self_train"
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy("../code/FIVES_BCP_train.py", self_snapshot_path)

    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    feature_extractor = None
    if args.use_sa:
        logging.info("Loading DINO feature extractor for SA loss...")
        feature_extractor = load_dino_model()

    pre_train(args, pre_snapshot_path, feature_extractor=feature_extractor)

    logging.basicConfig(
        filename=self_snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path, feature_extractor=feature_extractor)