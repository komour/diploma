import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb
from torchvision import transforms

from MODELS.model_resnet import *
from custom_dataset import DatasetISIC2018
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

parser = argparse.ArgumentParser(description='PyTorch ResNet+CBAM ISIC2018 Visualization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--vis-prefix', type=str, default=None,
                    help='prefix to save plots e.g. "baseline" or "SAM-1"')
parser.add_argument('--run-name', type=str, default='noname run', help='run name on the W&B service')
parser.add_argument('--is-server', type=int, choices=[0, 1], default=1)
parser.add_argument("--tags", nargs='+', default=['default-tag'])
parser.add_argument('--cuda-device', type=int, default=0)
parser.add_argument('--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')

args = parser.parse_args()
is_server = args.is_server == 1


# make and save Grad-CAM plot (original image, mask, Grad-CAM, Grad-CAM++)
def make_plot_and_save(input_img, img_name, no_norm_image, segm, model, train_or_val, epoch=None, vis_prefix=None):
    global is_server
    # get Grad-CAM results and prepare them to show on the plot
    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer=target_layer)
    gradcam_pp = GradCAMpp(model, target_layer=target_layer)

    # sam_output shapes:
    # [1, 1, 56, 56]x3 , [1, 1, 28, 28]x4 [1, 1, 14, 14]x6 , [1, 1, 7, 7]x3
    mask, no_norm_mask, logit, sam_output = gradcam(input_img)

    sam1_show = torch.squeeze(sam_output[0].cpu()).detach().numpy()
    sam4_show = torch.squeeze(sam_output[3].cpu()).detach().numpy()
    sam8_show = torch.squeeze(sam_output[7].cpu()).detach().numpy()
    sam14_show = torch.squeeze(sam_output[13].cpu()).detach().numpy()

    heatmap, result = visualize_cam(mask, no_norm_image)

    result_show = np.moveaxis(torch.squeeze(result).detach().numpy(), 0, -1)

    mask_pp, no_norm_mask_pp, logit_pp, sam_output_pp = gradcam_pp(input_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, no_norm_image)

    result_pp_show = np.moveaxis(torch.squeeze(result_pp).detach().numpy(), 0, -1)

    # prepare mask and original image to show on the plot
    segm_show = torch.squeeze(segm.cpu()).detach().numpy()
    segm_show = np.moveaxis(segm_show, 0, 2)
    input_show = np.moveaxis(torch.squeeze(no_norm_image).detach().numpy(), 0, -1)

    # draw and save the plot
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(24, 9))
    plt.suptitle(f'{train_or_val}-Image: {img_name}')
    axs[1][0].imshow(segm_show)
    axs[1][0].set_title('Mask')
    axs[0][0].imshow(input_show)
    axs[0][0].set_title('Original Image')

    axs[0][1].imshow(result_show)
    axs[0][1].set_title('Grad-CAM')
    axs[1][1].imshow(result_pp_show)
    axs[1][1].set_title('Grad-CAM++')

    axs[1][2].imshow(sam1_show, cmap='gray')
    axs[1][2].set_title('SAM-1 relative')
    axs[0][2].imshow(sam1_show, vmin=0., vmax=1., cmap='gray')
    axs[0][2].set_title('SAM-1 absolute')

    axs[1][3].imshow(sam4_show, cmap='gray')
    axs[1][3].set_title('SAM-4 relative')
    axs[0][3].imshow(sam4_show, vmin=0., vmax=1., cmap='gray')
    axs[0][3].set_title('SAM-4 absolute')

    axs[1][4].imshow(sam8_show, cmap='gray')
    axs[1][4].set_title('SAM-8 relative')
    axs[0][4].imshow(sam8_show, vmin=0., vmax=1., cmap='gray')
    axs[0][4].set_title('SAM-8 absolute')

    axs[1][5].imshow(sam14_show, cmap='gray')
    axs[1][5].set_title('SAM-14 relative')
    axs[0][5].imshow(sam14_show, vmin=0., vmax=1., cmap='gray')
    axs[0][5].set_title('SAM-14 absolute')
    plt.show()
    if vis_prefix is not None:
        plt.savefig(f'vis/{vis_prefix}/{train_or_val}/{img_name}.png', bbox_inches='tight')
    if is_server:
        if epoch is not None:
            wandb.log({f'{train_or_val}/{img_name}': fig}, step=epoch)
        else:
            wandb.log({f'{train_or_val}/{img_name}': fig})


def main():
    global args, is_server
    if is_server:
        wandb.login()

    config = dict(
        vis_prefix=args.vis_prefix,
        resume=args.resume,
    )

    if is_server:
        wandb.init(config=config, project="vol.4", name=args.run_name, tags=args.tags)

    # define constants
    # vis_prefix = 'baseline'
    CLASS_AMOUNT = 5
    DEPTH = 50
    root_dir = 'data/'
    # resume = "checkpoints/baseline_checkpoint.pth"
    traindir = os.path.join(root_dir, 'train')
    train_labels = os.path.join(root_dir, 'train', 'images_onehot_train.txt')
    valdir = os.path.join(root_dir, 'val')
    val_labels = os.path.join(root_dir, 'val', 'images_onehot_val.txt')

    # define the model
    model = ResidualNet('ImageNet', DEPTH, CLASS_AMOUNT, 'CBAM')
    if is_server:
        model = model.cuda(args.cuda_device)

    # # load the checkpoint
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict)
        print(f"=> loaded checkpoint '{args.resume}'")
        print(f"epoch = {checkpoint['epoch']}")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
        return -1

    # define datasets and data loaders
    size0 = 224
    segm_dir = "images/256ISIC2018_Task1_Training_GroundTruth/"
    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        segm_dir,
        size0,
        False,  # perform flips
        False,  # perform random resized crop with size = 224
        transforms.CenterCrop(size0)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        pin_memory=True
    )

    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        segm_dir,
        size0,
        False,
        False,
        transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        pin_memory=True
    )

    # create directories to save plots
    if args.vis_prefix is not None:
        if not os.path.exists(f'vis/{args.vis_prefix}'):
            os.mkdir(f'vis/{args.vis_prefix}')

        if not os.path.exists(f'vis/{args.vis_prefix}/train'):
            os.mkdir(f'vis/{args.vis_prefix}/train')

        if not os.path.exists(f'vis/{args.vis_prefix}/val'):
            os.mkdir(f'vis/{args.vis_prefix}/val')

    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        img_name = dictionary['name'][0]
        no_norm_image = dictionary['no_norm_image']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
        make_plot_and_save(input_img, img_name, no_norm_image, segm, model, 'train', vis_prefix=args.vis_prefix)
        return

    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        img_name = dictionary['name'][0]
        no_norm_image = dictionary['no_norm_image']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
        make_plot_and_save(input_img, img_name, no_norm_image, segm, model, 'val', vis_prefix=args.vis_prefix)


if __name__ == '__main__':
    main()
