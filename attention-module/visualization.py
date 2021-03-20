import torch
import argparse
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
import os
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import wandb

from MODELS.model_resnet import *
from custom_dataset import DatasetISIC2018

from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ResNet+CBAM ISIC2018 Visualization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--vis-prefix', type=str, default='dummy-prefix',
                    help='prefix to save plots e.g. "baseline" or "SAM-1"')
parser.add_argument('--run-name', type=str, default='noname run', help='run name on the W&B service')
parser.add_argument('--is-server', type=int, choices=[0, 1], default=1)
parser.add_argument("--tags", nargs='+', default=['default-tag'])
parser.add_argument('--cuda-device', type=int, default=0)


args = parser.parse_args()
is_server = args.is_server == 1


# make and save Grad-CAM plot (original image, mask, Grad-CAM, Grad-CAM++)
def make_plot_and_save(input_img, img_name, no_norm_image, segm, model, vis_prefix, train_or_val):
    global is_server
    # get Grad-CAM results and prepare them to show on the plot
    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer=target_layer)
    gradcam_pp = GradCAMpp(model, target_layer=target_layer)

    mask, _, sam_output = gradcam(input_img)

    sam1_show = torch.squeeze(sam_output[0].cpu()).detach().numpy()
    sam4_show = torch.squeeze(sam_output[3].cpu()).detach().numpy()
    sam8_show = torch.squeeze(sam_output[7].cpu()).detach().numpy()
    sam14_show = torch.squeeze(sam_output[13].cpu()).detach().numpy()

    heatmap, result = visualize_cam(mask, no_norm_image)

    result_show = np.moveaxis(torch.squeeze(result).detach().numpy(), 0, -1)

    mask_pp, _ = gradcam_pp(input_img)
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
    plt.savefig(f'vis/{vis_prefix}/{train_or_val}/{img_name}.png', bbox_inches='tight')
    if is_server:
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
    # if os.path.isfile(args.resume):
    #     print(f"=> loading checkpoint '{args.resume}'")
    #     checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    #     state_dict = checkpoint['state_dict']
    #
    #     model.load_state_dict(state_dict)
    #     print(f"=> loaded checkpoint '{args.resume}'")
    #     print(f"epoch = {checkpoint['epoch']}")
    # else:
    #     print(f"=> no checkpoint found at '{args.resume}'")
    #     return -1
    
    # create dummy layer to init weights in the state_dict
    dummy_fc = torch.nn.Linear(512 * 4, CLASS_AMOUNT)
    torch.nn.init.xavier_uniform_(dummy_fc.weight)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']

            state_dict['module.fc.weight'] = dummy_fc.weight
            state_dict['module.fc.bias'] = dummy_fc.bias

            # remove `module.` prefix because we don't use torch.nn.DataParallel

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            # model.load_state_dict(state_dict)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}'")
            # print(f"epoch = {checkpoint['epoch']}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return -1

    # define datasets and data loaders
    size0 = 224
    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        False,  # perform flips
        False,  # perform random resized crop with size = 224
        transforms.CenterCrop(size0)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, shuffle=False,
        pin_memory=True
    )

    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        False,
        False,
        transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        pin_memory=True
    )

    # create directories to save plots
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
        make_plot_and_save(input_img, img_name, no_norm_image, segm, model, args.vis_prefix, 'train')

    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        img_name = dictionary['name'][0]
        no_norm_image = dictionary['no_norm_image']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
        make_plot_and_save(input_img, img_name, no_norm_image, segm, model, args.vis_prefix, 'val')


if __name__ == '__main__':
    main()
