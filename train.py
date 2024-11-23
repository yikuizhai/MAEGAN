import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import numpy as np
import argparse
import random
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)
from tqdm import tqdm
from PIL import Image
from dropout_models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data[0], rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data[0], rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data[0], part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        

def train(args):
    # Fix random seed
    SEED = 2107
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = args.start_iter
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)

    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    txt_write = open(log_dir + '/' + 'flagfile.txt', "a")
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:3")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    # compute mpv (mean pixel value) of training dataset
    if args.mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(os.listdir(args.path)),
            desc='computing mean pixel value of training dataset...')
        for imgpath in os.listdir(args.path):
            img = Image.open(args.path + '/' + imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(dataset)
        pbar.close()
    else:
        mpv = np.array(args.mpv)
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(device)
    alpha = torch.tensor(
        args.alpha,
        dtype=torch.float32).to(device)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(args.batch_size, nz).normal_(0, 1).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    # optimizerD = torch.optim.SGD([p for p in netD.parameters() if p.requires_grad], args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        print("model has been loaded, start from %d epoch" % current_iteration)
        del ckpt

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        mask_real = gen_input_mask(
            shape=(real_image.shape[0], 1, real_image.shape[2], real_image.shape[3]),
            hole_size=(
                (args.hole_min_w, args.hole_max_w),
                (args.hole_min_h, args.hole_max_h)),
            hole_area=None,
            max_holes=args.max_holes,
        ).to(device)

        real_image_mask = real_image - real_image * mask_real + mpv * mask_real

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_part = train_d(netD, [real_image, real_image_mask], label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))
            txt_write.write(str({"iteration": iteration, "loss_d": err_dr, "loss_g": -err_g.item()}) + '\n')
          
        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'% iteration, nrow=4)
                vutils.save_image(torch.cat([
                        F.interpolate(real_image, args.im_size),
                        F.interpolate(rec_img_all, args.im_size), F.interpolate(rec_img_small, args.im_size), F.interpolate(rec_part, args.im_size)]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(),'d': netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MaskGan')
    parser.add_argument('--path', type=str, default='/dssg/home/zn_lzhx/PytorchPro/few-shot-images/100-shot-grumpy_cat', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='grumpy-cat', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--deterministic', type=bool, default=True, help='about torch.backends.cudnn.deterministic')
    # parser.add_argument('--ckpt', type=str, default='/media/zhihao/F05CC6255CC5E706/PytorchPro/MaskGAN/train_results/test8/models/all_100000.pth', help='checkpoint weight path if have one')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--benchmark', type=bool, default=True, help='about torch.backends.cudnn.benchmark')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  # 0.1
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--max_holes', type=int, default=1)
    parser.add_argument('--hole_min_w', type=int, default=95)
    parser.add_argument('--hole_max_w', type=int, default=97)
    parser.add_argument('--hole_min_h', type=int, default=95)
    parser.add_argument('--hole_max_h', type=int, default=97)
    parser.add_argument('--cn_input_size', type=int, default=160)
    parser.add_argument('--ld_input_size', type=int, default=96)
    parser.add_argument('--mpv', nargs=3, type=float, default=None)
    parser.add_argument('--alpha', type=float, default=4e-4)
    args = parser.parse_args()
    print(args)

    train(args)
