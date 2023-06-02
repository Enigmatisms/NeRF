"""
    Procedures reused in multiple executables
"""
import torch
import argparse
from tqdm import tqdm

from nerf.nerf_base import NeRF
from nerf.ref_model import RefNeRF
from torchvision import transforms
from torch.nn import functional as F
from nerf.dataset import CustomDataSet, AdaptiveResize
from nerf.addtional import ProposalNetwork, SoftL1Loss, LossPSNR
from torch.nn.functional import softplus
from torchvision.utils import save_image
from nerf.mip_methods import maxBlurFilter
from nerf.utils import fov2Focal, inverseSample, pose_spherical
from collections.abc import Iterable
from torch.cuda.amp import autocast as autocast

POSSIBLE_PATCH_SIZE = [50, 40, 60, 30]
RENDER_COARSE_PNUM = 64

def get_patch_size(image_size):
    sz = 50
    for patch_size in POSSIBLE_PATCH_SIZE:
        if image_size[1] % patch_size == 0:
            sz = patch_size
            patch_num = (image_size[0] // sz, image_size[1] // sz)
            break
    return sz, patch_num

# image size is (row, col)
def render_image(
    network:NeRF, prop_net:ProposalNetwork, render_pose:torch.Tensor, image_size, focal,
    near:float, far:float, sample_num:int=128, white_bkg:bool = False, render_depth = False, render_normal = False
) -> torch.Tensor:
    if not isinstance(image_size, Iterable):
        image_size = (image_size, image_size)
    is_ref_model = type(network) == RefNeRF
    render_normal &= is_ref_model
    target_device = render_pose.device
    col_idxs, row_idxs = torch.meshgrid(torch.arange(image_size[1]), torch.arange(image_size[0]), indexing = 'xy')      # output shape (imagesize[1], imagesize[0])
    coords = torch.stack((col_idxs - image_size[1] / 2, image_size[0] / 2 - row_idxs), dim = -1).to(target_device) + 0.5
    if isinstance(focal, Iterable):
        coords[..., 0] /= focal[1]
        coords[..., 1] /= focal[0]
    else:
        coords /= focal
    coords = torch.cat((coords, -torch.ones(image_size[0], image_size[1], 1, dtype = torch.float32, device = target_device).cuda()), dim = -1)
    ray_raw = torch.sum(coords.unsqueeze(-2) * render_pose[..., :-1], dim = -1)     # shape (H, W, 3)
    all_lengths = torch.linspace(near, far, 64, device = target_device)

    resulting_image = torch.zeros((3, image_size[0], image_size[1]), dtype = torch.float32, device = target_device)
    if render_normal:
        normal_img = torch.zeros((3, image_size[0], image_size[1]), dtype = torch.float32, device = target_device)
    if render_depth:
        depth_img = torch.zeros((3, image_size[0], image_size[1]), dtype = torch.float32, device = target_device)
    resolution = (far - near) / sample_num
    sz, patch_num = get_patch_size(image_size)

    for k in range(patch_num[0]):
        for j in range(patch_num[1]):
            camera_rays = torch.cat((render_pose[:, -1].expand(sz, sz, -1), ray_raw[(sz * k):(sz * (k + 1)), (sz * j):(sz * (j + 1))]), dim = -1).reshape(-1, 6)        # shape (2500, 6)
            sampled_lengths = (all_lengths + torch.rand((sz, sz, RENDER_COARSE_PNUM)).cuda() * resolution).view(-1, RENDER_COARSE_PNUM)        # shape (2500, sample_num)
            pts = render_pose[:, -1].unsqueeze(0) + sampled_lengths[..., None] * camera_rays[:, None, 3:]
            density = prop_net.forward(pts)
            prop_weights_raw = ProposalNetwork.get_weights(density, sampled_lengths, camera_rays[:, 3:])      # (ray_num, num of proposal interval)
            prop_weights = maxBlurFilter(prop_weights_raw, 0.01)
            fine_lengths, _ = inverseSample(prop_weights, sampled_lengths, sample_num + 1, sort = True)
            if is_ref_model:
                fine_samples, fine_lengths = NeRF.coarseFineMerge(camera_rays, sampled_lengths, fine_lengths)
                output_rgbo, normal = network.forward(fine_samples)
                output_rgbo[..., -1] = softplus(output_rgbo[..., -1] + 0.5)
            else:
                fine_lengths = fine_lengths[..., :-1]
                fine_samples = NeRF.length2pts(camera_rays, fine_lengths)
                output_rgbo = network.forward(fine_samples)
            
            part_image, _, extras = NeRF.render(
                output_rgbo, fine_lengths, camera_rays[..., 3:], 
                white_bkg = white_bkg, density_act = F.relu, 
                render_depth = (near, far) if render_depth else None, 
                normal_info = (normal, render_pose[:, -2]) if render_normal else None
            )          # originally outputs (2500, 3) -> (reshape) (sz, sz, 3) -> (to image) (3, sz, sz)
            resulting_image[:, (sz * k):(sz * (k + 1)), (sz * j):(sz * (j + 1))] = part_image.view(sz, sz, 3).permute(2, 0, 1)
            if render_depth == True:
                depth_img[:, (sz * k):(sz * (k + 1)), (sz * j):(sz * (j + 1))] = extras["depth_img"].view(sz, sz)
            if render_normal == True:
                normal_img[:, (sz * k):(sz * (k + 1)), (sz * j):(sz * (j + 1))] = extras["normal_img"].view(sz, sz)
    result = dict()
    result["rgb"] = resulting_image
    if render_depth:
        result["depth_img"] = depth_img
    if render_normal:
        result["normal_img"] = normal_img
    return result

def render_only(args, model_path: str, opt_level: str):
    use_amp             = args.use_scaler
    load_path_mip       = model_path + args.name + "_mip.pth"
    load_path_prop      = model_path + args.name + "_prop.pth"
    near_t              = args.near
    far_t               = args.far
    dataset_name        = args.dataset_name
    img_scale           = args.img_scale
    scene_scale         = args.scene_scale
    use_white_bkg       = args.white_bkg
    opt_mode            = args.opt_mode
    use_ref_nerf        = args.ref_nerf
    eval_poses          = args.eval_poses 
    render_normal       = args.render_normal & (not eval_poses)
    render_depth        = args.render_depth & (not eval_poses)
    transform_funcs = transforms.Compose([
        AdaptiveResize(img_scale),
        transforms.ToTensor(),
    ])
    testset = CustomDataSet("../dataset/refnerf/%s/"%(dataset_name), transform_funcs, scene_scale, False, use_alpha = False)

    cam_fov_test, _ = testset.getCameraParam()
    r_c = testset.r_c()
    if eval_poses:
        all_poses = testset.tfs.cuda()
        loss_func = SoftL1Loss()
        psnr_func = LossPSNR()
    else:
        all_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in torch.linspace(-180,180,120 + 1)[:-1]], 0).cuda()
        del testset
    test_focal = fov2Focal(cam_fov_test, r_c)

    if use_ref_nerf:
        from nerf.ref_model import RefNeRF
        mip_net = RefNeRF(10, args.ide_level, hidden_unit = args.nerf_net_width, perturb_bottle_neck_w = args.bottle_neck_noise, use_srgb = args.use_srgb).cuda()
    else:
        from nerf.mip_model import MipNeRF
        mip_net = MipNeRF(10, 4, hidden_unit = args.nerf_net_width)
    prop_net = ProposalNetwork(10, hidden_unit = args.prop_net_width).cuda()
    if use_amp and opt_mode != "native":
        from apex import amp
        [mip_net, prop_net] = amp.initialize([mip_net, prop_net], None, opt_level = opt_level)
    mip_net.loadFromFile(load_path_mip, use_amp and opt_mode != "native")
    prop_net.loadFromFile(load_path_prop, use_amp and opt_mode != "native")

    mip_net.eval()
    prop_net.eval()
    with torch.no_grad():
        for i, pose in tqdm(list(enumerate(all_poses))):
            pose[:3, -1] *= scene_scale
            if opt_mode == "native":
                with autocast():
                    result = render_image(mip_net, prop_net, pose[:3, :], r_c, test_focal, near_t, far_t, 128, 
                        white_bkg = use_white_bkg, render_normal = render_normal, render_depth = render_depth)
            else:
                result = render_image(mip_net, prop_net, pose[:3, :], r_c, test_focal, near_t, far_t, 128, 
                        white_bkg = use_white_bkg, render_normal = render_normal, render_depth = render_depth)
            if eval_poses == True:
                gt_img, _ = testset[i]
                gt_img = gt_img.cuda()
                loss = loss_func(result['rgb'], gt_img)
                psnr = psnr_func(loss)
                print("Image loss:%.6f\tPSNR:%.4f"%(loss.item(), psnr.item()))
                result['gt_img'] = gt_img
            output_dir = "given" if eval_poses else "sphere" 
            save_image(list(result.values()), "./output/%s/result_%03d.png"%(output_dir, i), nrow = 1 + render_depth + render_depth + eval_poses)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 2400, help = "Training lasts for . epochs")
    parser.add_argument("--max_save", type = int, default = 3, help = "Check point max save number")
    parser.add_argument("--sample_ray_num", type = int, default = 1024, help = "<x> rays to sample per training time")
    parser.add_argument("--coarse_sample_pnum", type = int, default = 64, help = "Points to sample in coarse net")
    parser.add_argument("--fine_sample_pnum", type = int, default = 128, help = "Points to sample in fine net")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Tensorboard output interval (train time)")
    parser.add_argument("--output_time", type = int, default = 20, help = "Image output interval (train time)")
    parser.add_argument("--center_crop_iter", type = int, default = 0, help = "Produce center")
    parser.add_argument("--prop_net_width", type = int, default = 256, help = "Width of proposal network")
    parser.add_argument("--nerf_net_width", type = int, default = 256, help = "Width of nerf network")
    parser.add_argument("--near", type = float, default = 2., help = "Nearest sample depth")
    parser.add_argument("--far", type = float, default = 6., help = "Farthest sample depth")
    parser.add_argument("--center_crop_x", type = float, default = 0.5, help = "Center crop x axis ratio")
    parser.add_argument("--center_crop_y", type = float, default = 0.5, help = "Center crop y axis ratio")
    parser.add_argument("--name", type = str, default = "model_1", help = "Model name for loading")
    parser.add_argument("--dataset_name", type = str, default = "lego", help = "Input dataset name in nerf synthetic dataset")
    parser.add_argument("--img_scale", type = float, default = 0.5, help = "Scale of the image")
    parser.add_argument("--scene_scale", type = float, default = 1.0, help = "Scale of the scene")
    parser.add_argument("--grad_clip", type = float, default = -0.01, help = "Gradient clipping parameter (Negative number means no clipping)")
    parser.add_argument("--pe_period_scale", type = float, default = 0.5, help = "Scale of positional encoding")
    # opt related
    parser.add_argument("--opt_mode", type = str, default = "O1", help = "Optimization mode: none, native (torch amp), O1, O2 (apex amp)")
    parser.add_argument("--min_ratio", type = float, default = 0.01, help = "Minimum for now_lr / lr")
    parser.add_argument("--decay_rate", type = float, default = 0.1, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--decay_step", type = int, default = 100000, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--warmup_step", type = int, default = 500, help = "Warm up step (from lowest lr to starting lr)")
    parser.add_argument("--lr", type = float, default = 1.5e-4, help = "Start lr")
    # short bool options
    parser.add_argument("-d", "--del_dir", default = False, action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
    parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
    parser.add_argument("-b", "--debug", default = False, action = "store_true", help = "Code debugging (detect gradient anomaly and NaNs)")
    parser.add_argument("-v", "--visualize", default = False, action = "store_true", help = "Visualize proposal network")
    parser.add_argument("-r", "--do_render", default = False, action = "store_true", help = "Only render the result")
    parser.add_argument("-w", "--white_bkg", default = False, action = "store_true", help = "Output white background")
    parser.add_argument("-t", "--ref_nerf", default = False, action = "store_true", help = "Test Ref NeRF")
    parser.add_argument("-u", "--use_srgb", default = False, action = "store_true", help = "Whether to use srgb in the output or not")
    parser.add_argument("-e", "--eval_poses", default = False, action = "store_true", help = "Whether to use test set poses to render image")
    # long bool options
    parser.add_argument("--render_depth", default = False, action = "store_true", help = "Render depth image")
    parser.add_argument("--render_normal", default = False, action = "store_true", help = "Render normal image")
    parser.add_argument("--prop_normal", default = False, action = "store_true", help = "(For proposal net) Whether to learn normals")
    # ref nerf options
    parser.add_argument("--ide_level", type = int, default = 4, help = "Max level of spherical harmonics to be used")
    parser.add_argument("--bottle_neck_noise", type = float, default = 0.02, help = "Noise std for perturbing bottle_neck vector")
    return parser
