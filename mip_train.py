#-*-coding:utf-8-*-
"""
    NeRF training executable
"""
import os
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

from torch import optim
from torch.utils.data import DataLoader

from functools import partial
from py.dataset import CustomDataSet
from torchvision.utils import save_image
from py.model import NeRF
from py.timer import Timer
from py.addtional import getBounds, ProposalLoss, ProposalNetwork
from py.nerf_helper import nan_hook, saveModel
from py.mip_methods import ipe_feature, maxBlurFilter
from py.utils import fov2Focal, inverseSample, getSummaryWriter, validSampler, randomFromOneImage, getRadius

default_chkpt_path = "./check_points/"
default_model_path = "./model/"

parser = argparse.ArgumentParser()

# general args
parser.add_argument("--epochs", type = int, default = 2000, help = "Training lasts for . epochs")
parser.add_argument("--ep_start", type = int, default = 0, help = "Start epoches from <ep_start>")
parser.add_argument("--sample_ray_num", type = int, default = 1024, help = "<x> rays to sample per training time")
parser.add_argument("--coarse_sample_pnum", type = int, default = 64, help = "Points to sample in coarse net")
parser.add_argument("--fine_sample_pnum", type = int, default = 128, help = "Points to sample in fine net")
parser.add_argument("--eval_time", type = int, default = 4, help = "Tensorboard output interval (train time)")
parser.add_argument("--center_crop_iter", type = int, default = 500, help = "Produce center")
parser.add_argument("--near", type = float, default = 2., help = "Nearest sample depth")
parser.add_argument("--far", type = float, default = 6., help = "Farthest sample depth")
parser.add_argument("--center_crop", type = float, default = 0.5, help = "Farthest sample depth")
parser.add_argument("--name", type = str, default = "model_1", help = "Model name for loading")
parser.add_argument("--dataset_name", type = str, default = "ship", help = "Input dataset name in nerf synthetic dataset")
# opt related
parser.add_argument("--min_ratio", type = float, default = 0.1, help = "lr exponential decay, final / intial min ratio")
parser.add_argument("--alpha", type = float, default = 0.99995, help = "lr exponential decay rate")
# bool options
parser.add_argument("-d", "--del_dir", default = False, action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
parser.add_argument("-b", "--debug", default = False, action = "store_true", help = "Code debugging (detect gradient anomaly and NaNs)")
parser.add_argument("-v", "--visualize", default = False, action = "store_true", help = "Visualize proposal network")
args = parser.parse_args()

def render_image(network:NeRF, render_pose:torch.Tensor, image_size:int, focal:float, near:float, far:float, sample_num:int=128, pixel_width:float=0.0) -> torch.Tensor:
    target_device = render_pose.device
    col_idxs, row_idxs = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing = 'xy')
    coords = torch.stack((col_idxs - image_size / 2, image_size / 2 - row_idxs), dim = -1).to(target_device) / focal
    coords = torch.cat((coords, -torch.ones(image_size, image_size, 1, dtype = torch.float32, device = target_device).cuda()), dim = -1)
    ray_raw = torch.sum(coords.unsqueeze(-2) * render_pose[..., :-1], dim = -1)     # shape (H, W, 3)

    all_lengths = torch.linspace(near, far, sample_num + 1, device = target_device)

    resulting_image = torch.zeros((3, image_size, image_size), dtype = torch.float32, device = target_device)
    patch_num = image_size // 50

    # TODO: render是否也需要改成 cone rendering？需要用mu_t进行渲染
    for k in range(patch_num):
        for j in range(patch_num):
            sampled_lengths = all_lengths + torch.zeros(50, 50, sample_num + 1, device = target_device, dtype = torch.float32)
            camera_rays = torch.cat((render_pose[:, -1].expand(50, 50, -1), ray_raw[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))]), dim = -1).reshape(-1, 6)
            encoded_ptx, mu, mu_t = ipe_feature(sampled_lengths.view(-1, sample_num + 1), camera_rays, 10, pixel_width)
            samples = torch.cat((mu, camera_rays.unsqueeze(-2).repeat(1, sample_num, 1)), dim = -1)

            output_rgbo = network.forward(samples, encoded_ptx)
            part_image, _ = NeRF.render(
                output_rgbo, mu_t, ray_raw[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, 3)
            )          # originally outputs (2500, 3) -> (reshape) (50, 50, 3) -> (to image) (3, 50, 50)
            resulting_image[:, (50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))] = part_image.view(50, 50, 3).permute(2, 0, 1)
    return resulting_image

def main():
    epochs              = args.epochs
    sample_ray_num      = args.sample_ray_num
    coarse_sample_pnum  = args.coarse_sample_pnum
    fine_sample_pnum    = args.fine_sample_pnum
    near_t              = args.near
    far_t               = args.far
    center_crop_iter    = args.center_crop_iter
    center_crop         = args.center_crop

    eval_time           = args.eval_time
    dataset_name        = args.dataset_name
    load_path_mip       = default_chkpt_path + args.name + "_mip.pt"
    load_path_prop      = default_chkpt_path + args.name + "_prop.pt"
    # Bool options
    del_dir             = args.del_dir
    use_load            = args.load
    debugging           = args.debug
    use_amp             = (args.use_scaler and (not debugging))
    viz_prop            = args.visualize
    ep_start            = args.ep_start
    if use_amp:
        from apex import amp
        opt_level = 'O2'

    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    # ======= instantiate model =====
    # NOTE: model is recommended to have loadFromFile method
    mip_net = NeRF(10, 4).cuda()
    prop_net = ProposalNetwork(10).cuda()
    if debugging:
        for submodule in mip_net.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    # ======= Loss function ==========
    loss_func = lambda x, y : torch.mean((x - y) ** 2)
    prop_loss_func = ProposalLoss().cuda()
    # ======= Optimizer and scheduler ========
    grad_vars = list(mip_net.parameters()) + list(prop_net.parameters())
    opt = optim.Adam(params = grad_vars, lr = 4.5e-4, betas=(0.9, 0.999))
    if use_amp:
        [mip_net, prop_net], opt = amp.initialize([mip_net, prop_net], opt, opt_level=opt_level)
    if use_load == True and os.path.exists(load_path_mip) and os.path.exists(load_path_prop):
        mip_net.loadFromFile(load_path_mip, use_amp, opt)
        prop_net.loadFromFile(load_path_prop, use_amp, opt)
    else:
        print("Not loading or load path '%s' or '%s' does not exist."%(load_path_mip, load_path_prop))
    def lr_func(x:int, alpha:float, min_ratio:float):
        val = alpha**x
        return val if val > min_ratio else min_ratio
    preset_lr_func = partial(lr_func, alpha = args.alpha, min_ratio = args.min_ratio)
    sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda = preset_lr_func, last_epoch=-1)
    transform_funcs = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    # 数据集加载
    trainset = CustomDataSet("../dataset/nerf_synthetic/%s/"%(dataset_name), transform_funcs, True, use_alpha = False)
    testset = CustomDataSet("../dataset/nerf_synthetic/%s/"%(dataset_name), transform_funcs, False, use_alpha = False)
    cam_fov_train, train_cam_tf = testset.getCameraParam()
    train_cam_tf = train_cam_tf[0].cuda()

    train_loader = DataLoader(trainset, 1, shuffle = True, num_workers = 4)

    cam_fov_test, tmp = testset.getCameraParam()
    del tmp
    train_focal = fov2Focal(cam_fov_train, 400)
    test_focal = fov2Focal(cam_fov_test, 400)
    # TODO:r是不知道的，只能用focal来算
    pixel_width = getRadius(train_focal)
    test_views = []
    for i in range(3, 6):
        test_views.append(testset[i * 11])
    torch.cuda.empty_cache()

    # ====== tensorboard summary writer ======
    writer = getSummaryWriter(epochs, del_dir)

    train_cnt, test_cnt = ep_start * 100, ep_start // 20
    train_timer, eval_timer, epoch_timer, render_timer = Timer(5), Timer(5), Timer(3), Timer(4)
    for ep in range(ep_start, epochs):
        epoch_timer.tic()
        
        for i, (train_img, train_tf) in enumerate(train_loader):
            train_timer.tic()
            train_img = train_img.cuda().squeeze(0)
            train_tf = train_tf.cuda().squeeze(0)
            now_crop = (center_crop if train_cnt < center_crop_iter else 1.)
            valid_pixels, valid_coords = randomFromOneImage(train_img, now_crop)

            # sample one more t to form (coarse_sample_pnum) proposal interval
            coarse_lengths, rgb_targets, coarse_cam_rays = validSampler(
                valid_pixels, valid_coords, train_tf, sample_ray_num, coarse_sample_pnum + 1, 400, 400, train_focal, near_t, far_t, False
            )
            ipe_pts, mu_pts, mu_ts = ipe_feature(coarse_lengths, coarse_cam_rays, 10, pixel_width)
            density = prop_net.forward(mu_pts, ipe_pts)
            prop_weights_raw = ProposalNetwork.get_weights(density, mu_ts, coarse_cam_rays[:, 3:])      # (ray_num, num of proposal interval)

            prop_weights = maxBlurFilter(prop_weights_raw, 0.01)
            fine_lengths, sort_inds, below_idxs = inverseSample(prop_weights, mu_ts, fine_sample_pnum + 1, sort = True)
            ipe_pts, mu_pts, mu_ts = ipe_feature(fine_lengths, coarse_cam_rays, 10, pixel_width)
            samples = torch.cat((mu_pts, coarse_cam_rays.unsqueeze(-2).repeat(1, fine_sample_pnum, 1)), dim = -1)
            fine_rgbo = mip_net.forward(samples)
            fine_rendered, weights = NeRF.render(fine_rgbo, mu_ts, coarse_cam_rays[:, 3:])
            # (mu_ts and weights) (coarse_length & prop_weights should be visualized)
            weight_bounds:torch.Tensor = getBounds(prop_weights, below_idxs, sort_inds)     # output shape: (ray_num, num of conical frustum)
            prop_loss:torch.Tensor = prop_loss_func(weight_bounds, weights.detach())             # stop the gradient of NeRF MLP 
            loss:torch.Tensor = prop_loss + loss_func(fine_rendered, rgb_targets)

            train_timer.toc()
            if use_amp:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            opt.step()
            opt.zero_grad()

            if train_cnt % eval_time == 1:
                # ========= Evaluation output ========
                remaining_cnt = (epochs - ep - 1) * 100 + 100 - i
                print("Traning Epoch: %4d / %4d\t Iter %4d / %4d\t train loss: %.4f\tlr:%.7lf\tcenter crop:%.1lf\taverage time: %.4lf\tremaining train time:%s"%(
                        ep, epochs, i, 100, loss.item(), sch.get_last_lr()[-1], now_crop, train_timer.get_mean_time(), train_timer.remaining_time(remaining_cnt)
                ))
                writer.add_scalar('Train Loss', loss, train_cnt)
                writer.add_scalar('Learning Rate', sch.get_last_lr()[-1], ep)
                sch.step()
            train_cnt += 1

        # model.eval()
        if (ep % 20 == 0) or ep == epochs - 1:
            mip_net.eval()
            with torch.no_grad():
                ## +++++++++++ Load from Test set ++++++++=
                eval_timer.tic()
                if viz_prop == True:
                    for k in range(4):
                        coarse_delta = coarse_lengths[k * 20, 1:] - coarse_lengths[k * 20, :-1]
                        plt.subplot(2, 2, 1 + k)
                        plt.bar((coarse_lengths[k * 20, :-1] + coarse_delta / 2).cpu(), prop_weights[k * 20].cpu(), width = coarse_delta.cpu(), alpha = 0.7, color = 'b', label = 'Proposal weights')
                        plt.bar(mu_ts[k * 20].cpu(), weights[k * 20].detach().cpu(), width = 0.01, color = 'r', label = 'NeRF weights')
                        plt.legend()
                    plt.savefig("./output/proposal_dist_%03d.png"%(test_cnt))
                    plt.clf()
                    plt.cla()
                render_timer.tic()
                test_results = []
                test_loss = torch.zeros(1).cuda()
                for test_img, test_tf in test_views:
                    test_result = render_image(mip_net, test_tf.cuda(), 400, test_focal, near_t, far_t, fine_sample_pnum, pixel_width = pixel_width)
                    test_results.append(test_result)
                    test_loss += loss_func(test_result, test_img.cuda())
                render_timer.toc()
                eval_timer.toc()
                writer.add_scalar('Test Loss', loss, test_cnt)
                print("Evaluation in epoch: %4d / %4d\t, test counter: %d test loss: %.4f\taverage time: %.4lf\tavg render time:%lf\tremaining eval time:%s"%(
                        ep, epochs, test_cnt, test_loss.item() / 2, eval_timer.get_mean_time(), render_timer.get_mean_time(), eval_timer.remaining_time(epochs - ep - 1)
                ))
                images_to_save = []
                images_to_save.extend(test_results)
                save_image(images_to_save, "./output/result_%03d.png"%(test_cnt), nrow = 3)
                # ======== Saving checkpoints ========
                saveModel(mip_net,  "%schkpt_%d_mip.pt"%(default_chkpt_path, train_cnt), opt = None, amp = (amp) if use_amp else None)
                saveModel(prop_net,  "%schkpt_%d_prop.pt"%(default_chkpt_path, train_cnt), opt = None, amp = (amp) if use_amp else None)
                test_cnt += 1
            mip_net.train()
        epoch_timer.toc()
        print("Epoch %4d / %4d completed\trunning time for this epoch: %.5lf\testimated remaining time: %s"
                %(ep, epochs, epoch_timer.get_mean_time(), epoch_timer.remaining_time(epochs - ep - 1))
        )
    # ======== Saving the model ========
    saveModel(mip_net,  "%model_%d_mip.pth"%(default_model_path, 2), opt = None, amp = (amp) if use_amp else None)
    writer.close()
    print("Output completed.")

if __name__ == "__main__":
    main()
