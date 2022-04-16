#-*-coding:utf-8-*-
"""
    NeRF training executable
"""
import os
from torchvision import transforms
import torch
import argparse

from torch import optim
from torch.utils.data import DataLoader

from py.utils import fov2Focal, inverseSample, getSummaryWriter, validSampler, randomFromOneImage, saveModel
from torchvision.utils import save_image
from py.dataset import CustomDataSet
from py.model import NeRF, nan_hook
from py.timer import Timer
from functools import partial

default_chkpt_path = "./check_points/"
default_model_path = "./model/"

parser = argparse.ArgumentParser()

# general args
parser.add_argument("--epochs", type = int, default = 2000, help = "Training lasts for . epochs")
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
args = parser.parse_args()

def render_image(network:NeRF, render_pose:torch.Tensor, image_size:int, focal:float, near:float, far:float, sample_num:int=128) -> torch.Tensor:
    col_idxs, row_idxs = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing = 'xy')
    coords = torch.stack((col_idxs - image_size / 2, image_size / 2 - row_idxs), dim = -1).cuda() / focal
    coords = torch.cat((coords, -torch.ones(image_size, image_size, 1, dtype = torch.float32).cuda()), dim = -1)
    ray_raw = torch.sum(coords.unsqueeze(-2) * render_pose[..., :-1], dim = -1)     # shape (H, W, 3)

    resolution = (far - near) / sample_num
    all_lengths = torch.linspace(near, far - resolution, sample_num).cuda()
    sampled_lengths = all_lengths + torch.rand((image_size, image_size, sample_num)).cuda() * resolution

    pts = render_pose[:, -1].unsqueeze(0) + all_lengths[..., None] * ray_raw[:, :, None, :]
    image_sampled = torch.cat((pts, ray_raw.unsqueeze(-2).repeat(1, 1, sample_num, 1)), dim = -1)

    resulting_image = torch.zeros((3, image_size, image_size), dtype = torch.float32).cuda()
    patch_num = image_size // 50
    for k in range(patch_num):
        for j in range(patch_num):
            output_rgbo = network.forward(image_sampled[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, sample_num, 6))
            part_image, _ = NeRF.render(
                output_rgbo, sampled_lengths[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, sample_num),
                ray_raw[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, 3)
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
    load_path_coarse    = default_chkpt_path + args.name + "_coarse.pt"
    load_path_fine      = default_chkpt_path + args.name + "_fine.pt"
    # Bool options
    del_dir             = args.del_dir
    use_load            = args.load
    debugging           = args.debug
    use_amp             = (args.use_scaler and (not debugging))
    if use_amp:
        from apex import amp
        opt_level = 'O2'

    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    # ======= instantiate model =====
    # NOTE: model is recommended to have loadFromFile method
    coarse_net = NeRF(10, 4).cuda()
    fine_net = NeRF(10, 4).cuda()

    if debugging:
        for submodule in coarse_net.modules():
            submodule.register_forward_hook(nan_hook)

        for submodule in fine_net.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    # ======= Loss function ==========
    loss_func = lambda x, y : torch.mean((x - y) ** 2)
    grad_vars = list(coarse_net.parameters())
    grad_vars += list(fine_net.parameters())
    # ======= Optimizer and scheduler ========
    opt = optim.Adam(params = grad_vars, lr = 4.5e-4, betas=(0.9, 0.999))
    if use_amp:
        [coarse_net, fine_net], opt = amp.initialize([coarse_net, fine_net], opt, opt_level=opt_level)
    if use_load == True and os.path.exists(load_path_coarse) and os.path.exists(load_path_fine):
        coarse_net.loadFromFile(load_path_coarse, use_amp, opt)
        fine_net.loadFromFile(load_path_fine, use_amp, opt)
    else:
        print("Not loading or load path '%s' or '%s' does not exist."%(load_path_coarse, load_path_fine))
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
    test_views = []
    for i in range(3, 6):
        test_views.append(testset[i * 11])
    torch.cuda.empty_cache()

    # ====== tensorboard summary writer ======
    writer = getSummaryWriter(epochs, del_dir)

    train_cnt, test_cnt = 0, 0
    train_timer, eval_timer, epoch_timer, render_timer = Timer(5), Timer(5), Timer(3), Timer(4)
    for ep in range(epochs):
        epoch_timer.tic()
        
        for i, (train_img, train_tf) in enumerate(train_loader):
            train_timer.tic()
            train_img = train_img.cuda().squeeze(0)
            train_tf = train_tf.cuda().squeeze(0)
            now_crop = (center_crop if train_cnt < center_crop_iter else 1.)
            valid_pixels, valid_coords = randomFromOneImage(train_img, now_crop)
            coarse_samples, coarse_lengths, rgb_targets, coarse_cam_rays = validSampler(
                valid_pixels, valid_coords, train_tf, sample_ray_num, coarse_sample_pnum, 400, 400, train_focal, near_t, far_t
            )
            coarse_rgbo = coarse_net.forward(coarse_samples)
            coarse_rendered, weights = NeRF.render(coarse_rgbo, coarse_lengths, coarse_cam_rays[:, 3:])
            fine_lengths = inverseSample(weights, coarse_lengths, fine_sample_pnum)
            fine_samples, fine_lengths = NeRF.coarseFineMerge(coarse_cam_rays, coarse_lengths, fine_lengths)      # (ray_num, 192, 6)
            # 此处存在逻辑问题，需要第二次sort，并且RGB需要整理出来
            fine_rgbo = fine_net.forward(fine_samples)
            fine_rendered, _ = NeRF.render(fine_rgbo, fine_lengths, coarse_cam_rays[:, 3:])

            loss:torch.Tensor = loss_func(coarse_rendered, rgb_targets) + loss_func(fine_rendered, rgb_targets)
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
            fine_net.eval()
            coarse_net.eval()
            with torch.no_grad():
                ## +++++++++++ Load from Test set ++++++++=
                eval_timer.tic()
                render_timer.tic()
                test_results = []
                test_loss = torch.zeros(1).cuda()
                for test_img, test_tf in test_views:
                    test_result = render_image(fine_net, test_tf.cuda(), 400, test_focal, near_t, far_t, fine_sample_pnum)
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
                saveModel(coarse_net,  "%schkpt_%d_coarse.pt"%(default_chkpt_path, train_cnt), opt = None, amp = (amp) if use_amp else None)
                saveModel(fine_net,  "%schkpt_%d_fine.pt"%(default_chkpt_path, train_cnt), opt = opt, amp = (amp) if use_amp else None)
                test_cnt += 1
            coarse_net.train()
            fine_net.train()
        epoch_timer.toc()
        print("Epoch %4d / %4d completed\trunning time for this epoch: %.5lf\testimated remaining time: %s"
                %(ep, epochs, epoch_timer.get_mean_time(), epoch_timer.remaining_time(epochs - ep - 1))
        )
    # ======== Saving the model ========
    saveModel(coarse_net,  "%model_%d_coarse.pth"%(default_model_path, 2), opt = None, amp = (amp) if use_amp else None)
    saveModel(fine_net,  "%model_%d_fine.pth"%(default_model_path, 2), opt = opt, amp = (amp) if use_amp else None)
    writer.close()
    print("Output completed.")

if __name__ == "__main__":
    main()
