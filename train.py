#-*-coding:utf-8-*-
"""
    NeRF training executable
"""
import os
from torchvision import transforms
import torch
import argparse

from torch import optim

from torch import nn
from nerf_helper import imageSampling, sampling
from py.utils import fov2Focal, inverseSample, getSummaryWriter, validSampler, getValidSamples
from torchvision.utils import save_image
from py.dataset import CustomDataSet
from py.model import NeRF
from py.timer import Timer
from functools import partial

default_chkpt_path = "./check_points/"
default_model_path = "./model/"

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--epochs", type = int, default = 800, help = "Training lasts for . epochs")
parser.add_argument("--train_per_epoch", type = int, default = 600, help = "Train (sample) <x> times in one epoch")
parser.add_argument("--sample_ray_num", type = int, default = 2048, help = "<x> rays to sample per training time")
parser.add_argument("--coarse_sample_pnum", type = int, default = 64, help = "Points to sample in coarse net")
parser.add_argument("--fine_sample_pnum", type = int, default = 128, help = "Points to sample in fine net")
parser.add_argument("--eval_time", type = int, default = 4, help = "Tensorboard output interval (train time)")
parser.add_argument("--near", type = float, default = 2., help = "Nearest sample depth")
parser.add_argument("--far", type = float, default = 6., help = "Farthest sample depth")
parser.add_argument("--name", type = str, default = "model_1", help = "Model name for loading")
parser.add_argument("--dataset_name", type = str, default = "lego", help = "Input dataset name in nerf synthetic dataset")
# opt related
parser.add_argument("--min_ratio", type = float, default = 0.05, help = "lr exponential decay, final / intial min ratio")
parser.add_argument("--alpha", type = float, default = 0.99996, help = "lr exponential decay rate")
# bool options
parser.add_argument("-d", "--del_dir", default = False, action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
args = parser.parse_args()

def main():
    epochs              = args.epochs
    train_per_epoch     = args.train_per_epoch
    sample_ray_num      = args.sample_ray_num
    coarse_sample_pnum  = args.coarse_sample_pnum
    fine_sample_pnum    = args.fine_sample_pnum
    near_t              = args.near
    far_t               = args.far

    eval_time           = args.eval_time
    dataset_name        = args.dataset_name
    load_path_coarse    = default_chkpt_path + args.name + "_coarse.pt"
    load_path_fine      = default_chkpt_path + args.name + "_fine.pt"
    # Bool options
    del_dir             = args.del_dir
    use_load            = args.load

    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    # ======= instantiate model =====
    # NOTE: model is recommended to have loadFromFile method
    coarse_net = NeRF(10, 4).cuda()
    fine_net = NeRF(10, 4).cuda()
    if use_load == True and os.path.exists(load_path_coarse) and os.path.exists(load_path_fine):
        coarse_net.loadFromFile(load_path_coarse)
        fine_net.loadFromFile(load_path_fine)
    else:
        print("Not loading or load path '%s' or '%s' does not exist."%(load_path_coarse, load_path_fine))

    # ======= Loss function ==========
    loss_func = nn.MSELoss().cuda()

    # ======= Optimizer and scheduler ========
    opt_c = optim.SGD(coarse_net.parameters(), lr = 0.4, momentum=0.1, dampening=0.1)
    opt_f = optim.SGD(fine_net.parameters(), lr = 0.4, momentum=0.1, dampening=0.1)

    # min_max_ratio = args.min_lr / args.max_lr
    # lec_sch_func = CosineLRScheduler(opt, t_initial = epochs // 2, t_mul = 1, lr_min = min_max_ratio, decay_rate = 0.1,
    #         warmup_lr_init = min_max_ratio, warmup_t = 10, cycle_limit = 2, t_in_epochs = True)
    def lr_func(x:int, alpha:float, min_ratio:float):
        val = alpha**x
        return val if val > min_ratio else min_ratio
    preset_lr_func = partial(lr_func, alpha = args.alpha, min_ratio = args.min_ratio)
    sch_c = optim.lr_scheduler.LambdaLR(opt_c, lr_lambda = preset_lr_func, last_epoch=-1)
    sch_f = optim.lr_scheduler.LambdaLR(opt_f, lr_lambda = preset_lr_func, last_epoch=-1)
    transform_funcs = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
    ])
    # 数据集加载
    trainset = CustomDataSet("../dataset/nerf_synthetic/%s/"%(dataset_name), transform_funcs, True, use_alpha = True)
    testset = CustomDataSet("../dataset/nerf_synthetic/%s/"%(dataset_name), transform_funcs, False, use_alpha = False)
    cam_fov_train, train_cam_tf, train_images = trainset.get_dataset(to_cuda = True)
    cam_fov_test, test_cam_tf = testset.getCameraParam()
    train_focal = fov2Focal(cam_fov_train, 400)
    test_focal = fov2Focal(cam_fov_test, 400)
    test_cam_tf = test_cam_tf.cuda()

    # ====== tensorboard summary writer ======
    writer = getSummaryWriter(epochs, del_dir)

    train_cnt, test_cnt = 0, 0
    train_timer, eval_timer, epoch_timer, render_timer = Timer(5), Timer(5), Timer(3), Timer(16)
    valid_pixels, valid_coords = getValidSamples(train_images)
    del train_images
    torch.cuda.empty_cache()
    print(train_cam_tf[0])
    for ep in range(epochs):
        epoch_timer.tic()
        coarse_net.train()
        fine_net.train()
        for i in range(train_per_epoch):
            train_timer.tic()
            loss = torch.zeros(1).cuda()
            # for _ in range(4):          # train 4 times, and then backward to complement the batch size
            coarse_samples, coarse_lengths = validSampler(
                valid_pixels, valid_coords, train_cam_tf, sample_ray_num, coarse_sample_pnum, 400, 400, train_focal, near_t, far_t
            )
            # coarse_samples:torch.Tensor = torch.zeros(sample_ray_num, coarse_sample_pnum + 1, 9, dtype = torch.float32).cuda()
            # coarse_lengths:torch.Tensor = torch.zeros(sample_ray_num, coarse_sample_pnum, dtype = torch.float32).cuda()
            # sampling(train_images, train_cam_tf, coarse_samples, coarse_lengths, sample_ray_num, coarse_sample_pnum, train_focal, near_t, far_t)
            coarse_cams = coarse_samples[:, -1, :-3].contiguous()
            gt_rgb = coarse_samples[:, -1, -3:].contiguous()
            coarse_samples = coarse_samples[:, :-1, :].contiguous()
            coarse_rgbo = coarse_net.forward(coarse_samples)
            coarse_rendered, normed_weights = NeRF.render(coarse_rgbo, coarse_lengths, coarse_samples[:, :, 3:6].norm(dim = -1))
            loss = loss_func(coarse_rendered, gt_rgb)
            fine_samples, fine_lengths = inverseSample(normed_weights, coarse_cams, fine_sample_pnum, near_t, far_t)
            fine_samples, fine_lengths = NeRF.coarseFineMerge(coarse_cams, coarse_lengths, fine_lengths)      # (ray_num, 192, 6)
            # 此处存在逻辑问题，需要第二次sort，并且RGB需要整理出来
            fine_rgbo = fine_net.forward(fine_samples)
            fine_rendered, _ = NeRF.render(fine_rgbo, fine_lengths, fine_samples[:, :, 3:6].norm(dim = -1))
            loss = loss + loss_func(fine_rendered, gt_rgb)
            train_timer.toc()
            
            opt_c.zero_grad()
            opt_f.zero_grad()
            loss.backward()
            opt_c.step()
            opt_f.step()

            if train_cnt % eval_time == 1:
                # ========= Evaluation output ========
                remaining_cnt = (epochs - ep - 1) * train_per_epoch + train_per_epoch - i
                print("Traning Epoch: %4d / %4d\t Iter %4d / %4d\t train loss: %.4f\tlr:%.7lf\taverage time: %.4lf\tremaining train time:%s"%(
                        ep, epochs, i, train_per_epoch, loss.item(), sch_f.get_last_lr()[-1], train_timer.get_mean_time(), train_timer.remaining_time(remaining_cnt)
                ))
                writer.add_scalar('Train Loss', loss, train_cnt)
                writer.add_scalar('Learning Rate', sch_f.get_last_lr()[-1], ep)
                sch_c.step()
                sch_f.step()
            train_cnt += 1

        # model.eval()
        fine_net.eval()
        with torch.no_grad():
            ## +++++++++++ Load from Test set ++++++++=
            eval_timer.tic()
            image_input = testset[0].cuda()
            image_sampled = torch.zeros(400, 400, fine_sample_pnum, 6, dtype = torch.float32).cuda()
            image_lengths = torch.zeros(400, 400, fine_sample_pnum, dtype = torch.float32).cuda()
            imageSampling(train_cam_tf[0], image_sampled, image_lengths, 400, 400, fine_sample_pnum, test_focal, near_t, far_t)
            resulting_image = torch.zeros_like(image_input, dtype = torch.float32).cuda()
            for k in range(8):
                for j in range(8):
                    render_timer.tic()
                    output_rgbo = fine_net.forward(image_sampled[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, fine_sample_pnum, 6))
                    part_image, _ = NeRF.render(
                        output_rgbo, image_lengths[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))].reshape(-1, fine_sample_pnum),
                        image_sampled[(50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1)), :, 3:6].reshape(-1, fine_sample_pnum, 3).norm(dim = -1)
                    )          # originally outputs (2500, 3) -> (reshape) (50, 50, 3) -> (to image) (3, 50, 50)
                    render_timer.toc()
                    resulting_image[:, (50 * k):(50 * (k + 1)), (50 * j):(50 * (j + 1))] = part_image.view(50, 50, 3).permute(2, 0, 1)
            test_loss = loss_func(resulting_image, image_input)
            eval_timer.toc()
            writer.add_scalar('Test Loss', loss, test_cnt)
            print("Evaluation in epoch: %4d / %4d\t, test counter: %d test loss: %.4f\taverage time: %.4lf\tavg render time:%lf\tremaining eval time:%s"%(
                    ep, epochs, test_cnt, test_loss.item(), eval_timer.get_mean_time(), render_timer.get_mean_time(), eval_timer.remaining_time(epochs - ep - 1)
            ))
            save_image(resulting_image, "./output/result_%03d.png"%(test_cnt))
            # ======== Saving checkpoints ========
            torch.save({
                'model': coarse_net.state_dict(),
                'optimizer': opt_c.state_dict()},
                "%schkpt_%d_coarse.pt"%(default_chkpt_path, train_cnt)
            )
            torch.save({
                'model': fine_net.state_dict(),
                'optimizer': opt_f.state_dict()},
                "%schkpt_%d_fine.pt"%(default_chkpt_path, train_cnt)
            )
            test_cnt += 1
        epoch_timer.toc()
        print("Epoch %4d / %4d completed\trunning time for this epoch: %.5lf\testimated remaining time: %s"
                %(ep, epochs, epoch_timer.get_mean_time(), epoch_timer.remaining_time(epochs - ep - 1))
        )
    # ======== Saving the model ========
    torch.save({
        'model': coarse_net.state_dict(),
        'optimizer': opt_c.state_dict()},
        "%schkpt_%d_coarse.pth"%(default_model_path, train_cnt)
    )
    torch.save({
        'model': fine_net.state_dict(),
        'optimizer': opt_f.state_dict()},
        "%schkpt_%d_fine.pth"%(default_model_path, train_cnt)
    )
    writer.close()
    print("Output completed.")

if __name__ == "__main__":
    main()


# TODO: 1 当前实现的版本，每一个相机采样的结果并不是标准的camera frustum，其终端是一个球面，而非平面，这是因为采样时将所有光线方向归一化了，但个人觉得影响应该不大