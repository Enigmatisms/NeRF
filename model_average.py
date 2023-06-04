#-*-coding:utf-8-*-
"""
    NeRF with mip ideas baseline model
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from nerf.param_com import *
from nerf.timer import Timer
from nerf.mip_model import MipNeRF
from nerf.nerf_base import DecayLrScheduler, NeRF
from nerf.local_shuffler import LocalShuffleSampler
from nerf.dataset import CustomDataSet, AdaptiveResize
from torchvision.utils import save_image
from nerf.nerf_helper import nan_hook, saveModel
from nerf.mip_methods import maxBlurFilter
from nerf.procedures import render_image, get_parser
from nerf.utils import fov2Focal, getSummaryWriter, validSampler, randomFromOneImage, inverseSample
from nerf.addtional import getBounds, ProposalLoss, ProposalNetwork, LossPSNR
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast as autocast


default_chkpt_path = "./check_points/"
default_model_path = "./model/"

def train(gpu, args):
    torch.manual_seed(0)
    epochs             = args.epochs
    sample_ray_num     = args.sample_ray_num
    coarse_sample_pnum = args.coarse_sample_pnum
    fine_sample_pnum   = args.fine_sample_pnum
    near_t             = args.near
    far_t              = args.far
    
    center_crop_iter   = args.center_crop_iter
    center_crop        = (args.center_crop_x, args.center_crop_y)
    eval_time          = args.eval_time
    dataset_name       = args.dataset_name
    load_path_mip      = default_chkpt_path + args.name + "_mip.pt"
    load_path_prop     = default_chkpt_path + args.name + "_prop.pt"

    # Bool options
    del_dir            = args.del_dir
    use_load           = args.load
    debugging          = args.debug
    use_amp            = (args.use_scaler and (not debugging))
    use_white_bkg      = args.white_bkg
    render_normal      = args.render_normal

    render_depth       = args.render_depth
    output_time        = args.output_time
    img_scale          = args.img_scale
    scene_scale        = args.scene_scale
    opt_mode           = args.opt_mode        
    grad_clip_val      = args.grad_clip
    actual_lr          = args.lr * sample_ray_num / 512        # bigger batch -> higher lr (linearity)
    ma_epoch           = args.ma_epoch
    ma_method          = args.ma_method
    
    train_cnt, ep_start = None, None

    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    if use_amp:
        try:
            from apex import amp
        except ModuleNotFoundError:
            print("Nvidia APEX module is not found. Only native model is available")

    for folder in ("./output/", "./check_points/", "./model/"):
        if not os.path.exists(folder):
            os.mkdir(folder)

    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    
    # ======= instantiate model =====
    # NOTE: model is recommended to have loadFromFile method
    
    # This is used to receive all information
    container = MipNeRF(10, 4, hidden_unit = args.nerf_net_width).cuda(gpu)
    container.requires_grad_(False)
    
    mip_net = MipNeRF(10, 4, hidden_unit = args.nerf_net_width).cuda(gpu)
    prop_net = ProposalNetwork(10, hidden_unit = args.prop_net_width).cuda(gpu)
    # Wrap the model

    if debugging:
        for submodule in mip_net.modules():
            submodule.register_forward_hook(nan_hook)
        torch.autograd.set_detect_anomaly(True)

    # ======= Loss function ==========
    loss_func = torch.nn.MSELoss()
    prop_loss_func = ProposalLoss().cuda(gpu)
    mse2psnr = LossPSNR()
    # ======= Optimizer and scheduler ========

    transform_funcs = transforms.Compose([
        AdaptiveResize(img_scale),
        transforms.ToTensor(),
    ])

    # ============ Loading dataset ===============
    # For model average uses, we should 
    trainset = CustomDataSet(f"../dataset/{dataset_name}/", transform_funcs, 
        scene_scale, True, use_alpha = False, white_bkg = use_white_bkg, use_div = opt.div)
    testset = CustomDataSet(f"../dataset/{dataset_name}/", transform_funcs, 
        scene_scale, False, use_alpha = False, white_bkg = use_white_bkg)
    cam_fov_train, _ = trainset.getCameraParam()
    r_c = trainset.r_c()
    
    # ============= Buiding dataloader ===============

    # train_set separation
    division = 4 if trainset.divisions is None else trainset.divisions
    train_sampler = LocalShuffleSampler(trainset, indices=division, num_replicas = args.world_size, 
                                                rank = rank, allow_imbalance = args.allow_imbalanced)
    train_loader = DataLoader(dataset=trainset, batch_size=1,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=train_sampler)
    
    # TODO: what about the test loader? Maybe we don't need to load
    cam_fov_test, _ = testset.getCameraParam()
    
    train_focal = fov2Focal(cam_fov_train, r_c)
    test_focal = fov2Focal(cam_fov_test, r_c)
    print("Training focal: (%f, %f), image size: (w: %d, h: %d)"%(train_focal[0], train_focal[1], r_c[1], r_c[0]))

    grad_vars = list(mip_net.parameters()) + list(prop_net.parameters())
    opt = optim.Adam(params = grad_vars, lr = actual_lr, betas=(0.9, 0.999))
    
    def grad_clip_func(parameters, grad_clip):
        if grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)

    if use_amp:
        if opt_mode.lower() != "native":
            [mip_net, prop_net], opt = amp.initialize([mip_net, prop_net], opt, opt_level=opt_mode)
        else:
            scaler = GradScaler()
    if use_load == True and os.path.exists(load_path_mip) and os.path.exists(load_path_prop):
        train_cnt, ep_start = mip_net.loadFromFile(load_path_mip, use_amp and opt_mode != "native", opt, ["train_cnt", "epoch"])
        prop_net.loadFromFile(load_path_prop, use_amp and opt_mode != "native")
    else:
        print("Not loading or load path '%s' / '%s' does not exist."%(load_path_mip, load_path_prop))
    lr_sch = DecayLrScheduler(args.min_ratio, args.decay_rate, args.decay_step, actual_lr, args.warmup_step)
    
    # Only two views therefore we do not need testloader
    test_views = []
    for i in (1, 4):
        test_views.append(testset[i])
    del testset
    torch.cuda.empty_cache()

    # ====== tensorboard summary writer ======
    if rank == 0:
        writer = getSummaryWriter(epochs, del_dir)
    train_set_len = len(trainset)

    if ep_start is None:
        ep_start = 0
        train_cnt = ep_start * train_set_len
    test_cnt = ep_start // 20
    train_timer, eval_timer, epoch_timer = Timer(5), Timer(5), Timer(3)
    for ep in range(ep_start, epochs):
        epoch_timer.tic()
        for i, (train_img, train_tf) in enumerate(train_loader):
            train_img = train_img.cuda().squeeze(0)
            train_tf = train_tf.cuda().squeeze(0)
            train_timer.tic()
            now_crop = (center_crop if train_cnt < center_crop_iter else (1., 1.))
            valid_pixels, valid_coords = randomFromOneImage(train_img, now_crop)

            # sample one more t to form (coarse_sample_pnum) proposal interval
            coarse_samples, coarse_lengths, rgb_targets, coarse_cam_rays = validSampler(
                valid_pixels, valid_coords, train_tf, sample_ray_num, coarse_sample_pnum, train_focal, near_t, far_t, True
            )
            # output 
            def run():
                density = prop_net.forward(coarse_samples)
                density = F.softplus(density)
                prop_weights_raw = ProposalNetwork.get_weights(density, coarse_lengths, coarse_cam_rays[:, 3:])      # (ray_num, num of proposal interval)
                prop_weights = maxBlurFilter(prop_weights_raw, 0.01)

                coarse_normal_loss = normal_loss = bf_loss = 0.
                fine_lengths, below_idxs = inverseSample(prop_weights, coarse_lengths, fine_sample_pnum + 1, sort = True)
                fine_lengths = fine_lengths[..., :-1]
                fine_samples = NeRF.length2pts(coarse_cam_rays, fine_lengths)
                fine_rgbo = mip_net.forward(fine_samples)
                fine_rendered, weights, _ = NeRF.render(fine_rgbo, fine_lengths, coarse_cam_rays[:, 3:])
                weight_bounds:torch.Tensor = getBounds(prop_weights, below_idxs)             # output shape: (ray_num, num of conical frustum)
                img_loss:torch.Tensor = loss_func(fine_rendered, rgb_targets)                           # stop the gradient of NeRF MLP 

                prop_loss:torch.Tensor = prop_loss_func(weight_bounds, weights.detach())                # stop the gradient of NeRF MLP 
                loss:torch.Tensor = prop_loss + img_loss + 4e-4 * (normal_loss + 0.1 * coarse_normal_loss)  + 0.1 * bf_loss
                return loss, img_loss
            if use_amp:
                if opt_mode == "native":
                    with autocast():
                        loss, img_loss = run()
                        scaler.scale(loss).backward()
                        grad_clip_func(grad_vars, grad_clip_val)
                        scaler.step(opt)
                        scaler.update()
                else:
                    loss, img_loss = run()
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                    grad_clip_func(grad_vars, grad_clip_val)
                    opt.step()
            else:
                loss, img_loss = run()
                loss.backward()
                grad_clip_func(grad_vars, grad_clip_val)
                opt.step()
            opt.zero_grad()

            train_timer.toc()

            opt, new_lr = lr_sch.update_opt_lr(train_cnt, opt)
            if train_cnt % eval_time == 1:
                # ========= Evaluation output ========
                remaining_cnt = (epochs - ep - 1) * train_set_len + train_set_len - i
                psnr = mse2psnr(img_loss)
                print("Traning Epoch: %4d / %4d\t Iter %4d / %4d\ttrain loss: %.4f\tPSNR: %.3lf\tlr:%.7lf\tcenter crop:%.1lf, %.1lf\tremaining train time:%s"%(
                        ep, epochs, i, train_set_len, loss.item(), psnr, new_lr, now_crop[0], now_crop[1], train_timer.remaining_time(remaining_cnt)
                ))
                if rank == 0:
                    writer.add_scalar('Train Loss', loss, train_cnt)
                    writer.add_scalar('Learning Rate', new_lr, train_cnt)
                    writer.add_scalar('PSNR', psnr, train_cnt)
            if train_cnt % ma_epoch == 0:
                # double barrier to ensure synchronized sending / receiving
                dist.barrier()
                print("Using model average...")
                if ma_method == 'p2p':
                    if rank == 0:
                        param_recv(container, source_rank = 1)
                        network_average(container, mip_net)
                        param_send(mip_net, dist_ranks = [1])
                    else:
                        param_send(mip_net, dist_ranks = [0])
                        # Then other process has nothing to do but waiting for the reduce and resend
                        # Directly replace the network with the averaged parameters
                        param_recv(mip_net, source_rank = 0)
                elif ma_method == 'broadcast':      # reduce-broadcast
                    pass
                elif ma_method == 'all_reduce':      # all-reduce (one-step reduce-broadcast)
                    pass
                else:
                    pass
                dist.barrier()
            train_cnt += 1
            train_sampler.set_epoch(train_cnt)

        if ((ep % output_time == 0) or ep == epochs - 1):
            mip_net.eval()
            prop_net.eval()
            with torch.no_grad():
                eval_timer.tic()
                test_results = []
                test_loss = 0.
                for test_img, test_tf in test_views:
                    test_result = render_image(
                        mip_net, prop_net, test_tf.cuda(), r_c, test_focal, near_t, far_t, fine_sample_pnum, 
                        white_bkg = use_white_bkg, render_depth = render_depth, render_normal = render_normal
                    )
                    for value in test_result.values():
                        test_results.append(value)
                    test_loss += loss_func(test_result["rgb"], test_img.cuda())
                eval_timer.toc()
                print("Evaluation in epoch: %4d / %4d\t, test counter: %d test loss: %.4f\taverage time: %.4lf\tremaining eval time:%s"%(
                        ep, epochs, test_cnt, test_loss.item() / 2, eval_timer.get_mean_time(), eval_timer.remaining_time(epochs - ep - 1)
                ))
                save_image(test_results, "./output/result_%03d.png"%(test_cnt), nrow = 1 + render_normal + render_depth)
                # ======== Saving checkpoints ========
                if rank == 0:
                    writer.add_scalar('Test Loss', loss, test_cnt)
                    saveModel(mip_net,  f"{default_chkpt_path}chkpt_{(train_cnt % args.max_save) + 1}_mip.pt",
                                         {"train_cnt": train_cnt, "epoch": ep}, opt = opt, amp = (amp) if use_amp and opt_mode != "native" else None)
                    saveModel(prop_net,  f"{default_chkpt_path}chkpt_{(train_cnt % args.max_save) + 1}_prop.pt", 
                                        opt = None, amp = (amp) if use_amp and opt_mode != "native" else None)
                test_cnt += 1
            mip_net.train()
            prop_net.train()
            
        if not args.allow_imbalanced:
            dist.barrier()  
        epoch_timer.toc()

        print("Epoch %4d / %4d completed\trunning time for this epoch: %.5lf\testimated remaining time: %s"
                %(ep, epochs, epoch_timer.get_mean_time(), epoch_timer.remaining_time(epochs - ep - 1))
        )
    # ======== Saving the model ========
    if rank == 0:
        saveModel(mip_net, f"{default_model_path}model_mip.pth", opt = opt, amp = (amp) if use_amp and opt_mode != "native" else None)
        saveModel(prop_net, f"{default_model_path}model_prop.pth", opt = None, amp = (amp) if use_amp and opt_mode != "native" else None)
        writer.close()
    print("Output completed.")

def main():
    parser = get_parser()
    # Distributed model settings
    parser.add_argument('--ma_epoch', required=True, type = int,
                        help='Model average will be used each <ma_epoch> epoch')
    parser.add_argument('--ma_method', choices=['p2p', 'broadcast', 'delicate', 'all_reduce'], type = 'p2p',
                        help='Model average strategies')
    
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-div', '--div', default=False, action = 'store_true',
                        help='Whether to use divided dataset')
    parser.add_argument('--allow_imbalanced', default=False, action = 'store_true',
                        help='Whether to allow imbalanced dataset')

    args = parser.parse_args()      # spherical rendering is disabled (for now)

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '177.177.94.23'
    os.environ['MASTER_PORT'] = '11451'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__ == "__main__":
    main()
