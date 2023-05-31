import os
import json
import random
from copy import deepcopy

import configargparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from colorama import Fore, Style
from colorama import init as colorama_init

colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
division_color = [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
styles = [Fore.YELLOW, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

# Draw arrow from https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(origin, end, length, scale=1, color = [1, 0, 0]):
    assert(not np.all(end == origin))
    vec = (end - origin) * length
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return (mesh)

def spatial_division(poses: list):
    """ Quad division """
    division = []
    for pose in poses:
        # quadrant 3: 00, quadrant 2: 01, quadrant 4: 10, quadrant 1: 11
        division.append(((pose[0, 0] > 0) << 1) + (pose[1, 0] > 0))
    cnts = [division.count(i) for i in range(4)]
    sum_cnts = sum(cnts) * 0.01
    print(f"Division information: {cnts[0] / sum_cnts}%, {cnts[1] / sum_cnts}%, {cnts[2] / sum_cnts}%, {cnts[3] / sum_cnts}%")
    return division

def mix_division(divisions: list, shuffle_num: int = 3, rand_seed: int = 114514):
    random.seed(rand_seed)
    if shuffle_num == 0:
        return divisions
    """ Randomly swap parts of the poses with nearby poses """
    to_shuffle = []
    length = len(divisions)
    np_divs = np.int32(divisions)
    for i in range(4):
        idx = np.arange(length)[np_divs == i]
        to_shuffle.append(random.choices(idx, k = shuffle_num << 1))

    left_div = to_shuffle[1]
    right_div = to_shuffle[2]
    div = to_shuffle[0]
    left_div[:shuffle_num], div[:shuffle_num] = div[:shuffle_num], left_div[:shuffle_num]
    right_div[shuffle_num:], div[shuffle_num:] = div[shuffle_num:], right_div[shuffle_num:]
    
    left_div = to_shuffle[2]
    right_div = to_shuffle[1]
    div = to_shuffle[3]
    left_div[:shuffle_num], div[:shuffle_num] = div[:shuffle_num], left_div[:shuffle_num]
    right_div[shuffle_num:], div[shuffle_num:] = div[shuffle_num:], right_div[shuffle_num:]
    for i, idx_list in enumerate(to_shuffle):
        np_divs[idx_list] = i
    return np_divs.tolist()

def visualize_paths(opts):
    train_pose_path  = os.path.join(opts.input_path, opts.name, opts.filename)
    output_pose_path = os.path.join(opts.output_path, opts.name, f"{opts.filename[:-5]}_div.json")

    with open(train_pose_path,'r')as f:     # original json file
        pose_data = json.load(f)
        
    gt_m = []
    gt = []
    for frame in pose_data['frames']:
        matrix = np.array(frame['transform_matrix'], np.float32)
        gt_m.append(matrix[:3, :3])
        gt.append(matrix[:3, -1:])
    if "mix_num" in pose_data:
        divisions = pose_data["division"]
        print("Found pre-computed division, skipping...")
    else:
        print("Calculating division...")
        output_json = deepcopy(pose_data)
        divisions = spatial_division(gt)
        divisions = mix_division(divisions, opts.mix_num)

        output_json["division"] = divisions
        output_json["mix_num"] = opts.mix_num
        for i, div_idx in enumerate(divisions):
            output_json["frames"][i]["div_id"] = div_idx

        with open(output_pose_path, 'w', encoding = 'utf-8') as file:
            json.dump(output_json, file, indent = 4)

    frame_pos = []
    frame_axes = []
    pos_colors = []

    # poses from nerfstudio generated path needs no transformation
    # the output should be transformed by transform_pose, then the result can be used
    for rot_m, trans_t, div_id in zip(gt_m, gt, divisions):
        frame_axes.append([
            rot_m[:3, :3] @ np.float32([0, 0, -2.5]),
            rot_m[:3, :3] @ np.float32([0, -1, 0]),
            rot_m[:3, :3] @ np.float32([-1, 0, 0])
        ])
        frame_pos.append(trans_t)
        pos_colors.append(division_color[div_id])

    frame_pos = np.asarray(frame_pos).squeeze()
    
    colorama_init()
    
    for i in range(4):
        print(f"Division: {styles[i]} {i} is shown in {division_color[i]}{Style.RESET_ALL}.")

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - Rendering path visualization", 1024, 768)
    vis.set_background((0.3, 0.3, 0.3, 1), None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_axes = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame_pos)
    pcd.colors = o3d.utility.Vector3dVector(pos_colors)
    vis.add_geometry('path pos', pcd)

    frame_len = frame_pos.shape[0]
    for i in range(0, frame_len, 1):
        start_p = frame_pos[i]
        arrows = frame_axes[i]
        for j in range(3):
            end_p = start_p + arrows[j]
            arrow = get_arrow(start_p, end_p, length = 0.2, scale=0.8, color = colors[j])
            vis.add_geometry(f"arrow{3 * i + j}", arrow)
    
    # for idx in range(0, cmp_pts.shape[0]):
    #     vis.add_3d_label(cmp_pts[idx], "{}".format(idx+1))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    app.quit()
    
def get_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--name"       , type = str, required = True, help = "Input json scene name")
    parser.add_argument("--filename"   , type = str, default = "transforms_train.json", help = "Input json scene name")
    parser.add_argument("--mix_num"    , type = int, default = 3, help = "Number of poses to mix with the adjacent division")
    
    parser.add_argument("--input_path" , type = str, default = "../../nerf_synthetic/", help = "Input json path")
    parser.add_argument("--output_path", type = str, default = "../../nerf_synthetic/", help = "Output json path")
    return parser.parse_args()

if __name__ == "__main__":
    opts = get_parser()
    visualize_paths(opts)
    