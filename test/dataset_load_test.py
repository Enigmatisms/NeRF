#-*-coding:utf-8-*-

"""
    Load dataset from json.
    Blender synthetic realistic dataset from NeRF
"""

import matplotlib.pyplot as plt
from torchvision import transforms

import sys
sys.path.append("..")
from py.dataset import CustomDataSet, DATASET_PREFIX

tf_func = transforms.ToTensor()
    
def loadSceneTest(path:str, is_train:bool):
    # do not shuffle images when training, since the information of each image is sorted
    image_folder = CustomDataSet(path, transform = tf_func, is_train = is_train)
    cam_fov, tfs, all_imgs = image_folder.get_dataset()
    print(all_imgs.shape)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(all_imgs[i * 20].permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    # readFromJson(DATASET_PREFIX + "drums/transforms_train.json")
    loadSceneTest(DATASET_PREFIX + "drums/", False)
