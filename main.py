from scipy import *
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import cv2 as cv
import get_statistics


def move_to_center(img: np.array, size: tuple):
    """
    moves object to center
    """
    cnts, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    a, b, j, i = cv.boundingRect(cnts[0])
    if size[0] < i or size[1] < j:
        return 0, img
    x_pad = max(0, size[0]-i) // 2
    y_pad = max(0, size[1]-j) // 2
    img = img[b:b+i, a:a+j]
    img = np.pad(img, ((x_pad, x_pad), (y_pad, y_pad)))
    return 1, img


def find_connected_components(mask: np.array):
    """
    gets mask of the image
    returns array of components on the mask and 
    a number of connected components
    """
    # smooth the image (to remove small objects)
    blur_radius = 1.0
    maskf = ndimage.gaussian_filter(mask, blur_radius)
    threshold = 50
    
    # erosion (to split up close cells)
    kernel_size = 20
    kernel = np.ones((kernel_size, kernel_size))
    mask_erosion = cv.erode(maskf, kernel)

    # find connected components
    labeled, num_objects = ndimage.label(mask_erosion > threshold) 
    return labeled, num_objects


def make_dataset(src_dataset_paths: list, dest_folder: str, size: tuple):
    '''
    creates dataset for pix2pix input
    src_dataset_path : containes masks folder and images folder
    dest_folder : will contain masks folder and images folder
    size : size of image in datset for pix2pix
    '''
    num_img = 0
    
    for path in src_dataset_paths:
        for file in os.listdir(os.path.join(path, 'masks')):
            f_mask_name = os.path.join(os.path.join(path, 'masks', file))
            mask = cv.imread(f_mask_name)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

            # find connected components
            labeled, num_objects = find_connected_components(mask) 

            full_num_objects = 0
            components = [] 
            cells = []

            #creating array of masks for each component
            for k in range(1, num_objects+1):
                #single_component = np.zeros(mask.shape)
                single_component = (labeled == k).astype(np.uint8)        
                
                #check if cell doesn't touches the edges of image
                x = np.pad(np.zeros((mask.shape[0]-2, mask.shape[1]-2)), pad_width=1, mode='constant', constant_values=1) 
                if np.sum(single_component*x) == 0:
                    components.append(single_component)
                    full_num_objects += 1
            
            # file with corresponding image
            f_img_name = os.path.join(path, 'images', file)
            img = cv.imread(f_img_name)
        
            # separate image to single cells dataset
            for i in range(0, full_num_objects):
                mask = components[i].astype(np.uint8)
                mask = np.expand_dims(mask, axis=2)
                cells.append(img*mask)
            

            for i in range(0, full_num_objects):
                cur_mask = components[i]
                cur_img = cells[i]
                b,g,r = cv.split(cur_img)
                mask = move_to_center(cur_mask, size)
                if mask[0]:
                    mask = np.expand_dims(mask[1], axis=2)*255
                    test_b = move_to_center(b, size)[1]
                    test_r = move_to_center(r, size)[1]
                    test_g = move_to_center(g, size)[1]
                    test_img_ = cv.merge([test_r,test_g,test_b])
               
                    images_dir = os.path.join(dest_folder, 'images')
                    masks_dir = os.path.join(dest_folder, 'masks')

                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)
                    cv.imwrite(os.path.join(images_dir, 'img_{}.bmp'.format(num_img)), test_img_)

                    if not os.path.exists(masks_dir):
                        os.makedirs(masks_dir)
                    cv.imwrite(os.path.join(masks_dir, 'img_{}.bmp'.format(num_img)), mask)

                    num_img += 1
        
        


if __name__ == '__main__':
    src_data_paths = ["C:/workspace/cell/cw/selected_dataset/selected_from_val",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_test",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_train"]
    
    destination_folder_path = "C:/workspace/cell/pytorch-CycleGAN-and-pix2pix/cell_data"

    images_dir = os.path.join(destination_folder_path, 'images')
    masks_dir = os.path.join(destination_folder_path, 'masks')
    if os.path.exists(images_dir) and os.path.exists(masks_dir):
        for f in os.listdir(images_dir):
            os.remove(os.path.join(images_dir, f))
            os.remove(os.path.join(masks_dir, f))

    size = get_statistics.get_size(src_data_paths)
    size = (int(max(size)), int(max(size)))
    #size = (263, 263)
    make_dataset(src_data_paths, destination_folder_path, size)

   