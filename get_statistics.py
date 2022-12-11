from scipy import *
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import cv2 as cv


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


def get_size(src_dataset_paths: list):
    '''
    return statistics about cell sizes
    '''
    num_img = 0
    widths = []
    heights = []
    
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
            
            for i in range(0, full_num_objects):
                cur_mask = components[i]
                cnts, _ = cv.findContours(cur_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv.boundingRect(cnts[0])
                heights.append(h)
                widths.append(w)
                num_img += 1
    h_q = np.quantile(heights, 0.95)
    w_q = np.quantile(widths, 0.95)
    return h_q, w_q
        
        


if __name__ == '__main__':
    src_data_paths = ["C:/workspace/cell/cw/selected_dataset/selected_from_val",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_test",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_train"]

    size = get_size(src_data_paths)
    print(size)


   