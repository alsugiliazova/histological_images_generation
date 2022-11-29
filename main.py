from scipy import *
from scipy import ndimage
from PIL import Image
import numpy as np
import os
import cv2 as cv

# calculates weight center
def calc_weight_center(x):
    max_val = np.max(x)
    min_val = np.min(x)
    X = 1 - (np.array(x) - min_val) / (max_val - min_val)
    center = [0, 0]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            center[0] += i * X[i][j]
            center[1] += j * X[i][j]
    sum = int(np.round(np.sum(X)))
    center[0]=int(center[0]/sum)
    center[1]=int(center[1]/sum)
    return center 

# moves component on image
def make_new_matrix(x, shift):
    new_matrix = np.zeros(x.shape)
    s = (int(shift[0]), int(shift[1]))
    for i in range(max(0, s[0]), min(x.shape[0], x.shape[0] + s[0])):
        for j in range(max(0, s[1]), min(x.shape[1], x.shape[1] + s[1])):
            new_matrix[i][j] = x[i - s[0]][j - s[1]]
    return new_matrix


def make_dataset(src_dataset_paths: list, folder: str):
    '''
    src_dataset_path containes masks folder and images folder
    '''
    num_img = 0
    max_i, max_j = -1, -1
    flag_i_j = True
    
    for path in src_dataset_paths:
        for file in os.listdir(os.path.join(path, 'masks')):
            src_dataset_path = path
            components = [] 
            cells = []
            f_mask_name = os.path.join(os.path.join(src_dataset_path, 'masks', file))
            mask = Image.open(f_mask_name).convert('L')
            mask = np.asarray(mask)

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

            full_num_objects = 0
            flag = 0
                    
            #creating array of masks for each component
            for k in range(1, num_objects+1):
                single_component = np.zeros(mask.shape)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if labeled[i][j] == k:
                            if i == 0 or j == 0 or i == mask.shape[0]-1 or j == mask.shape[1]-1:
                                flag = 1
                                break
                            single_component[i][j] = 1
                        if flag:
                            break
                if not flag:
                    components.append(single_component)
                    full_num_objects += 1
                else:
                    flag = 0
            
            # file with images
            f_img_name = os.path.join(src_dataset_path, 'images', file)
            img = cv.imread(f_img_name)
        
            for i in range(0, full_num_objects):
                mask = components[i].astype(np.uint8)
                mask = np.expand_dims(mask, axis=2)
                cells.append(img*mask)
            
            # TODO
            if flag_i_j:
                for i in range(0, full_num_objects):
                    test = (components[i]-1)*(-1) 
                    
                    # shift cell to center of image
                    weight_center = calc_weight_center(test)
                    shift = np.array([test.shape[0] / 2, test.shape[1] / 2]) - weight_center
    
                    
                    mask_cv = test.astype(np.uint8)
                    cnts, _ = cv.findContours(mask_cv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    _, _, j, i = cv.boundingRect(cnts[1])
                    if i > max_i:
                        max_i = i
                    if j > max_j:
                        max_j = j
                max_i = max_i + 100
                max_j = max_j + 100
                flag_i_j = 0

            for i in range(0, full_num_objects):
                test = (components[i]-1)*(-1) 
                
                # shift cell to center of image
                weight_center = calc_weight_center(test)
                shift = np.array([test.shape[0] / 2, test.shape[1] / 2]) - weight_center
                
                cur_img = cells[i]
                b,g,r = cv.split(cur_img)
                mask = make_new_matrix(test, shift)

                # crop image and mask
                center = test.shape
                w, h  = max(max_i, max_j), max(max_i, max_j)
                x = center[1]/2 - w/2
                y = center[0]/2 - h/2
                

                test_b = make_new_matrix(b, shift)[int(y):int(y+h), int(x):int(x+w)]
                test_r = make_new_matrix(r, shift)[int(y):int(y+h), int(x):int(x+w)]
                test_g = make_new_matrix(g, shift)[int(y):int(y+h), int(x):int(x+w)]
                test_img_ = cv.merge([test_r,test_g,test_b])
                mask = np.array(test_b != 0, dtype=int)*255


                images_dir = os.path.join(folder, 'images')
                masks_dir = os.path.join(folder, 'masks')

                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                cv.imwrite(os.path.join(images_dir, 'img_{}.bmp'.format(num_img)), test_img_)

                if not os.path.exists(masks_dir):
                    os.makedirs(masks_dir)
                cv.imwrite(os.path.join(masks_dir, 'img_{}.bmp'.format(num_img)), mask)

                num_img += 1
            flag = 1
        


if __name__ == '__main__':
    src_data_paths = ["C:/workspace/cell/cw/selected_dataset/selected_from_val",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_test",
                     "C:/workspace/cell/cw/selected_dataset/selected_from_train"]
    
    destination_folder_path = "C:/workspace/cell/dataset/single_cells_dataset"

    images_dir = os.path.join(destination_folder_path, 'images')
    masks_dir = os.path.join(destination_folder_path, 'masks')
    for f in os.listdir(images_dir):
        os.remove(os.path.join(images_dir, f))
        os.remove(os.path.join(masks_dir, f))

    make_dataset(src_data_paths, destination_folder_path)

   