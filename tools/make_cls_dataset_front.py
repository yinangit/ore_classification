import cv2
import numpy as np
import os

def otsu_thresholding(image):

    image = cv2.GaussianBlur(image, (5, 5), 0) # 否则背景会有很多雪花状噪声

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    non_white_cols = np.where(img < 255)[1]

    assert non_white_cols.size != 0

    leftmost_col = non_white_cols.min()

    cropped_image = img[:, leftmost_col:]

    _, binary = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 处理粘连
    # kernel = np.ones((5, 5), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary, leftmost_col

def find_and_crop_objects(binary_image_low, original_image_low, leftmost_col_low, binary_image_high, original_image_high, leftmost_col_high):

    # 使用binary_image对original_image进行掩码
    new_binary_image_low = np.zeros_like(original_image_low)
    new_binary_image_low[:,leftmost_col_low:] = binary_image_low
    new_binary_image_low[new_binary_image_low==255] = 1
    original_image_low = original_image_low * new_binary_image_low

    # 使用binary_image对original_image进行掩码
    new_binary_image_high = np.zeros_like(original_image_high)
    new_binary_image_high[:,leftmost_col_high:] = binary_image_high
    new_binary_image_high[new_binary_image_high==255] = 1
    original_image_high = original_image_high * new_binary_image_high

    assert leftmost_col_high == leftmost_col_low # hack implement

    # 获取轮廓, 以low的为准，low的会大一点
    contours, _ = cv2.findContours(binary_image_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 这里需要8位的二值图像
    
    cropped_images = []
    coordinates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x += leftmost_col_low
        cropped_low = original_image_low[y:y+h, x:x+w]
        croppped_high = original_image_high[y:y+h, x:x+w]
        cropped_images.append([cropped_low, croppped_high])
        coordinates.append((x, y, w, h))
    
    return cropped_images, coordinates

def save_cropped_images(cropped_images, coordinates, image_path_low, save_path):

    base = '_'.join(image_path_low.split('_')[:-1])
    base = '_'.join(base.split('/')[1:])
    
    for idx, (img, coord) in enumerate(zip(cropped_images, coordinates)):
        img_low = img[0]
        img_high = img[1]
        now_save_path = os.path.join(save_path, f"{base}-x{coord[0]}-y{coord[1]}-w{coord[2]}-h{coord[3]}")
        if not os.path.exists(now_save_path):
            os.makedirs(now_save_path)
        savename_low = os.path.join(now_save_path, 'low.tiff')
        cv2.imwrite(savename_low, img_low)
        savename_high = os.path.join(now_save_path, 'high.tiff')
        cv2.imwrite(savename_high, img_high)

def process_image(image_path_low, image_path_high, save_path):

    original_image_low = cv2.imread(image_path_low, cv2.IMREAD_UNCHANGED) # 读取原始位深
    image_low = cv2.imread(image_path_low)
    binary_image_low, leftmost_col_low = otsu_thresholding(image_low)
    

    original_image_high = cv2.imread(image_path_high, cv2.IMREAD_UNCHANGED) # 读取原始位深
    image_high = cv2.imread(image_path_high)
    binary_image_high, leftmost_col_high = otsu_thresholding(image_high)


    cropped_images, coordinates = find_and_crop_objects(binary_image_low, original_image_low, leftmost_col_low, binary_image_high, original_image_high, leftmost_col_high)
    
    save_cropped_images(cropped_images, coordinates, image_path_low, save_path)



if __name__ == "__main__":
    
    sub_folder = ['ore_1200', 'waste_1200']
    image_path = 'data/'
    save_path = 'data/dataset'

    for folder in sub_folder:
        samples_list = os.listdir(os.path.join(image_path, folder))
        high_list = []
        low_list = []
        for file in samples_list:
            if 'low' in file:
                low_list.append(file)
            else:
                high_list.append(file)
        for filename in low_list:
            if filename.endswith('.tiff'):
                print(filename)
                filename_high = filename.replace('low', 'high')
                process_image(os.path.join(image_path, folder, filename), os.path.join(image_path, folder, filename_high), os.path.join(save_path, folder.split('_')[0]))