import cv2
import numpy as np

def rgb_to_gray(rgb_img):
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    gray_img = 0.299 * r + 0.587 * g + 0.114 * b
    gray_img = gray_img.astype(np.uint8)
    return gray_img

def thresholdfunc(img, threshold):
    mask = np.where(img > threshold, 255, 0).astype(np.uint8)
    return mask
def erodefunc(img,kernel):
    height,width=img.shape[:2]
    result = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            result[i, j] = np.min(img[i:i + 2, j:j + 2] * kernel)
    return result

def dfs(image, visited, x, y, label_count, label):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and image[cx, cy] == 255 and not visited[cx, cy]:
            visited[cx, cy] = True
            label[cx, cy] = label_count
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

def connected_components(image):
    label = np.zeros_like(image, dtype=np.uint32)
    visited = np.zeros_like(image, dtype=bool)
    label_count = 0

    for (x, y), val in np.ndenumerate(image):
        if val == 255 and not visited[x, y]:
            label_count += 1
            dfs(image, visited, x, y, label_count, label)

    return label_count, label

def find_center_width_height(label_image, label):
    indices = np.argwhere(label_image == label)
    min_y, min_x = indices.min(axis=0)
    max_y, max_x = indices.max(axis=0)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    return center_x, center_y, width, height

def draw_rectangle(image, center_x, center_y, width, height):
    top_left = (int(center_x - width / 2), int(center_y - height / 2))
    bottom_right = (int(center_x + width / 2), int(center_y + height / 2))

    image[top_left[1] - 2:top_left[1], top_left[0]:bottom_right[0] + 2] = [0, 255, 0]
    image[bottom_right[1]:bottom_right[1] + 2, top_left[0]:bottom_right[0] + 2] = [0, 255, 0]

    image[top_left[1] - 2:bottom_right[1] + 2, top_left[0] - 2:top_left[0]] = [0, 255, 0]
    image[top_left[1] - 2:bottom_right[1] + 2, bottom_right[0]:bottom_right[0] + 2] = [0, 255, 0]

def add_alpha_channel(image):
    alpha_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    result = np.dstack((image, alpha_channel))

    return result
def main():
    orgimg=cv2.imread("dublin.jpg",cv2.IMREAD_UNCHANGED)
    edimg=cv2.imread("dublin_edited.jpg",cv2.IMREAD_UNCHANGED)
    orgimg = orgimg.astype(np.int32)
    edimg = edimg.astype(np.int32)
    img_diff = np.abs(orgimg - edimg)
    img_diff = np.clip(img_diff, 0, 255)
    img_diff = img_diff.astype(np.uint8)
    gray_img=rgb_to_gray(img_diff)
    mask=thresholdfunc(gray_img,4)
    kernel = np.ones((2, 2), np.uint8)
    eroded=erodefunc(mask,kernel)
    label_count, label_image = connected_components(eroded)
    max_area=0
    largest_object=0
    for i in range(1, label_count + 1):
        center_x, center_y, width, height = find_center_width_height(label_image, i)
        if width * height > 25:
            draw_rectangle(edimg, center_x, center_y, width, height)
        if width*height>max_area:
            max_area=width*height
            largest_object=i

    edimg=add_alpha_channel(edimg)

    center_x, center_y, width, height = find_center_width_height(label_image, largest_object)
    top_left_x, top_left_y = max(0, center_x - width // 2), max(0, center_y - height // 2)
    bottom_right_x, bottom_right_y = min(edimg.shape[1], center_x + width // 2), min(edimg.shape[0], center_y + height // 2)

    cutout_image = np.copy(edimg)
    largest_object_mask = (label_image == largest_object).astype(np.uint8)
    cutout_image[:, :, 3] = largest_object_mask * 255
    cutout_image = cutout_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    cv2.imshow('win', edimg.astype(np.uint8))
    cv2.waitKey()

    cv2.imwrite('win.png', cutout_image)
    cv2.waitKey()

if __name__ == "__main__":
    main()