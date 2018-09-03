import scipy.misc
import numpy as np
'''
def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2))
    i = int(round((w - crop_w)/2))

    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
'''
def center_crop(img, target_h, target_w=None):
    if not target_w:
        target_w = target_h
    h, w = img.shape[:2]
    k = min(h/target_h, w/target_w)
    center_h = int(k*target_h)
    center_w = int(k*target_w)
    j = (h - center_h) // 2
    i = (w - center_w) // 2
    return scipy.misc.imresize(img[j:j+center_h, i:i+center_w,:], [target_h, target_w])


def transform(image, input_height, input_width,
            resize_height=64, resize_width=64, crop=True):
    if crop:
        #cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
        cropped_image = center_crop(image, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5-1

def get_image(image_path, input_height, input_width,
            resize_height=64, resize_width=64, crop=True):
    image = scipy.misc.imread(image_path)#.astype(np.float)

    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def imread(path):
    '''
    return: int8 array
    '''
    return scipy.misc.imread(path)#.astype(np.float)

def merge(images, size):
    '''
    images: [number, height, width, channel]
    size:   (rows, columns)
    '''
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def to_rgb(image):
    return (image + 1) / 2

def get_layout(n):
    start = int(n ** 0.5)
    while start > 1:
        if n % start == 0:
            return start, n // start
        start -= 1
    return 1, n

if __name__ == '__main__':
    import glob
    paths = glob.glob('./data/faces/*.jpg')[:10]
    import matplotlib.pyplot as plt
    for img in paths:
        image = imread(img)
        print(image.shape)
        print(image.dtype)
        print(np.max(image), np.min(image))
        plt.imshow(imread(img))
        plt.show()
        image = to_rgb(get_image(img, 96, 96, 64, 64))
        print(image.shape)
        print(image.dtype)
        print(np.max(image), np.min(image))
        plt.imshow(image)
        plt.show()
