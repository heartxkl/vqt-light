from projection import new_projection, new_projection_inverse

import cv2
import numpy as np
import os

PI = np.pi
TWO_PI = 2 * PI
Half_PI = PI / 2

'''
Square Equal-Area Projection ---> Equirectangular Projection
'''
def SP2EP(hdr_path, hdr_name, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    square_img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)

    H, _, _ = square_img.shape
    W = H * 2
    print(H, W)

    pano_img = np.zeros((H, W, 3))

    I = np.arange(0, H)
    J = np.arange(0, W)

    II, JJ = np.meshgrid(I, J)
    lat = II * PI / H - Half_PI  # lat的范围是[-pi/2, pi/2]
    lon = JJ * TWO_PI / W  # lon的范围是[0, 2PI]

    lon_adj = (lon - 1 * np.pi / 4) % (2 * np.pi)

    x, y = new_projection(lon_adj, lat)

    X = (-x + 1) * H / 2
    Y = (y + 1) * H / 2

    X = X.astype(int)
    Y = Y.astype(int)

    X[X >= H] = H - 1
    Y[Y >= H] = H - 1

    pano_img[II, JJ] = square_img[X, Y]

    pano_img = np.flip(pano_img, axis=0)

    new_hdr_path = save_path + '/' + hdr_name
    pano_img = pano_img.astype(np.float32)
    cv2.imwrite(new_hdr_path, pano_img)

    img_name = hdr_name.replace('.exr', '.png')
    ldr_img = np.clip(pano_img, 0, 1.0)
    img_path = save_path + '/' + img_name
    cv2.imwrite(img_path, np.uint8(255 * ldr_img))


'''
Equirectangular Projection ---> Square Equal-Area Projection
'''
def EP2SP(img_path, img_name, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pano_img = cv2.imread(img_path)

    pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
    pano_img = pano_img / 255

    H, W, _ = pano_img.shape
    print(H, W)

    resolution = H
    p = 1 - 1 / resolution
    grid_x, grid_y = np.mgrid[
                     -p + 1e-6: p + 1e-6: resolution * 1j, -p: p: resolution * 1j
                     ]  # Pixel centers are not at the edge
    x = grid_x.flatten()
    y = -grid_y.flatten()

    # 经纬度  lat的范围[-PI/2 ~ PI/2], lon范围[0 ~ 2PI]
    lon, lat = new_projection_inverse(x, y)

    lon_adj = (lon - 3 * np.pi / 4) % (2 * np.pi)

    py = np.maximum(np.floor(H * (np.pi / 2 - lat) / np.pi).astype(int), 0)

    px = np.minimum(
        np.maximum(np.floor(W * lon_adj / (2 * np.pi)).astype(int), 0),
        W - 1,
    )

    img = pano_img[py, px].reshape((resolution, resolution, 3))

    img = (img[:, :, :3] * 255).astype(np.uint8)

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path + '/' + img_name, img)


import glob
import os
if __name__ == '__main__':

    hdr_list = glob.glob('./test/*_pred.exr')
    index = 1
    for hdr_path in hdr_list:
        hdr_name = os.path.basename(hdr_path)
        SP2EP(hdr_path, hdr_name, './test/EP')
        print(index)
        index += 1

