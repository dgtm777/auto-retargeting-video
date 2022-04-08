import torch
import numpy as np
from copy import copy
from constants import transform
import torchvision.transforms as transforms
from PIL import Image


l_prev = None
r_prev = None
speed_upgrade = 0
speed_prev = None


def check_border(crop_size, mask, scale, vect, constant_speed=None):
    global l_prev, r_prev, speed_upgrade, speed_prev
    compress_arr = np.sum(mask**5, 1)
    prefix_sum = np.insert(np.cumsum(compress_arr), 0, 0)
    dif_sum = [
        prefix_sum[r] - prefix_sum[r - int(crop_size * scale)]
        for r in range(int(crop_size * scale), len(mask))
    ]
    l_new = int(np.argmax(dif_sum) / scale)
    r_new = l_new + crop_size
    if r_new > len(mask) / scale:
        r_new = len(mask) / scale
        l_new = r_new - crop_size

    if l_prev is not None and l_prev != l_new:
        general_sign = (l_prev - l_new) / abs(l_prev - l_new)
        l_new += int(general_sign * 1 / scale)
        r_new += int(general_sign * 1 / scale)
    if constant_speed is None:
        if (l_prev is not None) and l_new - l_prev != 0:
            if (l_new - l_prev) / abs(l_new - l_prev) >= 0:
                non_zero = np.count_nonzero(vect[l_new:r_new] > 128)
                max_speed = max(vect.max() - 128, 0)
            else:
                non_zero = np.count_nonzero(vect[l_new:r_new] < 128)
                max_speed = min(vect.min() - 128, 0)
            if non_zero > len(mask[0]) * crop_size * 0.01:
                speed_upgrade = max_speed / 10000
            else:
                speed_upgrade = 0
        if speed_prev is not None:
            speed_upgrade = speed_upgrade * 0.3 + speed_prev * 0.7
        speed_prev = speed_upgrade
    else:
        speed_upgrade = constant_speed
    if l_prev is None or (
        # abs(l_new - l_prev) >= len(mask) / scale * 0.25
        abs(l_new - l_prev) > 2 / 3 * (r_new - l_new)
        and abs(
            (
                prefix_sum[int((r_new - general_sign * 1 / scale) * scale)]
                - prefix_sum[int((l_new - general_sign * 1 / scale) * scale)]
            )
            - (prefix_sum[int(r_prev * scale)] - prefix_sum[int(l_prev * scale)])
        )
        > (5 * crop_size) ** 5
    ):
        l_prev = l_new
        r_prev = r_new
    elif l_prev is not None and l_new != l_prev:
        div = int(len(mask) / scale * speed_upgrade)
        l_prev += div
        r_prev += div

    return l_prev, r_prev


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_mask(img, net):
    # sig = torch.nn.Sigmoid()
    mask = net(torch.unsqueeze(transform["test"]["images"](img), 0))[0].detach().numpy()
    mask = sigmoid(mask[0]) * 255
    return mask


def crop(img, net, transform, crop_size, mask, vect):
    out_mask = np.array([mask for i in range(3)])
    mask_numpy = np.moveaxis(out_mask, 0, 2)
    my_img = np.array(img.transpose(2, 0, 1))
    if crop_size[0] != len(my_img[0]):
        h, b = check_border(crop_size[0], mask, len(mask) / len(my_img[0]), vect[0])
        l, r = 0, len(my_img[0][0])
    else:
        l, r = check_border(
            crop_size[1],
            mask.transpose(1, 0),
            len(mask[0]) / len(my_img[0][0]),
            vect[0].transpose(1, 0),
        )
        h, b = 0, len(my_img[0])

    ans = copy(my_img[:, h:b, l:r])

    my_img[0, max(h - 1, 0) : min(len(my_img[0]), h + 1), :] = 255
    my_img[0, max(b - 1, 0) : min(len(my_img[0]), b + 1), :] = 255
    my_img[0, :, max(l - 1, 0) : min(len(my_img[0][0]), l + 1)] = 255
    my_img[0, :, max(r - 1, 0) : min(len(my_img[0][0]), r + 1)] = 255
    ans = np.moveaxis(ans, 0, 2)
    my_img = np.moveaxis(my_img, 0, 2)

    mask_numpy = np.asarray(
        transforms.Resize((len(my_img), len(my_img[0])))(
            Image.fromarray(np.uint8(mask_numpy))
        )
    )

    croped_mask = copy(mask_numpy[h:b, l:r, :])

    for i, val in enumerate([255, 0, 0]):
        mask_numpy[max(h - 1, 0) : min(len(mask_numpy), h + 1), :, i] = val
        mask_numpy[max(b - 1, 0) : min(len(mask_numpy), b + 1), :, i] = val
        mask_numpy[:, max(l - 1, 0) : min(len(mask_numpy[0]), l + 1), i] = val
        mask_numpy[:, max(r - 1, 0) : min(len(mask_numpy[0]), r + 1), i] = val

    return (my_img, ans), (mask_numpy, croped_mask)