import torch
import numpy as np
from copy import copy
from constants import transform
import torchvision.transforms as transforms
from PIL import Image
import math


l_prev = None
r_prev = None
speed_upgrade = 0
speed_prev = 0


def speed(mask, crop_size, vect, parameters):
    global l_prev, r_prev
    if crop_size[0] != len(vect[0]):
        l_new, r_new, prefix_sum, sign = find_new_borders(
            mask,
            l_prev,
            r_prev,
            crop_size[0],
            len(mask) / len(vect[0]),
            parameters,
        )
        len_vect = len(vect[0])
    else:
        l_new, r_new, prefix_sum, sign = find_new_borders(
            mask.transpose(1, 0),
            l_prev,
            r_prev,
            crop_size[1],
            len(mask[0]) / len(vect[0][0]),
            parameters,
        )
        len_vect = len(vect[0][0])
    if parameters["constant_speed"] is None:
        if sign is None or sign > 0:
            max_speed = (vect.max() - 128) / parameters["speed_coef"]
        elif sign < 0:
            max_speed = (vect.min() - 128) / parameters["speed_coef"]
        else:
            max_speed = 0
    else:
        if sign is None or sign > 0:
            max_speed = parameters["constant_speed"]
        elif sign < 0:
            max_speed = -parameters["constant_speed"]
        else:
            max_speed = 0
    flag = False
    if l_prev is not None and (
        (abs(l_new - l_prev) >= parameters["jump_coef_wrap_size"] * (r_new - l_new))
        and abs(l_new - l_prev) >= len_vect * parameters["jump_coef_img_size"]
    ):
        flag = True
    return max_speed, flag, l_new, sign


def find_new_borders(mask_, l_prev, r_prev, crop_size, scale, parameters):
    mask = np.array(mask_, dtype=np.int64)
    deg_mask = mask ** parameters["mask_coef"]
    compress_arr = np.sum(deg_mask, 1)
    if parameters["weighted_sum"] is True:
        prefix_sum = np.insert(
            np.cumsum(
                np.hstack(
                    (
                        compress_arr[: (int(crop_size * scale - 1)) // 2],
                        compress_arr,
                        compress_arr[-(int(crop_size * scale) // 2) :],
                    )
                )
            ),
            0,
            0,
        )
        dif_sum_help = np.array(
            [
                prefix_sum[r] - prefix_sum[r - int(crop_size * scale)]
                for r in range(int(crop_size * scale), len(prefix_sum))
            ]
        )
        prefix_sum = np.insert(np.cumsum(dif_sum_help), 0, 0)
        dif_sum = [
            prefix_sum[r] - prefix_sum[r - int(crop_size * scale)]
            for r in range(int(crop_size * scale), len(prefix_sum))
        ]
    else:
        prefix_sum = np.insert(np.cumsum(compress_arr), 0, 0)
        dif_sum = np.array(
            [
                prefix_sum[r] - prefix_sum[r - int(crop_size * scale)]
                for r in range(int(crop_size * scale), len(prefix_sum))
            ]
        )

    l_new = int(np.argmax(dif_sum) / scale)
    r_new = l_new + crop_size

    general_sign = None

    if l_prev is not None:
        if l_prev == l_new:
            general_sign = 0
        else:
            general_sign = (l_new - l_prev) / abs(l_prev - l_new)
        l_new -= int(general_sign * 1 / scale)
        r_new -= int(general_sign * 1 / scale)

    if r_new > len(mask) / scale:
        r_new = len(mask) / scale
        l_new = r_new - crop_size
    if l_new < 0:
        l_new = 0
        r_new = crop_size

    return int(l_new), int(r_new), prefix_sum, general_sign


def count_speed(mask, vect, l_new, r_new, crop_size, future_speed, parameters):
    global l_prev, r_prev, speed_upgrade, speed_prev
    speed_upgrade = 0
    if parameters["constant_speed"] is None:
        if (l_prev is not None) and l_new - l_prev != 0:
            if (l_new - l_prev) / abs(l_new - l_prev) >= 0:
                non_zero = np.count_nonzero(vect[l_new:r_new] > 128)
                max_speed = max(vect.max() - 128, 0)
            else:
                non_zero = np.count_nonzero(vect[l_new:r_new] < 128)
                max_speed = min(vect.min() - 128, 0)
            if non_zero > len(vect) * parameters["speed_error"]:
                speed_upgrade = max_speed / parameters["speed_coef"]
            else:
                speed_upgrade = 0
    else:
        if (l_prev is None) or l_new - l_prev == 0:
            speed_upgrade = 0
        elif (l_new - l_prev) / abs(l_new - l_prev) > 0:
            speed_upgrade = parameters["constant_speed"]
        else:
            speed_upgrade = -parameters["constant_speed"]
    speed_upgrade = (1 - parameters["future_speed_coef"]) * speed_upgrade + parameters[
        "future_speed_coef"
    ] * future_speed
    speed_upgrade = (
        speed_upgrade * (1 - parameters["prev_speed_coef"])
        + speed_prev * parameters["prev_speed_coef"]
    )
    speed_prev = speed_upgrade
    return speed_upgrade


def move_borders(
    mask,
    prefix_sum,
    scene_change_flag,
    l_new,
    r_new,
    crop_size,
    speed_upgrade,
    general_sign,
    scale,
    parameters,
):
    global l_prev, r_prev, speed_prev, div_mem, jump_delay
    if (
        l_prev is None
        or scene_change_flag == 1
        or (
            abs(l_new - l_prev) >= len(mask) / scale * parameters["jump_coef_img_size"]
            and (
                abs(l_new - l_prev)
                >= parameters["jump_coef_wrap_size"] * (r_new - l_new)
            )
            and (
                (
                    prefix_sum[min(int(r_new * scale), len(prefix_sum) - 1)]
                    - prefix_sum[min(int(l_new * scale), len(prefix_sum) - 1)]
                )
                - (
                    (
                        prefix_sum[min(int(r_prev * scale), len(prefix_sum) - 1)]
                        - prefix_sum[min(int(l_prev * scale), len(prefix_sum) - 1)]
                    )
                )
                >= (
                    parameters["jump_coef_mask_value"] ** parameters["mask_coef"]
                    * crop_size
                    * len(mask[0])
                    / scale
                )
            )
        )
    ):
        l_prev = l_new
        r_prev = r_new
        if scene_change_flag:
            speed_prev = 0
    elif l_prev is not None:
        div = math.ceil(len(mask) / scale * speed_upgrade)
        l_prev += div
        r_prev += div
        if l_prev < 0:
            l_prev = 0
            r_prev = l_prev + crop_size
        if r_prev > len(mask) / scale:
            r_prev = len(mask) / scale
            l_prev = r_prev - crop_size
    return int(l_prev), int(r_prev)


def check_border(
    crop_size, mask, scale, vect, future_speed, scene_change_flag, parameters
):
    global l_prev, r_prev, speed_upgrade, speed_prev
    l_new, r_new, prefix_sum, sign = find_new_borders(
        mask, l_prev, r_prev, crop_size, scale, parameters
    )
    speed_upgrade = count_speed(
        mask, vect, l_new, r_new, crop_size, future_speed, parameters
    )
    return move_borders(
        mask,
        prefix_sum,
        scene_change_flag,
        l_new,
        r_new,
        crop_size,
        speed_upgrade,
        sign,
        scale,
        parameters,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_mask(img, net):
    # sig = torch.nn.Sigmoid()
    mask = net(torch.unsqueeze(transform["test"]["images"](img), 0))[0].detach().numpy()
    mask = sigmoid(mask[0]) * 255
    return mask


def crop(
    img,
    net,
    transform,
    crop_size,
    mask,
    vect,
    future_speed,
    scene_change_flag,
    parameters,
):
    out_mask = np.array([mask for i in range(3)])
    mask_numpy = np.moveaxis(out_mask, 0, 2)
    my_img = np.array(img.transpose(2, 0, 1))
    if crop_size[0] != len(my_img[0]):
        h, b = check_border(
            crop_size[0],
            mask,
            len(mask) / len(my_img[0]),
            vect[0],
            future_speed,
            scene_change_flag,
            parameters,
        )
        l, r = 0, len(my_img[0][0])
    else:
        l, r = check_border(
            crop_size[1],
            mask.transpose(1, 0),
            len(mask[0]) / len(my_img[0][0]),
            vect[0].transpose(1, 0),
            future_speed,
            scene_change_flag,
            parameters,
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
