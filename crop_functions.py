import torch
import numpy as np
from copy import copy
from constants import transform
import torchvision.transforms as transforms
from PIL import Image
import math

from cache import Cache


l_prev = None
r_prev = None
speed_upgrade = 0
speed_prev = 0
cache_vect_min = Cache(1000)
cache_vect_max = Cache(1000)
cache_borders = Cache(1000)


def speed(cur_id, mask, crop_size, vect, parameters):
    global l_prev, r_prev
    len_image = len(vect[0])
    len_image_0 = len(vect[0][0])
    if crop_size[0] != len_image:
        l_new, r_new, prefix_sum, sign = find_new_borders(
            cur_id,
            mask,
            l_prev,
            r_prev,
            crop_size[0],
            len(mask) / len_image,
            parameters,
        )

        len_im = len_image
    else:
        l_new, r_new, prefix_sum, sign = find_new_borders(
            cur_id,
            mask.transpose(1, 0),
            l_prev,
            r_prev,
            crop_size[1],
            len(mask[0]) / len_image_0,
            parameters,
        )

        len_im = len_image_0
    if parameters["constant_speed"] is None:
        if sign is None or sign > 0:
            max_speed = (
                cache_vect_max.get_from_cache(cur_id, max_vect, vect) - 128
            ) / parameters["speed_coef"]
        elif sign < 0:
            max_speed = (
                cache_vect_min.get_from_cache(cur_id, min_vect, vect) - 128
            ) / parameters["speed_coef"]
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
        (abs(l_new - l_prev) >= parameters["jump_coef_wrap_size"] * (r_prev - l_prev))
        and (abs(l_new - l_prev) >= len_im * parameters["jump_coef_img_size"])
    ):
        flag = True

    return max_speed, flag, l_new, sign


def max_vect(cur_id, vect):
    ans = vect.max()
    cache_vect_max.put(cur_id, ans)
    return ans


def min_vect(cur_id, vect):
    ans = vect.min()
    cache_vect_min.put(cur_id, ans)
    return ans


def raw_window_borders(cur_id, mask_, crop_size, scale, parameters):
    """
    Функция, определяющая первоначальное положение окна обрезки.
    Без учета положения окна обрезки для предыдущих кадров.
    """
    mask = np.array(mask_, dtype=np.int64)
    deg_mask = mask ** parameters["mask_coef"]
    # Возводим значения карты значимости в степень,
    # чтобы увеличить разницу между высокой вероятностью и низкой
    compress_arr = np.sum(deg_mask, 1)
    if parameters["weighted_sum"] is True:
        # чтобы вычислить взвешенные суммы (вероятности, распложенные ближе
        # к центру окна обрезки берутся с большим коэффициентом)
        # сумма вероятностей в окне обрезки вычисляется последовательно два раза
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
    cache_borders.put(cur_id, (l_new, r_new, prefix_sum))
    return l_new, r_new, prefix_sum


def find_new_borders(cur_id, mask_, l_prev, r_prev, crop_size, scale, parameters):
    """
    Так как окно обрезки вычисляется на основе карты значимости размером 64x64,
    сдвигаем окно обрезки на погрешность в сторону предыдущего положения окна обрезки
    """
    l_new, r_new, prefix_sum = cache_borders.get_from_cache(
        cur_id, raw_window_borders, mask_, crop_size, scale, parameters
    )
    general_sign = None

    if l_prev is not None:
        if l_prev == l_new:
            general_sign = 0
        else:
            general_sign = (l_new - l_prev) / abs(l_prev - l_new)
        l_new -= int(general_sign * 1 / scale)
        r_new -= int(general_sign * 1 / scale)
    image_size = len(mask_) / scale
    if r_new > image_size:
        r_new = image_size
        l_new = r_new - crop_size
    if l_new < 0:
        l_new = 0
        r_new = crop_size
    return int(l_new), int(r_new), prefix_sum, general_sign


def count_speed(cur_id, mask, vect, l_new, r_new, crop_size, future_speed, parameters):
    """
    Выяисляется скорость окна обрезки
    """
    global l_prev, r_prev, speed_upgrade, speed_prev
    speed_upgrade = 0
    # получаем скорость объекта в кадре
    if parameters["constant_speed"] is None:
        if (l_prev is not None) and l_new - l_prev != 0:
            if (l_new - l_prev) / abs(l_new - l_prev) >= 0:
                non_zero = np.count_nonzero(vect[l_new:r_new] > 128)
                max_speed = max(
                    cache_vect_max.get_from_cache(cur_id, max_vect, vect) - 128, 0
                )
            else:
                non_zero = np.count_nonzero(vect[l_new:r_new] < 128)
                max_speed = min(
                    cache_vect_min.get_from_cache(cur_id, min_vect, vect) - 128, 0
                )
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
    ] * future_speed  # Смешиваем текущую скорость и скорость в некотором кадре в будущем.
    # (Для того, чтобы успевать реагировать на изменения в движении объекта
    speed_upgrade = (
        speed_upgrade * (1 - parameters["prev_speed_coef"])
        + speed_prev * parameters["prev_speed_coef"]
    )  # Применяем метод моментов для стабилизации движения окна обрезки
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
    """
    Определяем политику движения окна обрезки (плавное движение или скачок между сценами)
    """
    global l_prev, r_prev, speed_prev, div_mem, jump_delay
    len_mask = len(mask)
    image_size = len_mask / scale
    prefix_sum_len = len(prefix_sum)
    if (
        l_prev is None
        or scene_change_flag == 1
        or (
            abs(l_new - l_prev) >= image_size * parameters["jump_coef_img_size"]
            and (
                abs(l_new - l_prev)
                >= parameters["jump_coef_wrap_size"] * (r_new - l_new)
            )
            and (
                (
                    prefix_sum[min(int(r_new * scale), prefix_sum_len - 1)]
                    - prefix_sum[min(int(l_new * scale), prefix_sum_len - 1)]
                )
                - (
                    (
                        prefix_sum[min(int(r_prev * scale), prefix_sum_len - 1)]
                        - prefix_sum[min(int(l_prev * scale), prefix_sum_len - 1)]
                    )
                )
                >= (
                    parameters["jump_coef_mask_value"] ** parameters["mask_coef"]
                    * crop_size
                    * image_size
                )
            )
        )
    ):
        l_prev = l_new
        r_prev = r_new
        if scene_change_flag:
            speed_prev = 0
    elif l_prev is not None:
        div = math.ceil(image_size * speed_upgrade)
        l_prev += div
        r_prev += div
        if l_prev < 0:
            l_prev = 0
            r_prev = l_prev + crop_size
        if r_prev > image_size:
            r_prev = image_size
            l_prev = r_prev - crop_size
    return int(l_prev), int(r_prev)


def check_border(
    cur_id, crop_size, mask, scale, vect, future_speed, scene_change_flag, parameters
):
    global l_prev, r_prev, speed_upgrade, speed_prev
    l_new, r_new, prefix_sum, sign = find_new_borders(
        cur_id, mask, l_prev, r_prev, crop_size, scale, parameters
    )
    speed_upgrade = count_speed(
        cur_id, mask, vect, l_new, r_new, crop_size, future_speed, parameters
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
    """
    Получение карты значимости
    """
    mask = net(torch.unsqueeze(transform["test"]["images"](img), 0))[0].detach().numpy()
    mask = sigmoid(mask[0]) * 255
    return mask


def crop(
    cur_id,
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
    """
    Функция обрезает кадр видео и делает все запрошенные визуализации
    """
    my_img = np.array(img.transpose(2, 0, 1))
    image_size = len(my_img[0])
    mask_size = len(mask)
    mask_size_0 = len(mask[0])
    image_size_0 = len(my_img[0][0])
    if crop_size[0] != image_size:
        h, b = check_border(
            cur_id,
            crop_size[0],
            mask,
            mask_size / image_size,
            vect[0],
            future_speed,
            scene_change_flag,
            parameters,
        )
        l, r = 0, image_size_0
        ans = copy(my_img[:, h:b, l:r])
        if (
            parameters["out_filename_wrap"] is not None
            or parameters["out_filename_both"] is not None
            or parameters["out_filename_mask"] is not None
        ):
            my_img[0, max(h - 6, 0) : h, :] = 255
            my_img[0, b : min(image_size, b + 6), :] = 255
            my_img[1, max(h - 6, 0) : h, :] = 0
            my_img[1, b : min(image_size, b + 6), :] = 0
            my_img[2, max(h - 6, 0) : h, :] = 0
            my_img[2, b : min(image_size, b + 6), :] = 0

            my_img[2, h:b, l : min(l + 6, r)] = 255
            my_img[2, h:b, max(l, r - 6) : r] = 255
    else:
        l, r = check_border(
            cur_id,
            crop_size[1],
            mask.transpose(1, 0),
            mask_size_0 / image_size_0,
            vect[0].transpose(1, 0),
            future_speed,
            scene_change_flag,
            parameters,
        )
        h, b = 0, image_size
        ans = copy(my_img[:, h:b, l:r])
        if (
            parameters["out_filename_wrap"] is not None
            or parameters["out_filename_both"] is not None
            or parameters["out_filename_mask"] is not None
        ):
            my_img[0, :, max(l - 6, 0) : l] = 255
            my_img[0, :, r : min(image_size_0, r + 6)] = 255
            my_img[1, :, max(l - 6, 0) : l] = 0
            my_img[1, :, r : min(image_size_0, r + 6)] = 0
            my_img[2, :, max(l - 6, 0) : l] = 0
            my_img[2, :, r : min(image_size_0, r + 6)] = 0

            my_img[2, h : max(h + 6, b), l:r] = 255
            my_img[2, min(h, b - 6) : b, l:r] = 255
    ans = np.moveaxis(ans, 0, 2)
    my_img = np.moveaxis(my_img, 0, 2)
    mask_numpy = None
    croped_mask = None
    if (
        parameters["out_filename_mask"] is not None
        or parameters["out_compare_mask_filename"] is not None
        or parameters["out_filename_both_mask"] is not None
    ):
        mask = np.array(mask, dtype=np.int64)
        deg_mask = mask ** parameters["mask_coef"]
        out_mask = np.array([deg_mask for i in range(3)], dtype=np.int64)
        mask_numpy = np.moveaxis(out_mask, 0, 2)
        mask_numpy = mask_numpy / mask_numpy.max() * 255

        mask_numpy = np.asarray(
            transforms.Resize((len(my_img), len(my_img[0])))(
                Image.fromarray(np.uint8(mask_numpy))
            ),
            dtype=np.int64,
        )

        croped_mask = copy(mask_numpy[h:b, l:r, :])

        mask_numpy[h:b, l:r, 2] = 255
        for i, val in enumerate([255, 0, 0]):
            mask_numpy[max(h - 6, 0) : min(len(mask_numpy), h + 6), l:r, i] = val
            mask_numpy[max(b - 6, 0) : min(len(mask_numpy), b + 6), l:r, i] = val
            mask_numpy[h:b, max(l - 6, 0) : min(len(mask_numpy[0]), l + 6), i] = val
            mask_numpy[h:b, max(r - 6, 0) : min(len(mask_numpy[0]), r + 6), i] = val

    return (my_img, ans), (mask_numpy, croped_mask)
