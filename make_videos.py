import numpy as np
from datetime import datetime

from copy import deepcopy
from preprocess import get_nn

from constants import transform
from crop_functions import crop, make_mask, speed
import crop_functions

from vapoursynth import core
from video_processing import make_processes, set_processes, upload_videos

core.std.LoadPlugin(
    path="/home/linuxbrew/.linuxbrew/Cellar/ffms2/2.40_2/lib/libffms2.so"
)

core.std.LoadPlugin(
    path="/home/linuxbrew/.linuxbrew/Cellar/mvtools/23_1/lib/libmvtools.so"
)


def frame_to_numpy(frame):
    image = np.stack([np.array(cur_col) for cur_col in frame]).astype(np.uint8)
    image = np.moveaxis(
        image,
        0,
        -1,
    )
    return image


def set_jump_flag(
    future_array,
    future_array_len,
    it,
    parameters,
    new_height,
    new_width,
    crop_size,
    image_len,
):
    jump_flag = True
    scene_flag = True
    l_prev = None
    for i in range(parameters["jump_delay_coef"]):
        _, jump_flag_cur, l_new = speed(
            future_array[(it + i) % future_array_len]["image_mask"],
            (new_height, new_width),
            future_array[(it + i) % future_array_len]["vect"],
            parameters,
        )
        # if int((it - future_array_len) / (0.5 * future_array_len)) == 21:
        #     print(i, l_prev, l_new, jump_flag)
        if jump_flag_cur is False:
            jump_flag = False
        if l_prev is not None and (
            (abs(l_new - l_prev) >= parameters["jump_coef_wrap_size"] * crop_size)
            and abs(l_new - l_prev) >= image_len * parameters["jump_coef_img_size"]
        ):
            jump_flag = False
        if future_array[(it + i) % future_array_len]["scene_flag"]:
            scene_flag = False
        l_prev = l_new
    return jump_flag, scene_flag


def make_videos(
    in_filename,
    out_filename,
    ratio=None,
    constant_speed=None,
    speed_error=0.001,
    mask_coef=5,
    fps_coef=2,
    speed_coef=8000,
    prev_speed_coef=0.9,
    future_speed_coef=0.3,
    jump_coef_img_size=0.2,
    jump_coef_wrap_size=1 / 2,
    jump_coef_mask_value=5,
    jump_delay_coef=1,
    scene_detection_flag=True,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    in_compare_filename=None,
    out_compare_filename=None,
    out_compare_mask_filename=None,
):
    crop_functions.l_prev = None
    crop_functions.r_prev = None
    crop_functions.speed_prev = 0
    crop_functions.speed_upgrade = 0

    (
        yuv_video,
        input_video,
        input_compare_video,
        video_vector,
        scene_detection_video,
    ) = upload_videos(in_filename, in_compare_filename)

    fps = float(yuv_video.fps)

    parameters = dict(
        {
            "constant_speed": constant_speed,
            "speed_error": speed_error,
            "mask_coef": mask_coef,
            "fps_coef": fps_coef,
            "speed_coef": speed_coef,
            "prev_speed_coef": prev_speed_coef,
            "future_speed_coef": future_speed_coef,
            "jump_coef_img_size": jump_coef_img_size,
            "jump_coef_wrap_size": jump_coef_wrap_size,
            "jump_coef_mask_value": jump_coef_mask_value,
            "jump_delay_coef": int(jump_delay_coef * fps),
            "scene_detection_flag": scene_detection_flag,
        },
    )
    height = yuv_video.height
    width = yuv_video.width
    if ratio is None:
        if height > width:
            ratio = (8, 12)
        else:
            ratio = (12, 8)

    new_height = height
    new_width = width
    k = ratio[0] * width / ratio[1] / height
    net = get_nn()
    add_height = 0
    add_width = 0
    coef_v = 1
    coef_h = 1

    if k < 1:
        new_height = int(height * k)
        add_height = height
        crop_size = new_height
        func = np.vstack
        coef_h = 2
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=4)
        image_len = new_height
    else:
        new_width = int(width / k)
        add_width = width
        func = np.hstack
        crop_size = new_width
        coef_v = 2
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=3)
        image_len = new_width

    if new_height % 2 == 1:
        new_height -= 1
    if new_width % 2 == 1:
        new_width -= 1

    processes = make_processes(
        width,
        height,
        fps,
        new_width,
        new_height,
        add_width,
        add_height,
        coef_v,
        coef_h,
        out_filename,
        out_filename_wrap,
        out_filename_both,
        out_filename_mask,
        out_compare_filename,
        out_compare_mask_filename,
    )
    make_mask_time = 0
    crop_function_time = 0
    processes_time = 0
    whole_time = datetime.now()
    future_array_len = int(parameters["fps_coef"] * fps)
    future_array = [None] * (future_array_len)
    it = 0
    changed_flag = 0
    for cur_moving_vector, cur_image, cur_compare_image, cur_scene in zip(
        speed_video.frames(),
        input_video.frames(),
        input_compare_video.frames(),
        scene_detection_video.frames(),
    ):
        # print("second: ", (it - future_array_len) / fps)
        image = frame_to_numpy(cur_image)
        make_mask_start_time = datetime.now()
        image_mask = make_mask(image, net)
        make_mask_time += (datetime.now() - make_mask_start_time).microseconds
        np_moving_vector = np.array(
            [np.array(cur_col) for cur_col in cur_moving_vector]
        )

        if it < future_array_len:
            future_array[it] = {
                "image_mask": image_mask,
                "image": image,
                "vect": np_moving_vector,
                "compare": cur_compare_image,
                "scene_flag": cur_scene.props["_SceneChangePrev"]
                and parameters["scene_detection_flag"],
            }
            it += 1
            continue

        if cur_scene.props["_SceneChangePrev"] and parameters["scene_detection_flag"]:
            changed_flag = it + future_array_len
        updated_parameters = deepcopy(parameters)
        if it < changed_flag:
            updated_parameters["future_speed_coef"] = 0
            future_speed = 0
        else:
            future_speed, _, _ = speed(
                image_mask,
                (new_height, new_width),
                np_moving_vector,
                parameters,
            )
        jump_flag, scene_flag = set_jump_flag(
            future_array,
            future_array_len,
            it,
            parameters,
            new_height,
            new_width,
            crop_size,
            image_len,
        )
        if jump_flag is False:
            updated_parameters["jump_coef_wrap_size"] = max(len(image), len(image[0]))

        start_crop_time = datetime.now()
        img, mask = crop(
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            future_array[it % future_array_len]["image_mask"],
            future_array[it % future_array_len]["vect"],
            future_speed,
            future_array[it % future_array_len]["scene_flag"] and scene_flag,
            updated_parameters,
        )
        crop_function_time += (datetime.now() - start_crop_time).seconds * 10**6 + (
            datetime.now() - start_crop_time
        ).microseconds

        start_processes_time = datetime.now()
        set_processes(
            img,
            mask,
            future_array[it % future_array_len]["compare"],
            func,
            processes,
            out_filename,
            out_filename_wrap,
            out_filename_both,
            out_filename_mask,
            out_compare_filename,
            out_compare_mask_filename,
        )

        future_array[it % future_array_len] = {
            "image_mask": image_mask,
            "image": image,
            "vect": np_moving_vector,
            "compare": cur_compare_image,
            "scene_flag": cur_scene.props["_SceneChangePrev"]
            and parameters["scene_detection_flag"],
        }
        processes_time += (datetime.now() - start_processes_time).microseconds

        it += 1

    parameters["jump_coef_wrap_size"] = max(len(image), len(image[0]))

    for i in range(future_array_len):
        scene_flag = True
        if i < updated_parameters["jump_delay_coef"]:
            jump_flag, scene_flag = set_jump_flag(
                future_array,
                future_array_len,
                it,
                parameters,
                new_height,
                new_width,
                crop_size,
                image_len,
            )
            if jump_flag is False:
                updated_parameters["jump_coef_wrap_size"] = max(
                    len(image), len(image[0])
                )
        start_crop_time = datetime.now()
        future_speed, _, _ = speed(
            future_array[it % future_array_len]["image_mask"],
            (new_height, new_width),
            future_array[it % future_array_len]["vect"],
            parameters,
        )
        img, mask = crop(
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            future_array[it % future_array_len]["image_mask"],
            future_array[it % future_array_len]["vect"],
            future_speed,
            future_array[it % future_array_len]["scene_flag"] and scene_flag,
            parameters,
        )
        crop_function_time += (datetime.now() - start_crop_time).seconds * 10**6 + (
            datetime.now() - start_crop_time
        ).microseconds

        start_processes_time = datetime.now()
        set_processes(
            img,
            mask,
            future_array[it % future_array_len]["compare"],
            func,
            processes,
            out_filename,
            out_filename_wrap,
            out_filename_both,
            out_filename_mask,
            out_compare_filename,
            out_compare_mask_filename,
        )
        it += 1
        processes_time += (datetime.now() - start_processes_time).microseconds

    print(
        "Iteration :",
        it,
    )
    print("All work :", (datetime.now() - whole_time).total_seconds())
    print(f"make_mask : {make_mask_time / 10**6}")
    print(f"crop_function : {crop_function_time / 10**6}")
    print(f"processes : {processes_time / 10**6}")
    for process in processes.values():
        process.stdin.close()

    for process in processes.values():
        process.wait()
