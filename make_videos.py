import numpy as np
from datetime import datetime

from copy import deepcopy
from preprocess import get_nn
import constants

from constants import transform
from crop_functions import crop, make_mask, speed
import crop_functions
from cache import Cache
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


global_id = 0


def set_jump_flag(
    cur_id,
    future_array,
    future_array_len,
    it,
    parameters,
    new_height,
    new_width,
    crop_size,
    image_len,
    max_iteration,
):
    jump_flag = True
    scene_flag = True
    move_flag = True
    move_flag_cur = True
    future_flag = True
    l_prev = None
    sign = None
    image_len_jump_coef_wrap_size = crop_size * parameters["jump_coef_wrap_size"]
    image_len_jump_coef_img_size = image_len * parameters["jump_coef_img_size"]
    for i in range(
        min(
            max(parameters["jump_delay_coef"], parameters["moving_available_coef"]),
            max_iteration,
        )
    ):
        _, jump_flag_cur, l_new, sign = speed(
            cur_id + i,
            future_array[(it + i) % future_array_len]["image_mask"],
            (new_height, new_width),
            future_array[(it + i) % future_array_len]["vect"],
            parameters,
        )
        if i == 0:
            # mem_sign = sign
            move_flag_cur = sign
        # elif sign != mem_sign and i < parameters["moving_available_coef"]:
        elif sign != move_flag_cur and i < parameters["moving_available_coef"]:
            move_flag_cur = False
        elif (
            # sign == mem_sign
            sign == move_flag_cur
            and i < parameters["moving_available_coef"]
            and move_flag_cur is False
        ):
            move_flag = False
        if i < parameters["jump_delay_coef"]:
            if jump_flag_cur is False:
                jump_flag = False
            if l_prev is not None and (
                (abs(l_new - l_prev) >= image_len_jump_coef_wrap_size)
                and abs(l_new - l_prev) >= image_len_jump_coef_img_size
            ):
                jump_flag = False
            if i != 0 and future_array[(it + i) % future_array_len]["scene_flag"]:
                scene_flag = False

        if l_prev is not None and (
            (abs(l_new - l_prev) >= image_len_jump_coef_wrap_size)
            and abs(l_new - l_prev) >= image_len_jump_coef_img_size
        ):
            future_flag = False
        l_prev = l_new
    return jump_flag, scene_flag, move_flag, future_flag


def make_videos(
    in_filename,
    out_filename,
    ratio=None,
    # ratio,
    constant_speed=None,
    speed_error=constants.speed_error,
    mask_coef=constants.mask_coef,
    fps_coef=constants.fps_coef,
    speed_coef=constants.speed_coef,
    prev_speed_coef=constants.prev_speed_coef,
    future_speed_coef=constants.future_speed_coef,
    jump_coef_img_size=constants.jump_coef_img_size,
    jump_coef_wrap_size=constants.jump_coef_wrap_size,
    jump_coef_mask_value=constants.jump_coef_mask_value,
    jump_delay_coef=constants.jump_delay_coef,
    scene_detection_parameters=constants.scene_detection_parameters,
    moving_available_coef=constants.moving_available_coef,
    scene_detection_flag=True,
    weighted_sum=True,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_both_mask=None,
    out_filename_mask=None,
    in_compare_filename=None,
    out_compare_filename=None,
    out_compare_mask_filename=None,
):
    global global_id
    global_id = 0
    crop_functions.l_prev = None
    crop_functions.r_prev = None
    crop_functions.speed_prev = 0
    crop_functions.speed_upgrade = 0
    crop_functions.cache_borders = Cache(1000)
    crop_functions.cache_vect_max = Cache(1000)
    crop_functions.cache_vect_min = Cache(1000)

    make_mask_time = 0
    # crop_function_time = 0
    # processes_time = 0
    # future_time = 0
    # to_numpy_time = 0
    # whole_time = datetime.now()
    # start_processes_time = datetime.now()
    (
        yuv_video,
        input_video,
        input_compare_video,
        video_vector,
        scene_detection_video,
    ) = upload_videos(in_filename, in_compare_filename, scene_detection_parameters)
    # processes_time += (datetime.now() - start_processes_time).seconds * 10**6 + (
    #     datetime.now() - start_processes_time
    # ).microseconds

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
            "moving_available_coef": int(moving_available_coef * fps),
            "scene_detection_flag": scene_detection_flag,
            "weighted_sum": weighted_sum,
            "out_filename_wrap": out_filename_wrap,
            "out_filename_both": out_filename_both,
            "out_filename_both_mask": out_filename_both_mask,
            "out_filename_mask": out_filename_mask,
            "in_compare_filename": in_compare_filename,
            "out_compare_filename": out_compare_filename,
            "out_compare_mask_filename": out_compare_mask_filename,
        },
    )
    height = yuv_video.height
    width = yuv_video.width
    if ratio is None:
        if height > width:
            # ratio = (8, 12)
            ratio = (2, 3)
        else:
            # ratio = (12, 8)
            ratio = (3, 2)

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
        # start_processes_time = datetime.now()
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=4)
        # processes_time += (datetime.now() - start_processes_time).seconds * 10**6 + (
        #     datetime.now() - start_processes_time
        # ).microseconds
        image_len = height
    else:
        new_width = int(width / k)
        add_width = width
        func = np.hstack
        crop_size = new_width
        coef_v = 2
        # start_processes_time = datetime.now()
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=3)
        # processes_time += (datetime.now() - start_processes_time).seconds * 10**6 + (
        #     datetime.now() - start_processes_time
        # ).microseconds
        image_len = width

    if new_height % 2 == 1:
        new_height -= 1
    if new_width % 2 == 1:
        new_width -= 1
    # start_processes_time = datetime.now()
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
        out_filename_both_mask,
        out_filename_mask,
        out_compare_filename,
        out_compare_mask_filename,
    )
    # processes_time += (datetime.now() - start_processes_time).seconds * 10**6 + (
    #     datetime.now() - start_processes_time
    # ).microseconds
    future_array_len = max(
        max(int(parameters["fps_coef"] * fps), parameters["jump_delay_coef"]), 1
    )
    future_array = [None] * future_array_len
    it = 0
    changed_flag = 0
    after_scene_changed = 0
    add = 0
    for cur_moving_vector, cur_image, cur_compare_image, cur_scene in zip(
        speed_video.frames(),
        input_video.frames(),
        input_compare_video.frames(),
        scene_detection_video.frames(),
    ):
        # start_to_numpy_time = datetime.now()
        image = frame_to_numpy(cur_image)
        # to_numpy_time += (datetime.now() - start_to_numpy_time).seconds * 10**6 + (
        #     datetime.now() - start_to_numpy_time
        # ).microseconds
        make_mask_start_time = datetime.now()
        image_mask = make_mask(image, net)
        make_mask_time += (datetime.now() - make_mask_start_time).seconds * 10**6 + (
            datetime.now() - make_mask_start_time
        ).microseconds
        # start_to_numpy_time = datetime.now()
        np_moving_vector = np.array(
            [np.array(cur_col) for cur_col in cur_moving_vector]
        )
        # to_numpy_time += (datetime.now() - start_to_numpy_time).seconds * 10**6 + (
        #     datetime.now() - start_to_numpy_time
        # ).microseconds

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
        # start_future_time = datetime.now()
        jump_flag, scene_flag, move_flag, future_flag = set_jump_flag(
            it - future_array_len + global_id,
            future_array,
            future_array_len,
            it,
            parameters,
            new_height,
            new_width,
            crop_size,
            image_len,
            future_array_len,
        )
        # future_time += (datetime.now() - start_future_time).seconds * 10**6 + (
        #     datetime.now() - start_future_time
        # ).microseconds
        updated_parameters = deepcopy(parameters)
        if jump_flag is False:
            updated_parameters["jump_coef_wrap_size"] = max(len(image), len(image[0]))
            updated_parameters["scene_detection_flag"] = False

        if future_flag is False:
            updated_parameters["future_speed_coef"] = 0

        if move_flag is False:
            updated_parameters["constant_speed"] = 0
            updated_parameters["future_speed_coef"] = 0
            updated_parameters["prev_speed_coef"] = 0

        mask = future_array[it % future_array_len]["image_mask"]
        vect = future_array[it % future_array_len]["vect"]
        if cur_scene.props["_SceneChangePrev"] and parameters["scene_detection_flag"]:
            changed_flag = it + future_array_len
        if it < changed_flag:
            updated_parameters["future_speed_coef"] = 0
            future_speed = 0
        else:
            # start_future_time = datetime.now()
            future_speed, _, _, _ = speed(
                it + global_id,
                image_mask,
                (new_height, new_width),
                np_moving_vector,
                parameters,
            )
            # future_time += (datetime.now() - start_future_time).seconds * 10**6 + (
            #     datetime.now() - start_future_time
            # ).microseconds
        if (
            future_array[it % future_array_len]["scene_flag"]
            and scene_flag
            and parameters["scene_detection_flag"]
            and it > changed_flag - (future_array_len - parameters["jump_delay_coef"])
        ) or it < after_scene_changed:
            if it > after_scene_changed:
                after_scene_changed = it + max(parameters["jump_delay_coef"] - 1, 0)
            mask = future_array[after_scene_changed % future_array_len]["image_mask"]
            vect = future_array[after_scene_changed % future_array_len]["vect"]
            updated_parameters["future_speed_coef"] = 0
            add = (
                after_scene_changed
                - future_array_len
                - (it - future_array_len + global_id)
            )

        # start_crop_time = datetime.now()
        img, mask = crop(
            it - future_array_len + global_id + add,
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            mask,
            vect,
            future_speed,
            future_array[it % future_array_len]["scene_flag"] and scene_flag,
            updated_parameters,
        )
        add = 0
        # crop_function_time += (datetime.now() - start_crop_time).seconds * 10**6 + (
        #     datetime.now() - start_crop_time
        # ).microseconds

        # start_processes_time = datetime.now()
        set_processes(
            img,
            mask,
            future_array[it % future_array_len]["compare"],
            func,
            processes,
            out_filename,
            out_filename_wrap,
            out_filename_both,
            out_filename_both_mask,
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
        # processes_time += (datetime.now() - start_processes_time).microseconds

        it += 1

    for i in range(future_array_len):
        scene_flag = True
        # start_future_time = datetime.now()
        jump_flag, scene_flag, move_flag, future_flag = set_jump_flag(
            it - future_array_len + global_id,
            future_array,
            future_array_len,
            it,
            parameters,
            new_height,
            new_width,
            crop_size,
            image_len,
            future_array_len - i,
        )
        # future_time += (datetime.now() - start_future_time).seconds * 10**6 + (
        #     datetime.now() - start_future_time
        # ).microseconds
        updated_parameters = deepcopy(parameters)
        if jump_flag is False:
            updated_parameters["jump_coef_wrap_size"] = max(len(image), len(image[0]))
            updated_parameters["scene_detection_flag"] = False
        if future_flag is False:
            updated_parameters["future_speed_coef"] = 0
        if move_flag is False:
            updated_parameters["constant_speed"] = 0
            updated_parameters["future_speed_coef"] = 0
            updated_parameters["prev_speed_coef"] = 0

        # start_crop_time = datetime.now()
        img, mask = crop(
            it - future_array_len + global_id,
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            future_array[it % future_array_len]["image_mask"],
            future_array[it % future_array_len]["vect"],
            0,
            future_array[it % future_array_len]["scene_flag"] and scene_flag,
            updated_parameters,
        )
        # crop_function_time += (datetime.now() - start_crop_time).seconds * 10**6 + (
        #     datetime.now() - start_crop_time
        # ).microseconds

        # start_processes_time = datetime.now()
        set_processes(
            img,
            mask,
            future_array[it % future_array_len]["compare"],
            func,
            processes,
            out_filename,
            out_filename_wrap,
            out_filename_both,
            out_filename_both_mask,
            out_filename_mask,
            out_compare_filename,
            out_compare_mask_filename,
        )
        it += 1
        # processes_time += (datetime.now() - start_processes_time).microseconds
    global_id += it - future_array_len
    # print(
    #     "Iteration :",
    #     it - future_array_len,
    # )
    # print("All work :", (datetime.now() - whole_time).total_seconds())
    print(f"make_mask : {make_mask_time / 10**6}")
    print(f"make_mask_ones : {make_mask_time / 10**6 / (it-future_array_len)}")
    # print(f"crop_function : {crop_function_time / 10**6}")
    # print(f"processes : {processes_time / 10**6}")
    # print(f"future : {future_time / 10**6}")
    # print(f"numpy_time : {to_numpy_time / 10**6}")
    for process in processes.values():
        process.stdin.close()

    for process in processes.values():
        process.wait()
