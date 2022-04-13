import numpy as np
from datetime import datetime


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


def make_videos(
    in_filename,
    out_filename,
    parameters={
        "mask_coef": 5,
        "constant_speed": None,
        "speed_error": 0.01,
        "speed_coef": 7000,
        "future_speed_coef": 0.1,
        "prev_speed_coef": 0.8,
        "jump_coef_wrap_size": 2 / 3,
        "jump_coef_mask_value": 5,
        "fps_coef": 2,
        "scene_detection_flag": True,
    },
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

    # vert
    # crop_size = (12, 8)
    # hor
    # crop_size = (8, 12)

    height = yuv_video.height
    width = yuv_video.width
    if height > width:
        crop_size = (8, 12)
    else:
        crop_size = (12, 8)

    new_height = height
    new_width = width
    k = crop_size[0] * width / crop_size[1] / height
    net = get_nn()
    add_height = 0
    add_width = 0
    coef_v = 1
    coef_h = 1

    if k < 1:
        new_height = int(height * k)
        add_height = height
        func = np.vstack
        coef_h = 2
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=4)
    else:
        new_width = int(width / k)
        add_width = width
        func = np.hstack
        coef_v = 2
        speed_video = core.mv.Mask(yuv_video, video_vector, kind=3)

    fps = int(yuv_video.fps)
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
    for cur_moving_vector, cur_image, cur_compare_image, cur_scene in zip(
        speed_video.frames(),
        input_video.frames(),
        input_compare_video.frames(),
        scene_detection_video.frames(),
    ):
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

        start_crop_time = datetime.now()
        img, mask = crop(
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            future_array[it % future_array_len]["image_mask"],
            future_array[it % future_array_len]["vect"],
            speed(np_moving_vector),
            future_array[it % future_array_len]["scene_flag"],
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

    for i in range(future_array_len):
        start_crop_time = datetime.now()
        img, mask = crop(
            future_array[it % future_array_len]["image"],
            net,
            transform,
            (new_height, new_width),
            future_array[it % future_array_len]["image_mask"],
            future_array[it % future_array_len]["vect"],
            0,
            future_array[it % future_array_len]["scene_flag"],
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
