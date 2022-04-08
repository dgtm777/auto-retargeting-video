import numpy as np
from datetime import datetime
import os
import sys
import click


import vapoursynth as vs

from preprocess import get_nn

from constants import (
    all_videos,
    mask_size,
    transform,
    root_dir,
)
from crop_functions import crop, make_mask
import crop_functions

import ffmpeg
from vapoursynth import core

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

core.std.LoadPlugin(
    path="/home/linuxbrew/.linuxbrew/Cellar/ffms2/2.40_2/lib/libffms2.so"
)

core.std.LoadPlugin(
    path="/home/linuxbrew/.linuxbrew/Cellar/mvtools/23_1/lib/libmvtools.so"
)


def avs_subtitle(video, text: str, position: int = 2):
    # type: (vs.VideoNode, str, int) -> vs.VideoNode
    """Adds quick-and-easy subtitling wrapper."""

    # Use FullHD as a reference resolution
    scale1 = 100 * video.height // 1080
    scale2 = 100 * video.width // 1920

    scale = str(max(scale1, scale2))

    style = (
        r"""{\fn(Asul),"""
        + r"""\bord(2.4),"""
        + r"""\b900,"""
        + r"""\fsp(1.0),"""
        + r"""\fs82,"""
        + r"""\fscx"""
        + scale
        + r""","""
        + r"""\fscy"""
        + scale
        + r""","""
        + r"""\1c&H00FFFF,"""
        + r"""\3c&H000000,"""
        + r"""\an"""
        + str(position)
        + r"""}"""
    )

    return core.sub.Subtitle(clip=video, text=style + text)


def check_premiere_shape(premiere_image, img, func):
    if premiere_image.shape[0] < img[1].shape[0]:
        premiere_image = func(
            (
                premiere_image,
                np.zeros(
                    shape=(
                        img[1].shape[0] - premiere_image.shape[0],
                        img[1].shape[1],
                        img[1].shape[2],
                    )
                ),
            )
        )

    if premiere_image.shape[1] < img[1].shape[1]:
        premiere_image = func(
            (
                premiere_image,
                np.zeros(
                    shape=(
                        img[1].shape[0],
                        img[1].shape[1] - premiere_image.shape[1],
                        img[1].shape[2],
                    )
                ),
            )
        )
    return premiere_image


def frame_to_numpy(frame):
    image = np.stack([np.array(cur_col) for cur_col in frame]).astype(np.uint8)
    image = np.moveaxis(
        image,
        0,
        -1,
    )
    return image


def make_processes(
    width,
    height,
    fps,
    new_width,
    new_height,
    add_width,
    add_height,
    coef_v,
    coef_h,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    out_premiere_filename=None,
    out_premiere_mask_filename=None,
):
    processes = dict()
    if out_filename_wrap is not None:
        processes[out_filename_wrap] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
            )
            .filter("fps", fps=fps, round="up")
            .output(out_filename_wrap)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_filename_both is not None:
        processes[out_filename_both] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(new_width + add_width, new_height + add_height),
            )
            .filter("fps", fps=fps, round="up")
            .output(out_filename_both)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_filename_mask is not None:
        processes[out_filename_mask] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(
                    coef_v * new_width + 2 * add_width,
                    coef_h * new_height + 2 * add_height,
                ),
            )
            .filter("fps", fps=fps, round="up")
            .output(out_filename_mask)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_premiere_filename is not None:
        processes[out_premiere_filename] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(coef_v * new_width, coef_h * new_height),
            )
            .filter("fps", fps=fps, round="up")
            .output(out_premiere_filename)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_premiere_mask_filename is not None:
        processes[out_premiere_mask_filename] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(
                    coef_v * new_width + 2 * add_width,
                    coef_h * new_height + 2 * add_height,
                ),
            )
            .filter("fps", fps=fps, round="up")
            .output(out_premiere_mask_filename)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    return processes


def in_for(
    cur_moving_vector,
    image,
    image_mask,
    cur_premiere_image,
    net,
    new_height,
    new_width,
    func,
    process2,
    processes,
    out_filename,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    out_premiere_filename=None,
    out_premiere_mask_filename=None,
    make_numpy_time=0,
    make_mask_time=0,
    crop_function_time=0,
    processes_time=0,
):

    np_moving_vector = np.array([np.array(cur_col) for cur_col in cur_moving_vector])

    start_crop_time = datetime.now()
    img, mask = crop(
        image,
        net,
        transform,
        (new_height, new_width),
        image_mask,
        np_moving_vector,
    )
    crop_function_time += (datetime.now() - start_crop_time).seconds * 10**6 + (
        datetime.now() - start_crop_time
    ).microseconds

    start_processes_time = datetime.now()
    process2.stdin.write(img[1].astype(np.uint8).tobytes())

    if out_filename_wrap in processes:
        processes[out_filename_wrap].stdin.write(img[0].astype(np.uint8).tobytes())
    if out_filename_both in processes:
        processes[out_filename_both].stdin.write(
            func((img[0], img[1])).astype(np.uint8).tobytes()
        )
    if out_filename_mask in processes:
        processes[out_filename_mask].stdin.write(
            func((mask[0], img[0], img[1], mask[1])).astype(np.uint8).tobytes()
        )
    if out_premiere_filename in processes:
        premiere_image = np.stack(
            [np.array(cur_col) for cur_col in cur_premiere_image]
        ).astype(np.uint8)
        premiere_image = np.moveaxis(
            premiere_image,
            0,
            -1,
        )
        premiere_image = check_premiere_shape(premiere_image, img, func)

        processes[out_premiere_filename].stdin.write(
            func((img[1], premiere_image)).astype(np.uint8).tobytes()
        )
    if out_premiere_mask_filename in processes:
        processes[out_premiere_mask_filename].stdin.write(
            func((img[0], mask[0], img[1], premiere_image)).astype(np.uint8).tobytes()
        )
    processes_time += (datetime.now() - start_processes_time).microseconds
    return make_numpy_time, make_mask_time, crop_function_time, processes_time


def make_videos(
    in_filename,
    out_filename,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    in_premiere_filename=None,
    out_premiere_filename=None,
    out_premiere_mask_filename=None,
):
    crop_functions.l_prev = None
    crop_functions.r_prev = None
    crop_functions.speed_upgrade = 0

    input_video = core.ffms2.Source(source=in_filename)

    if in_premiere_filename is not None:
        input_premiere_video = core.ffms2.Source(source=in_premiere_filename)

    video = core.resize.Bicubic(
        clip=input_video,
        width=input_video.width,
        height=input_video.height,
        format=vs.YUV444P8,
        # format=vs.RGB24,
    )
    input_video = core.resize.Bicubic(
        clip=input_video,
        width=input_video.width,
        height=input_video.height,
        # format=vs.YUV444P8,
        format=vs.RGB24,
        matrix_in_s="709",
    )

    if in_premiere_filename is not None:
        input_premiere_video = core.resize.Bicubic(
            clip=input_premiere_video,
            width=input_premiere_video.width,
            height=input_premiere_video.height,
            # format=vs.YUV444P8,
            format=vs.RGB24,
            matrix_in_s="709",
        )
        input_premiere_video = avs_subtitle(input_premiere_video, "premiere")
    else:
        input_premiere_video = input_video

    super_video = core.mv.Super(video)
    video_vector = core.mv.Analyse(super_video)
    # vert
    # crop_size = (12, 8)
    # hor
    # crop_size = (8, 12)

    height = video.height
    width = video.width
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
        speed_video = core.mv.Mask(video, video_vector, kind=4)
    else:
        new_width = int(width / k)
        add_width = width
        func = np.hstack
        coef_v = 2
        speed_video = core.mv.Mask(video, video_vector, kind=3)

    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(new_width, new_height),
        )
        .filter("fps", fps=video.fps, round="up")
        .output(out_filename)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    processes = make_processes(
        width,
        height,
        video.fps,
        new_width,
        new_height,
        add_width,
        add_height,
        coef_v,
        coef_h,
        out_filename_wrap,
        out_filename_both,
        out_filename_mask,
        out_premiere_filename,
        out_premiere_mask_filename,
    )

    make_mask_time = 0
    crop_function_time = 0
    make_numpy_time = 0
    processes_time = 0
    premiere_time = 0
    whole_time = datetime.now()
    it = 0
    worker = 8
    id_array = 0
    images_array = [None] * worker
    executor = ProcessPoolExecutor(max_workers=worker)
    # make_masks(executor, input_video.frames(), net, futures)
    for cur_moving_vector, cur_image, cur_premiere_image in zip(
        speed_video.frames(), input_video.frames(), input_premiere_video.frames()
    ):
        if it < worker:
            image = frame_to_numpy(cur_image)
            images_array[id_array] = {
                "moving_vector": cur_moving_vector,
                "image": image,
                "image_mask": executor.submit(make_mask, image, net),
                "premier": cur_premiere_image,
            }
        else:
            (
                cur_make_numpy_time,
                cur_make_mask_time,
                cur_crop_function_time,
                cur_processes_time,
            ) = in_for(
                images_array[id_array]["moving_vector"],
                images_array[id_array]["image"],
                images_array[id_array]["image_mask"].result(),
                images_array[id_array]["premier"],
                net,
                new_height,
                new_width,
                func,
                process2,
                processes,
                out_filename,
                out_filename_wrap,
                out_filename_both,
                out_filename_mask,
                out_premiere_filename,
                out_premiere_mask_filename,
                make_numpy_time,
                make_mask_time,
                crop_function_time,
                processes_time,
            )

            make_numpy_time += cur_make_numpy_time
            make_mask_time += cur_make_mask_time
            crop_function_time += cur_crop_function_time
            processes_time += cur_processes_time

            image = frame_to_numpy(cur_image)

            images_array[id_array] = {
                "moving_vector": cur_moving_vector,
                "image": image,
                "image_mask": executor.submit(make_mask, image, net),
                "premier": cur_premiere_image,
            }
        it += 1
        id_array = (id_array + 1) % 8

    for i in range(worker):
        cur_make_numpy_time, cur_make_mask_time, cur_crop_function_time = in_for(
            images_array[id_array]["moving_vector"],
            images_array[id_array]["image"],
            images_array[id_array]["image_mask"].result(),
            images_array["premier"][id_array],
            net,
            new_height,
            new_width,
            func,
            process2,
            processes,
            out_filename,
            out_filename_wrap,
            out_filename_both,
            out_filename_mask,
            out_premiere_filename,
            out_premiere_mask_filename,
            make_numpy_time,
            make_mask_time,
            crop_function_time,
        )
        id_array = (id_array + 1) % 8
    print(
        "Iteration :",
        it,
    )
    print("All work :", (datetime.now() - whole_time).total_seconds())
    print(f"make_mask : {make_mask_time / 10**6}")
    print(f"crop_function : {crop_function_time / 10**6}")
    print(f"make_numpy : {make_numpy_time / 10**6}")
    print(f"processes : {processes_time / 10**6}")
    print(f"premiere : {premiere_time / 10**6}")
    process2.stdin.close()
    for process in processes.values():
        process.stdin.close()

    process2.wait()
    for process in processes.values():
        process.wait()
    executor.shutdown()


@click.command()
@click.option("--input", required=True, type=str, help="path to your input video")
@click.option("--output", required=True, type=str, help="path to resulting video")
@click.option(
    "--wraper", default=None, type=str, help="path to resulting video with wrapping"
)
@click.option(
    "--compare",
    default=None,
    type=str,
    help="path to resulting video compared with input video",
)
@click.option(
    "--compare_mask",
    default=None,
    type=str,
    help="path to resulting video compared with input video with mask showing",
)
@click.option(
    "--input_compare", default=None, type=str, help="path to the video for comparing"
)
@click.option(
    "--output_compare",
    default=None,
    type=str,
    help="path to the resulting video compared input_compare video",
)
@click.option(
    "--output_compare_mask",
    default=None,
    type=str,
    help="path to the resulting video compared input_compare video with mask showing",
)
def read_flags(
    input,
    output,
    wraper,
    compare,
    compare_mask,
    input_compare,
    output_compare,
    output_compare_mask,
):
    make_videos(
        input,
        output,
        out_filename_wrap=wraper,
        out_filename_both=compare,
        out_filename_mask=compare_mask,
        in_premiere_filename=input_compare,
        out_premiere_filename=output_compare,
        out_premiere_mask_filename=output_compare_mask,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", True)
    # read_flags()
    # flags_val = dict(
    #     {
    #         # "in_filename": "/home/daria/Documents/pro/diploma/videos/Fits.mp4",
    #         # "out_filename": "/home/daria/Documents/pro/diploma/videos/Fits_command_line.mp4",
    #         "in_filename": None,
    #         "out_filename": None,
    #         "out_filename_wrap": None,
    #         "out_filename_both": None,
    #         "out_filename_mask": None,
    #         "in_premiere_filename": None,
    #         "out_premiere_filename": None,
    #         "out_premiere_mask_filename": None,
    #     }
    # )
    for in_filename in all_videos:
        make_videos(
            os.path.join(root_dir, "videos/" + in_filename + ".mp4"),
            os.path.join(root_dir, "videos/" + in_filename + "_cur.mp4"),
            in_premiere_filename=os.path.join(
                root_dir, "videos/premiere/" + in_filename + ".mp4"
            ),
            out_filename_wrap=os.path.join(
                root_dir, "videos/" + in_filename + "_wrap.mp4"
            ),
            out_filename_both=os.path.join(
                root_dir, "videos/" + in_filename + "_both.mp4"
            ),
            out_filename_mask=os.path.join(
                root_dir, "videos/" + in_filename + "_mask.mp4"
            ),
            out_premiere_filename=os.path.join(
                root_dir, "videos/" + in_filename + "_premiere.mp4"
            ),
            out_premiere_mask_filename=os.path.join(
                root_dir, "videos/" + in_filename + "_premiere_mask.mp4"
            ),
        )

# Fits
# 0
# Iteration : 202
# All work : 64.293668
# make_mask : 19.872874
# crop_function : 4.230969
# make_numpy : 1.155165
# processes : 37.948042
# premiere : 0.003809
# 138731143 - 3
# 197553477 - 8

# Dogs
# 0
# Iteration : 497
# All work : 202.830247
# make_mask : 63.349225
# crop_function : 10.93727
# make_numpy : 3.395281
# processes : 122.271023
# premiere : 0.011215
# 492008769 - 8

# HospitalTrick
# 0
# Iteration : 166
# All work : 23.422753
# make_mask : 12.271159
# crop_function : 1.103546
# make_numpy : 0.328848
# processes : 8.856382
# premiere : 0.00241

# Domino
# 0
# Iteration : 966
# All work : 259.218114
# make_mask : 90.026925
# crop_function : 15.039175
# make_numpy : 3.620245
# processes : 141.795894
# premiere : 3.480874

# Sherbakov
# 0
# Iteration : 993
# All work : 240.303535
# make_mask : 81.475975
# crop_function : 14.975807
# make_numpy : 3.391496
# processes : 132.469432
# premiere : 3.476289

# Masyanya
# 0
# Iteration : 1156
# All work : 251.45128
# make_mask : 75.054891
# crop_function : 16.039096
# make_numpy : 4.540995
# processes : 152.550436
# premiere : 0.017667

# NaskaFigurist
# 0
# Iteration : 578
# All work : 86.352523
# make_mask : 45.262637
# crop_function : 4.157637
# make_numpy : 1.001158
# processes : 32.697212
# premiere : 0.00884

# NaskaFigurist2
# 0
# Iteration : 898
# All work : 279.999228
# make_mask : 94.04784
# crop_function : 12.527587
# make_numpy : 3.777386
# processes : 164.277239
# premiere : 0.015305

# DenkaPastuh
# 0
# Iteration : 15623
# All work : 619.974231
# make_mask : 184.506371
# crop_function : 35.27427
# make_numpy : 8.394206
# processes : 381.591798
# premiere : 0.031176

# DenkaPastuh2
# 0
# Iteration : 17039
# All work : 453.129789
# make_mask : 194.536499
# crop_function : 20.324122
# make_numpy : 7.130253
# processes : 217.670513
# premiere : 0.030084

# Sobaki
# 0
# Iteration : 10345
# All work : 212.873195
# make_mask : 96.987275
# crop_function : 10.85541
# make_numpy : 3.801291
# processes : 93.334737
# premiere : 2.259186

# Sobaki2
# 0
# Iteration : 761
# All work : 83.36558
# make_mask : 47.296938
# crop_function : 4.294106
# make_numpy : 1.178804
# processes : 27.843399
# premiere : 0.010694

# Richi
# 0
# Iteration : 12863
# All work : 451.302306
# make_mask : 144.698839
# crop_function : 26.494984
# make_numpy : 8.963305
# processes : 263.101963
# premiere : 0.025751


# import vapoursynth as vs
# import ffmpeg
# from vapoursynth import core
# import os
# import numpy as np
# from constants import root_dir

# core.std.LoadPlugin(
#     path="/home/linuxbrew/.linuxbrew/Cellar/ffms2/2.40_2/lib/libffms2.so"
# )

# core.std.LoadPlugin(
#     path="/home/linuxbrew/.linuxbrew/Cellar/mvtools/23_1/lib/libmvtools.so"
# )

# input_video = core.ffms2.Source(source=os.path.join(root_dir, "videos/Masyanya.mp4"))
# video = core.resize.Bicubic(
#     clip=input_video,
#     width=input_video.width,
#     height=input_video.height,
#     format=vs.YUV444P8,
# )
# super_video = core.mv.Super(video)
# video_vector = core.mv.Analyse(super_video)

# result = core.mv.SCDetection(video, video_vector)

# process2 = (
#     ffmpeg.input(
#         "pipe:",
#         format="rawvideo",
#         pix_fmt="rgb24",
#         s="{}x{}".format(video.width, video.height),
#     )
#     .filter("fps", fps=video.fps, round="up")
#     .output(os.path.join(root_dir, "scene.mp4"))
#     .overwrite_output()
#     .run_async(pipe_stdin=True)
# )


# for cur_frame in result.frames():
#     frame = np.stack([np.array(cur_col) for cur_col in cur_frame]).astype(np.uint8)
#     print(frame.max(), frame.min())
#     process2.stdin.write(frame.tobytes())

# process2.stdin.close()
# process2.wait()
