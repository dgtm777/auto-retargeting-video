import numpy as np
import os
import random
from decoration import avs_subtitle
from constants import all_videos, root_dir
import constants
from vapoursynth import core
import vapoursynth as vs
import ffmpeg
from make_videos import make_videos
import cv2


# core.std.LoadPlugin(
#     path="/home/linuxbrew/.linuxbrew/Cellar/ffms2/2.40_2/lib/libffms2.so"
# )

# core.std.LoadPlugin(
#     path="/home/linuxbrew/.linuxbrew/Cellar/mvtools/23_1/lib/libmvtools.so"
# )


def frame_to_numpy(frame, shape):
    image = np.stack([np.array(cur_col) for cur_col in frame]).astype(np.uint8)
    image = np.moveaxis(
        image,
        0,
        -1,
    )
    image = cv2.resize(image, dsize=shape)
    return image


def make_presentation(
    my_filename, premier_filename, out_filename_secret, out_filename_labeled, labeles
):
    """
    Функция для сравнения разных вариаций изменения соотношения сторон видео
    """
    comp_video = core.ffms2.Source(source=premier_filename)
    my_video = core.ffms2.Source(source=my_filename)
    comp_video = core.resize.Bicubic(
        clip=comp_video,
        width=comp_video.width,
        height=comp_video.height,
        format=vs.RGB24,
        matrix_in_s="709",
    )
    my_video = core.resize.Bicubic(
        clip=my_video,
        width=my_video.width,
        height=my_video.height,
        format=vs.RGB24,
        matrix_in_s="709",
    )
    if comp_video.width > my_video.width or comp_video.height > my_video.height:
        shape = (comp_video.width, comp_video.height)
    else:
        shape = (my_video.width, my_video.height)
    height = shape[1]
    width = shape[0]
    if my_video.height <= my_video.width:
        func = np.vstack
        height += shape[1]
    else:
        func = np.hstack
        width += shape[0]
    comp = (comp_video, labeles[1])
    my = (my_video, labeles[0])
    sequences = random.choice([[comp, my], [my, comp]])
    subtitled_seq = [
        avs_subtitle(sequences[0][0], sequences[0][1]),
        avs_subtitle(sequences[1][0], sequences[1][1]),
    ]
    process1 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
        )
        .filter("setpts", f"N/({my_video.fps}*TB)+STARTPTS")
        .filter("fps", fps=my_video.fps, round="up")
        .output(out_filename_secret, preset="ultrafast", pix_fmt="yuv420p")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    process2 = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
        )
        .filter("setpts", f"N/({my_video.fps}*TB)+STARTPTS")
        .filter("fps", fps=my_video.fps, round="up")
        .output(
            out_filename_labeled,
            preset="ultrafast",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for left, right, left_labeled, right_labeled in zip(
        sequences[0][0].frames(),
        sequences[1][0].frames(),
        subtitled_seq[0].frames(),
        subtitled_seq[1].frames(),
    ):
        process1.stdin.write(
            func((frame_to_numpy(left, shape), frame_to_numpy(right, shape)))
            .astype(np.uint8)
            .tobytes()
        )
        process2.stdin.write(
            func(
                (
                    frame_to_numpy(left_labeled, shape),
                    frame_to_numpy(right_labeled, shape),
                )
            )
            .astype(np.uint8)
            .tobytes()
        )
    process1.stdin.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()


# сравнение с альтернативными алгоритмами изменения соотношения сторон видео
comp_names = [
    # "premiere",
    # "multimedia",
    "google",
]

# root_dir_cur = "/media/daria/5AAE-51EA"

# for comp in comp_names:
#     for in_filename in all_videos:
#         print(in_filename)
#         # comp_video = core.ffms2.Source(
#         #     os.path.join(root_dir_cur, "videos/" + comp + "/" + in_filename + ".mp4"),
#         # )
#         # make_videos(
#         #     os.path.join(root_dir_cur, "videos/" + in_filename + ".mp4"),
#         #     os.path.join(
#         #         root_dir_cur,
#         #         "videos/" + in_filename + "_cur.mp4",
#         #     ),
#         #     # ratio=(comp_video.height, comp_video.width),
#         # )
#         make_presentation(
#             os.path.join(root_dir_cur, "videos/" + in_filename + "_cur.mp4"),
#             os.path.join(root_dir_cur, "videos/" + comp + "/" + in_filename + ".mp4"),
#             os.path.join(
#                 root_dir_cur, "videos/presentation/" + comp + "/" + in_filename + ".mp4"
#             ),
#             os.path.join(
#                 root_dir_cur,
#                 "videos/presentation/" + comp + "/" + in_filename + "_labeled.mp4",
#             ),
#             ("mine", comp),
#         )


videos = [
    # "Sherbakov",
    # "Svaty1",
    # "Svaty2",
    # "Svaty3",
    # "trailer",
    # "Masyanya",
    # "TheBoysScene",
    # "Fits",
    # "Dogs",
    # "HospitalTrick",
    # "Domino",
    # "NaskaFigurist",
    # "NaskaFigurist2",
    # "DenkaPastuh",
    # "DenkaPastuh2",
    # "Sobaki",
    # "Richi",
    # "Yulka",
    # "Yulka2",
    # "Denka1",
]


modes = [
    # (
    #     "weighted_sum/",
    #     ("weighted sum", "no weighted sum"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         constants.jump_coef_mask_value,
    #         False,
    #     ),
    # ),
    # (
    #     "mask_coef/",
    #     ("mask coef", "no mask coef"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         1,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         constants.jump_coef_mask_value,
    #         True,
    #     ),
    # ),
    # (
    #     "wrong_scene/",
    #     ("no wrong scene", "wrong scene"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         0,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         constants.jump_coef_mask_value,
    #         True,
    #     ),
    # ),
    # (
    #     "shake/",
    #     ("no shake", "shake"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         0,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         constants.jump_coef_mask_value,
    #         True,
    #     ),
    # ),
    # (
    #     "prev_speed/",
    #     ("prev speed", "no prev speed"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         0,
    #         constants.jump_coef_img_size,
    #         True,
    #     ),
    # ),
    # (
    #     "jump available/",
    #     ("jump available", "no jump available"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         1,
    #         True,
    #     ),
    # ),
    # (
    #     "future/",
    #     ("future", "no future"),
    #     (
    #         0,
    #         True,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         True,
    #     ),
    # ),
    # (
    #     "scene_detection/",
    #     ("scene detection", "no scene detection"),
    #     (
    #         constants.future_speed_coef,
    #         False,
    #         (None,),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         True,
    #     ),
    # ),
    # (
    #     "speed/",
    #     ("adaptive speed", "constant speed"),
    #     (
    #         constants.future_speed_coef,
    #         True,
    #         (0, 0.002, 0.004, 0.006, 0.008, 0.01),
    #         constants.fps_coef,
    #         constants.jump_delay_coef,
    #         constants.moving_available_coef,
    #         constants.mask_coef,
    #         constants.prev_speed_coef,
    #         constants.jump_coef_img_size,
    #         True,
    #     ),
    # ),
]

# Сравнение разных вариаций работы алгоритма
for place, labeles, flags in modes:
    print(place)
    for in_filename in videos:
        print(in_filename)
        for speed_flag in flags[2]:
            label = ""
            if speed_flag is not None:
                label = f"_{speed_flag}"
            make_videos(
                os.path.join(root_dir, "videos/" + in_filename + ".mp4"),
                os.path.join(
                    root_dir,
                    "videos/presentation/" + place + in_filename + label + ".mp4",
                ),
                future_speed_coef=flags[0],
                scene_detection_flag=flags[1],
                constant_speed=speed_flag,
                fps_coef=flags[3],
                jump_delay_coef=flags[4],
                moving_available_coef=flags[5],
                mask_coef=flags[6],
                prev_speed_coef=flags[7],
                jump_coef_img_size=flags[8],
                weighted_sum=flags[9],
            )
        if len(flags[2]) == 1:
            make_presentation(
                os.path.join(root_dir, "videos/" + in_filename + "_cur.mp4"),
                os.path.join(
                    root_dir, "videos/presentation/" + place + in_filename + ".mp4"
                ),
                os.path.join(
                    root_dir, "videos/presentation/" + place + in_filename + "_both.mp4"
                ),
                os.path.join(
                    root_dir,
                    "videos/presentation/" + place + in_filename + "_labeled.mp4",
                ),
                labeles,
            )


# Сравнения работы алгоритма при фиксированной и адаптивной скорости движения окна обрезки
place = "speed/"
names = [
    # ("Denka1", "_0.01"),
    # ("Dogs", "_0.01"),
    # ("NaskaFigurist2", "_0.006"),
    # ("Svaty1", "_0.002"),
    # ("Svaty2", "_0.002"),
]
for in_filename, adds in names:
    print("filename")
    make_presentation(
        os.path.join(root_dir, "videos/" + in_filename + "_cur.mp4"),
        os.path.join(
            root_dir, "videos/presentation/" + place + in_filename + adds + ".mp4"
        ),
        os.path.join(
            root_dir, "videos/presentation/" + place + in_filename + "_both.mp4"
        ),
        os.path.join(
            root_dir,
            "videos/presentation/" + place + in_filename + "_labeled.mp4",
        ),
        ("adaptive speed", "constant speed = " + adds[1:]),
    )
