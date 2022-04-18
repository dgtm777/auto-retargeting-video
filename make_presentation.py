import numpy as np
import os
import random
from decoration import avs_subtitle
from constants import all_videos, root_dir
from vapoursynth import core
import vapoursynth as vs
import ffmpeg


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


def make_presentation(
    my_filename, premier_filename, out_filename_secret, out_filename_labeled
):
    premiere_video = core.ffms2.Source(source=premier_filename)
    my_video = core.ffms2.Source(source=my_filename)
    premiere_video = core.resize.Bicubic(
        clip=premiere_video,
        width=premiere_video.width,
        height=premiere_video.height,
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
    height = my_video.height
    width = my_video.width
    if my_video.height <= my_video.width:
        func = np.vstack
        height += premiere_video.height
    else:
        func = np.hstack
        width += premiere_video.width
    premiere = (premiere_video, "premiere")
    my = (my_video, "mine")
    sequences = random.choice([[premiere, my], [my, premiere]])
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
        .filter("fps", fps=my_video.fps, round="up")
        .output(
            out_filename_secret,
            preset="ultrafast",
        )
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
        .filter("fps", fps=my_video.fps, round="up")
        .output(
            out_filename_labeled,
            preset="ultrafast",
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
            func((frame_to_numpy(left), frame_to_numpy(right)))
            .astype(np.uint8)
            .tobytes()
        )
        process2.stdin.write(
            func((frame_to_numpy(left_labeled), frame_to_numpy(right_labeled)))
            .astype(np.uint8)
            .tobytes()
        )
    process1.stdin.close()
    process2.stdin.close()
    process1.wait()
    process2.wait()


for in_filename in all_videos:
    make_presentation(
        os.path.join(root_dir, "videos/" + in_filename + "_cur.mp4"),
        os.path.join(root_dir, "videos/premiere/" + in_filename + ".mp4"),
        os.path.join(root_dir, "videos/presentation/" + in_filename + ".mp4"),
        os.path.join(root_dir, "videos/presentation/" + in_filename + "_labeled.mp4"),
    )
