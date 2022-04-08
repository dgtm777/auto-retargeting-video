import ffmpeg
import numpy as np
from decoration import avs_subtitle
from vapoursynth import core
import vapoursynth as vs


def upload_videos(in_filename, in_premiere_filename):

    input_video = core.ffms2.Source(source=in_filename)

    if in_premiere_filename is not None:
        input_premiere_video = core.ffms2.Source(source=in_premiere_filename)

    yuv_video = core.resize.Bicubic(
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

    super_video = core.mv.Super(yuv_video)
    video_vector = core.mv.Analyse(super_video)
    return yuv_video, input_video, input_premiere_video, video_vector


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
    out_filename,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    out_premiere_filename=None,
    out_premiere_mask_filename=None,
):
    processes = dict()
    processes[out_filename] = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(new_width, new_height),
        )
        .filter("fps", fps=fps, round="up")
        .output(out_filename)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
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


def set_processes(
    img,
    mask,
    cur_premiere_image,
    func,
    processes,
    out_filename,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    out_premiere_filename=None,
    out_premiere_mask_filename=None,
):

    processes[out_filename].stdin.write(img[1].astype(np.uint8).tobytes())

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
