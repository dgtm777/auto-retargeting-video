import ffmpeg
import numpy as np
from decoration import avs_subtitle
from vapoursynth import core
import vapoursynth as vs


def upload_videos(in_filename, in_compare_filename):

    input_video = core.ffms2.Source(source=in_filename)

    if in_compare_filename is not None:
        input_compare_video = core.ffms2.Source(source=in_compare_filename)

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

    if in_compare_filename is not None:
        input_compare_video = core.resize.Bicubic(
            clip=input_compare_video,
            width=input_compare_video.width,
            height=input_compare_video.height,
            # format=vs.YUV444P8,
            format=vs.RGB24,
            matrix_in_s="709",
        )
        input_compare_video = avs_subtitle(
            input_compare_video, in_compare_filename.split("/")[-2]
        )
    else:
        input_compare_video = input_video

    super_video = core.mv.Super(yuv_video)
    video_vector = core.mv.Analyse(super_video)
    scene_detection = core.mv.SCDetection(yuv_video, video_vector)
    return yuv_video, input_video, input_compare_video, video_vector, scene_detection


def check_compare_shape(compare_image, img, func):
    if compare_image.shape[0] < img[1].shape[0]:
        compare_image = func(
            (
                compare_image,
                np.zeros(
                    shape=(
                        img[1].shape[0] - compare_image.shape[0],
                        img[1].shape[1],
                        img[1].shape[2],
                    )
                ),
            )
        )

    if compare_image.shape[1] < img[1].shape[1]:
        compare_image = func(
            (
                compare_image,
                np.zeros(
                    shape=(
                        img[1].shape[0],
                        img[1].shape[1] - compare_image.shape[1],
                        img[1].shape[2],
                    )
                ),
            )
        )
    return compare_image


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
    out_compare_filename=None,
    out_compare_mask_filename=None,
):
    processes = dict()
    processes[out_filename] = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(new_width, new_height),
        )
        .filter("setpts", f"N/({fps}*TB)+STARTPTS")
        .filter("fps", fps=fps, round="up")
        .output(out_filename, preset="ultrafast", pix_fmt="yuv420p")
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
            .filter("setpts", f"N/({fps}*TB)+STARTPTS")
            .filter("fps", fps=fps, round="up")
            .output(out_filename_wrap, preset="ultrafast", pix_fmt="yuv420p")
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
            .filter("setpts", f"N/({fps}*TB)+STARTPTS")
            .filter("fps", fps=fps, round="up")
            .output(out_filename_both, preset="ultrafast", pix_fmt="yuv420p")
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
            .filter("setpts", f"N/({fps}*TB)+STARTPTS")
            .filter("fps", fps=fps, round="up")
            .output(out_filename_mask, preset="ultrafast", pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_compare_filename is not None:
        processes[out_compare_filename] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(coef_v * new_width, coef_h * new_height),
            )
            .filter("setpts", f"N/({fps}*TB)+STARTPTS")
            .filter("fps", fps=fps, round="up")
            .output(out_compare_filename, preset="ultrafast", pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    if out_compare_mask_filename is not None:
        processes[out_compare_mask_filename] = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(
                    coef_v * new_width + 2 * add_width,
                    coef_h * new_height + 2 * add_height,
                ),
            )
            .filter("setpts", f"N/({fps}*TB)+STARTPTS")
            .filter("fps", fps=fps, round="up")
            .output(out_compare_mask_filename, preset="ultrafast", pix_fmt="yuv420p")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    return processes


def set_processes(
    img,
    mask,
    cur_compare_image,
    func,
    processes,
    out_filename,
    out_filename_wrap=None,
    out_filename_both=None,
    out_filename_mask=None,
    out_compare_filename=None,
    out_compare_mask_filename=None,
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
    if out_compare_filename in processes:
        compare_image = np.stack(
            [np.array(cur_col) for cur_col in cur_compare_image]
        ).astype(np.uint8)
        compare_image = np.moveaxis(
            compare_image,
            0,
            -1,
        )
        compare_image = check_compare_shape(compare_image, img, func)

        processes[out_compare_filename].stdin.write(
            func((img[1], compare_image)).astype(np.uint8).tobytes()
        )
    if out_compare_mask_filename in processes:
        processes[out_compare_mask_filename].stdin.write(
            func((img[0], mask[0], img[1], compare_image)).astype(np.uint8).tobytes()
        )
