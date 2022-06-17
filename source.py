import os
import click

from constants import root_dir, all_videos

from make_videos import make_videos
import constants
from datetime import datetime


@click.command()
@click.option(
    "--input", required=True, type=click.Path(), help="path to your input video"
)
@click.option(
    "--output", required=True, type=click.Path(), help="path to resulting video"
)
@click.option(
    "--height",
    default=1,
    type=click.IntRange(min=0, min_open=True),
    help="ratio value",
)
@click.option(
    "--width",
    default=1,
    type=click.IntRange(min=0, min_open=True),
    help="ratio value",
)
@click.option(
    "--wrapper",
    default=None,
    type=click.Path(),
    help="path to resulting video with wrapping",
)
@click.option(
    "--compare",
    default=None,
    type=click.Path(),
    help="path to resulting video compared with input video",
)
@click.option(
    "--compare_with_mask",
    default=None,
    type=click.Path(),
    help="path to resulting video compared with mask of the input video",
)
@click.option(
    "--compare_mask",
    default=None,
    type=click.Path(),
    help="path to resulting video compared with input video with mask showing",
)
@click.option(
    "--input_compare",
    default=None,
    type=click.Path(),
    help="path to the video for comparing",
)
@click.option(
    "--output_compare",
    default=None,
    type=click.Path(),
    help="path to the resulting video compared input_compare video",
)
@click.option(
    "--output_compare_mask",
    default=None,
    type=click.Path(),
    help="path to the resulting video compared input_compare video with mask showing",
)
@click.option(
    "--default_speed",
    default=None,
    type=float,
    help="set fixed speed for frame",
)
@click.option(
    "--scene_detection_flag",
    default=True,
    type=bool,
    help="if flag True, programm will use scene detection",
)
@click.option(
    "--scene_detection_ysc",
    default=255,
    type=click.IntRange(min=0),
    help="parameter for MVTools  MSCDetection",
)
@click.option(
    "--scene_detection_thSCD1",
    default=100,
    type=click.IntRange(min=0),
    help="parameter for MVTools  MSCDetection",
)
@click.option(
    "--speed_coef",
    default=constants.speed_coef,
    type=click.FloatRange(min=0, min_open=True),
    help="speed calculated as value in range (-128; 128) divided by speed_coef",
)
@click.option(
    "--fps_coef",
    default=constants.fps_coef,
    type=click.FloatRange(min=0, min_open=True),
    help="coefficient indicating how far into future is possible to look to calculate the speed",
)
@click.option(
    "--future_speed_coef",
    default=constants.future_speed_coef,
    type=click.FloatRange(0, 1),
    help="coefficient indicating how much weight has the future speed, when the current speed calculating \n Set it 0 to reduce future speed",
)
@click.option(
    "--prev_speed_coef",
    default=constants.prev_speed_coef,
    type=click.FloatRange(0, 1),
    help="coefficient indicating how much weight has the previous speed, when the current speed calculating",
)
@click.option("--mask_coef", default=5, type=float, help="add weight to values in mask")
@click.option(
    "--speed_error",
    default=constants.speed_error,
    type=click.FloatRange(min=0, max=1),
    help="what part of frame pixels have to move to move the frame",
)
@click.option(
    "--jump_coef_wrap_size",
    default=constants.jump_coef_wrap_size,
    type=click.FloatRange(min=0),
    help="if the frame want to move on jump_coef_wrap_size of its size the moving is allowed",
)
@click.option(
    "--jump_coef_img_size",
    default=constants.jump_coef_img_size,
    type=click.FloatRange(min=0),
    help="if the frame want to move on jump_coef_img_size of image size the moving is allowed",
)
@click.option(
    "--jump_coef_mask_value",
    default=constants.jump_coef_mask_value,
    type=click.FloatRange(min=0),
    help="ignored difference in mask weights",
)
@click.option(
    "--jump_delay_coef",
    default=constants.jump_delay_coef,
    type=click.FloatRange(min=0),
    help="number of seconds waited before strong changing \n should be less than future_speed_coef",
)
@click.option(
    "--moving_available_coef",
    default=constants.moving_available_coef,
    type=click.FloatRange(min=0),
    help="requeired number of seconds with willing to move to the same direction to start moving",
)
@click.option(
    "--weighted_sum",
    default=True,
    type=bool,
    help="use weighted sum to find mask area with hightest sum of probabilities ",
)
def read_flags(
    input,
    output,
    height,
    width,
    wrapper,
    compare,
    compare_with_mask,
    compare_mask,
    input_compare,
    output_compare,
    output_compare_mask,
    default_speed,
    scene_detection_flag,
    scene_detection_ysc,
    scene_detection_thSCD1,
    speed_coef,
    fps_coef,
    future_speed_coef,
    prev_speed_coef,
    mask_coef,
    speed_error,
    jump_coef_wrap_size,
    jump_coef_img_size,
    jump_coef_mask_value,
    jump_delay_coef,
    moving_available_coef,
    weighted_sum,
):
    make_videos(
        input,
        output,
        (height, width),
        mask_coef=mask_coef,
        constant_speed=default_speed,
        speed_error=speed_error,
        speed_coef=speed_coef,
        future_speed_coef=future_speed_coef,
        prev_speed_coef=prev_speed_coef,
        jump_coef_wrap_size=jump_coef_wrap_size,
        jump_coef_img_size=jump_coef_img_size,
        jump_coef_mask_value=jump_coef_mask_value,
        jump_delay_coef=jump_delay_coef,
        fps_coef=fps_coef,
        scene_detection_flag=scene_detection_flag,
        scene_detection_parameters=(scene_detection_ysc, scene_detection_thSCD1),
        out_filename_wrap=wrapper,
        out_filename_both=compare,
        out_filename_both_mask=compare_with_mask,
        out_filename_mask=compare_mask,
        in_compare_filename=input_compare,
        out_compare_filename=output_compare,
        out_compare_mask_filename=output_compare_mask,
        moving_available_coef=moving_available_coef,
        weighted_sum=weighted_sum,
    )


if __name__ == "__main__":
    # read_flags()
    for in_filename in all_videos:
        print("\n\n\n", in_filename, "\n\n\n")
        start = datetime.now()
        make_videos(
            os.path.join(root_dir, "videos/" + in_filename + ".mp4"),
            os.path.join(root_dir, "videos/" + in_filename + "_future_speed_cur.mp4"),
            # in_compare_filename=os.path.join(
            #     root_dir, "videos/premiere/" + in_filename + ".mp4"
            # ),
            # out_filename_wrap=os.path.join(
            #     root_dir, "videos/" + in_filename + "_wrap.mp4"
            # ),
            # out_filename_both=os.path.join(
            #     root_dir, "videos/" + in_filename + "_both.mp4"
            # ),
            # out_filename_both_mask=os.path.join(
            #     root_dir, "videos/" + in_filename + "_mask_coef_both.mp4"
            # ),
            # out_filename_mask=os.path.join(
            #     root_dir, "videos/" + in_filename + "_mask.mp4"
            # ),
            # out_compare_filename=os.path.join(
            #     root_dir, "videos/" + in_filename + "_comp_both_mask_coef.mp4"
            # ),
            # out_compare_mask_filename=os.path.join(
            #     root_dir, "videos/" + in_filename + "_mask.mp4"
            # ),
            # weighted_sum=False,
            # mask_coef=1,
            # jump_coef_img_size=0,
            # jump_coef_wrap_size=0,
            # jump_coef_mask_value=0,
            # jump_delay_coef=0,
            # moving_available_coef=0,
            # speed_error=0,
            # fps_coef=0,
            # prev_speed_coef=0,
            future_speed_coef=0,
            # scene_detection_parameters=(0, 0),
        )
        print(datetime.now() - start)
