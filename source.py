import os
import click

from constants import root_dir, all_videos

from make_videos import make_videos


@click.command()
@click.option("--input", required=True, type=str, help="path to your input video")
@click.option("--output", required=True, type=str, help="path to resulting video")
@click.option(
    "--wrapper", default=None, type=str, help="path to resulting video with wrapping"
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
@click.option(
    "--default_speed",
    default=None,
    type=float,
    help="set fixed speed for frame",
)
@click.option(
    "--scene_detection_flag",
    default=None,
    type=bool,
    help="if flag True, programm will use scene detection",
)
def read_flags(
    input,
    output,
    wrapper,
    compare,
    compare_mask,
    input_compare,
    output_compare,
    output_compare_mask,
    default_speed,
    scene_detection_flag,
):
    make_videos(
        input,
        output,
        out_filename_wrap=wrapper,
        out_filename_both=compare,
        out_filename_mask=compare_mask,
        in_compare_filename=input_compare,
        out_compare_filename=output_compare,
        out_compare_mask_filename=output_compare_mask,
        constant_speed=default_speed,
        scene_detection_flag=scene_detection_flag,
    )


if __name__ == "__main__":
    # read_flags()
    for in_filename in all_videos:
        print(in_filename)
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
