from vapoursynth import core


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
