from scripts.cache_gtea_restricted_features import build_parser


def test_restricted_runner_accepts_one_video() -> None:
    args = build_parser().parse_args(["--video-name", "S1_Cheese_C1.mp4"])
    assert args.video_name == "S1_Cheese_C1.mp4"


def test_restricted_runner_exposes_cudnn_fallback() -> None:
    args = build_parser().parse_args(["--disable-cudnn"])
    assert args.disable_cudnn is True
