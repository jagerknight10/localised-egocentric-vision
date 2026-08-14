from scripts.cache_gtea_features import build_parser


def test_batch_runner_defaults_are_configurable() -> None:
    args = build_parser().parse_args([])
    assert args.stride == 15
    assert args.overwrite is False
    assert args.videos.name == "videos"
