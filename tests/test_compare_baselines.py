import json

from scripts.compare_baselines import build_parser, run


def test_compare_baselines_reads_saved_results(tmp_path, capsys) -> None:
    frame_dir = tmp_path / "frame"
    transformer_dir = tmp_path / "transformer"
    frame_dir.mkdir()
    transformer_dir.mkdir()
    for subject in ("S1", "S2", "S3", "S4"):
        for directory, accuracy, balanced in (
            (frame_dir, 0.4, 0.2),
            (transformer_dir, 0.5, 0.3),
        ):
            (directory / f"held_out_{subject}.json").write_text(
                json.dumps({"test_accuracy": accuracy, "balanced_accuracy": balanced})
            )
    args = build_parser().parse_args(["--frame-dir", str(frame_dir), "--transformer-dir", str(transformer_dir)])
    run(args)
    assert "mean test_accuracy: frame=0.4000 transformer=0.5000" in capsys.readouterr().out
