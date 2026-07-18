"""Tests for canonical result serialization."""

from pathlib import Path

from test_models import make_result

from image_clustering import load_result, write_result


def test_write_and_load_result(tmp_path: Path) -> None:
    result = make_result()
    write_result(result=result, output_dir=tmp_path)
    restored = load_result(path=tmp_path / "clustering.json")
    assert restored == result
    assert (tmp_path / "pair_scores.csv").is_file()
    assert (tmp_path / "run.json").is_file()
