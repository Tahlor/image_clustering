"""Tests for independent folder sequence discovery."""

from pathlib import Path

from image_clustering.clustering.discovery import discover_sequences


def test_sequences_are_sorted_and_never_cross_folders(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for path in (first / "b.jpg", first / "a.jpg", second / "a.jpg"):
        path.write_bytes(b"placeholder")

    sequences = discover_sequences(input_dir=tmp_path)
    assert [[image.image_id for image in sequence] for sequence in sequences] == [
        ["first/a.jpg", "first/b.jpg"],
        ["second/a.jpg"],
    ]
