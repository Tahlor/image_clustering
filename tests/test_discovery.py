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


def test_triplet_manifest_creates_disjoint_sequences_and_ignores_unlisted_files(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media-a"
    media.mkdir()
    for filename in ("a.j2k", "b.j2k", "c.j2k", "d.j2k"):
        (media / filename).write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "target_index,relation,source_sample_row,neighbor_of,project_id,media_item_id,sequence_index,filename,file_size\n"
        "1,predecessor,10,10,project,media-a,0,a.j2k,1\n"
        "2,sample,10,10,project,media-a,1,b.j2k,1\n"
        "3,successor,10,10,project,media-a,2,c.j2k,1\n"
        "4,sample,20,20,project,media-a,3,d.j2k,1\n",
        encoding="utf-8",
    )

    from image_clustering.clustering.discovery import discover_triplet_sequences

    sequences = discover_triplet_sequences(tmp_path, manifest)
    assert [[image.path.name for image in sequence] for sequence in sequences] == [
        ["a.j2k", "b.j2k", "c.j2k"],
        ["d.j2k"],
    ]
    assert all(
        "triplet/" in image.sequence_id
        for sequence in sequences
        for image in sequence
    )
