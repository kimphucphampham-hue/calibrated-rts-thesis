import tarfile
from pathlib import Path

import pytest

import scripts.setup_tcp_ci as setup


def create_fake_dataset(root: Path) -> None:
    for project in setup.PILOT_PROJECTS:
        project_root = (
            root
            / "datasets"
            / project
        )

        project_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename in setup.REQUIRED_FILES:
            (
                project_root
                / filename
            ).write_text(
                "fixture\n",
                encoding="utf-8",
            )


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "abc",
        encoding="utf-8",
    )

    assert (
        setup.sha256_file(path)
        == (
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad"
        )
    )


def test_required_pilot_files_are_defined():
    assert setup.REQUIRED_FILES == [
        "exe.csv",
        "builds.csv",
        "id_map.csv",
    ]


def test_five_pilot_projects_are_frozen():
    assert len(setup.PILOT_PROJECTS) == 5
    assert len(set(setup.PILOT_PROJECTS)) == 5

