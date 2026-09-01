from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


RAW_ROOT = Path("data/raw/tcp_ci")

DOWNLOAD_ROOT = RAW_ROOT / "downloads"

DEFAULT_ARCHIVE = (
    DOWNLOAD_ROOT
    / "TCP-CI-main-dataset.tar.gz"
)

EXPECTED_ROOT = RAW_ROOT / "main_dataset"
PROJECT_ROOT = EXPECTED_ROOT / "datasets"

EXPECTED_SHA256 = "92af159c116e06e98d7c8348605adb6e9acb25aad982a1ff9fbef7103cb9e36e"

PILOT_PROJECTS = [
    "camunda@camunda-bpm-platform",
    "apache@shardingsphere",
    "facebook@buck",
    "Angel-ML@angel",
    "zolyfarkas@spf4j",
]


REQUIRED_FILES = [
    "exe.csv",
    "builds.csv",
    "id_map.csv",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dataset(
    project_root: Path = PROJECT_ROOT,
) -> None:
    if not project_root.is_dir():
        raise FileNotFoundError(
            "Expected TCP-CI project root "
            f"not found: {project_root}"
        )

    missing: list[str] = []

    for project in PILOT_PROJECTS:
        root = project_root / project

        for filename in REQUIRED_FILES:
            path = root / filename

            if not path.is_file():
                missing.append(str(path))

    if missing:
        details = "\n".join(
            f" - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required TCP-CI pilot files "
            "are missing:\n"
            f"{details}"
        )


def safe_extract(
    archive: tarfile.TarFile,
    destination: Path,
) -> None:
    destination_resolved = destination.resolve()

    for member in archive.getmembers():
        member_path = (
            destination / member.name
        ).resolve()

        if (
            member_path != destination_resolved
            and destination_resolved
            not in member_path.parents
        ):
            raise ValueError(
                "Unsafe archive member path: "
                f"{member.name}"
            )

    try:
        archive.extractall(
            destination,
            filter="data",
    )
    except TypeError:
        # Compatibility with Python versions
        # that do not support extraction filters.
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and verify the external "
            "TCP-CI v3 main dataset."
        )
    )

    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=(
            "Path to TCP-CI-main-dataset.tar.gz. "
            "Default: "
            "data/raw/tcp_ci/downloads/"
            "TCP-CI-main-dataset.tar.gz"
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already extracted "
            "TCP-CI dataset without extracting."
        ),
    )

    args = parser.parse_args()

    if args.verify_only:
        validate_dataset()

        print(
            "[PASS] TCP-CI pilot raw inputs "
            "are available."
        )

        print(
            f"[INFO] Project root: {PROJECT_ROOT}"
        )

        return

    archive_path = args.archive

    if not archive_path.is_file():
        raise FileNotFoundError(
            "TCP-CI archive not found.\n"
            f"Expected archive: {archive_path}\n"
            "Provide the TCP-CI v3 main-dataset "
            "archive and rerun this command."
        )

    print(f"[INFO] Archive: {archive_path}")


    actual_sha256 = sha256_file(archive_path)

    print(
        "[INFO] SHA256: "
        f"{actual_sha256}"
    )

    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "TCP-CI archive SHA256 mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha256}\n"
            "The supplied archive does not match "
            "the frozen TCP-CI artifact used by "
            "this study."
        )

    print(
        "[PASS] TCP-CI archive SHA256 verified."
    )

    if PROJECT_ROOT.is_dir():
        print(
            "[INFO] Existing extracted dataset "
            "found; validating it."
        )

        validate_dataset()

        print(
            "[PASS] Existing TCP-CI dataset is "
            "valid for the five-project pilot."
        )

        return

    EXPECTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[INFO] Extracting archive into: "
        f"{EXPECTED_ROOT}"
    )

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as archive:
        safe_extract(
            archive,
            EXPECTED_ROOT,
        )

    validate_dataset()

    print(
        "[PASS] TCP-CI v3 main dataset "
        "prepared successfully."
    )

    print(
        f"[INFO] Project root: {PROJECT_ROOT}"
    )


if __name__ == "__main__":
    main()