import os
import pathlib
import subprocess

TEST_E2E_DIR = pathlib.Path(__file__).parent.parent
TEST_E2E_TESTDATA = TEST_E2E_DIR.joinpath("testdata")


def extract_tarball(name: str | pathlib.Path) -> pathlib.Path:
    """
    Extract tarball with the given path to the directory that that tarball is
    in.

    The returned path is the folder containing the content of the tarball, named
    after the tarball name itself without the extension.
    """
    name = pathlib.Path(name).resolve()

    dest = name
    while dest.suffix:
        dest = dest.with_suffix("")

    if not dest.is_dir():
        if not name.is_file():
            raise FileNotFoundError(f"Tarball {name} not found!")

        os.makedirs(dest)
        subprocess.run(["tar", "-xzvf", name, "-C", dest], check=True)

    return dest
