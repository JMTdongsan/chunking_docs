from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class TesseractOCRBackend:
    def __init__(self, executable: str = "tesseract"):
        self.executable = executable
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not installed or not on PATH")

    def recognize(self, image_path: Path, language: str = "kor+eng") -> str:
        result = subprocess.run(
            [self.executable, str(image_path), "stdout", "-l", language],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
