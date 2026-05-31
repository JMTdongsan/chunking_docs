from pathlib import Path

import httpx


URL = "https://urban.seoul.go.kr/UpisArchive/DATA/PWEB/STATIC/1_seoul_plan.pdf"
OUTPUT = Path("data/raw/1_seoul_plan.pdf")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with OUTPUT.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    print(f"downloaded {OUTPUT}")


if __name__ == "__main__":
    main()
