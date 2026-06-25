#!/usr/bin/env python3
import requests
import subprocess
from pathlib import Path

def download_and_extract_bdortho(dept: str, year: str):
    archive_name = f"BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D{dept}_{year}-01-01"
    base_url = f"https://data.geopf.fr/telechargement/download/BDORTHO/{archive_name}/{archive_name}.7z"
    output_dir = Path(f"D{dept}_{year}")
    output_dir.mkdir(exist_ok=True)

    print(f"Téléchargement D{dept}_{year}...")
    for i in range(1, 50):
        part_url = f"{base_url}.{i:03d}"
        r = requests.head(part_url)
        if r.status_code == 404:
            print(f"  → {i-1} partie(s)")
            break
        print(f"  Part {i:03d}...")
        subprocess.run(["wget", "-q", "--show-progress", "-P", str(output_dir), "-c", part_url], check=True)

    print("Extraction...")
    subprocess.run(["7z", "x", str(output_dir / f"{archive_name}.7z.001"), f"-o{output_dir}", "-y"], check=True)

    for part in output_dir.glob("*.7z.*"):
        part.unlink()
    print(f"Done → {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Télécharge et extrait la BDORTHO RVB par département.")
    parser.add_argument("dept", help="Numéro de département (ex: 017)")
    parser.add_argument("year", help="Année (ex: 2024)")
    args = parser.parse_args()
    download_and_extract_bdortho(dept=args.dept, year=args.year)