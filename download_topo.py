#!/usr/bin/env python3
import requests
import subprocess
import sys
from pathlib import Path

def download_and_extract_bdtopo(dept: str, year: str, date: str):
    """
    dept : '017'
    year : '2026'
    date : '2026-03-15'
    """
    archive_name = f"BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D{dept}_{date}"
    url = f"https://data.geopf.fr/telechargement/download/BDTOPO/{archive_name}/{archive_name}.7z"
    output_dir = Path(f"D{dept}_{year}")
    output_dir.mkdir(exist_ok=True)

    # 1. Téléchargement
    print(f"Téléchargement D{dept}_{year}...")
    r = requests.head(url)
    if r.status_code == 404:
        print(f"Archive introuvable : {url}")
        sys.exit(1)

    subprocess.run(["wget", "-q", "--show-progress", "-P", str(output_dir), "-c", url], check=True)

    # 2. Extraction
    print("Extraction...")
    archive_path = output_dir / f"{archive_name}.7z"
    subprocess.run(["7zr", "x", str(archive_path), f"-o{output_dir}", "-y"], check=True)

    # 3. Nettoyage
    archive_path.unlink()
    print(f"Done → {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Télécharge et extrait la BDTOPO par département.")
    parser.add_argument("dept", help="Numéro de département (ex: 017)")
    parser.add_argument("year", help="Année (ex: 2026)")
    parser.add_argument("date", help="Date de la version (ex: 2026-03-15)")
    args = parser.parse_args()
    download_and_extract_bdtopo(dept=args.dept, year=args.year, date=args.date)