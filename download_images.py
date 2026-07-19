#!/usr/bin/env python3
import subprocess
from pathlib import Path

def try_download_version(archive_name: str, base_url: str, output_dir: Path) -> bool:
    """Tente de télécharger toutes les parties. Retourne False si le part 001 échoue (mauvaise version)."""
    for i in range(1, 50):
        part_url = f"{base_url}.{i:03d}"
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-P", str(output_dir), "-c", part_url],
            check=False
        )
        if result.returncode != 0:
            if i == 1:
                return False  # cette version n'existe pas du tout
            print(f"  → {i-1} partie(s)")
            return True
        print(f"  Part {i:03d}...")
    return True

def download_and_extract_bdortho(dept: str, year: str):
    output_dir = Path(f"D{dept}_{year}")
    output_dir.mkdir(exist_ok=True)

    archive_name = None
    for version in ["2-0", "1-0"]:
        candidate = f"BDORTHO_{version}_RVB-0M20_JP2-E080_LAMB93_D{dept}_{year}-01-01"
        base_url = f"https://data.geopf.fr/telechargement/download/BDORTHO/{candidate}/{candidate}.7z"
        print(f"Tentative version {version}...")
        if try_download_version(candidate, base_url, output_dir):
            archive_name = candidate
            break

    if archive_name is None:
        raise RuntimeError(f"Aucune version trouvée pour D{dept}_{year}")

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
