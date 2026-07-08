#!/bin/bash
set -uo pipefail

# -------- Config --------
DEPTS_YEARS=(
)

TOPO_YEAR="2026"
TOPO_DATE="2026-03-15"
CONFIG_FILE="/workspace/deeppvmapper/config.yml"
IMAGES_BASE="/workspace/images"
TOPO_BASE="/workspace/topo"
LOG_FILE="/workspace/deeppvmapper/pipeline_log.txt"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
TILES_JSON="${SCRIPT_DIR}/tiles_list.json"

USE_TILES=0

# -------- Arguments --------
for arg in "$@"; do
    case "$arg" in
        --tiles)
            USE_TILES=1
            ;;
        *)
            echo "Argument inconnu: $arg" >&2
            exit 1
            ;;
    esac
done

if [ "$USE_TILES" -eq 1 ] && [ ! -f "$TILES_JSON" ]; then
    echo "ERREUR: --tiles demande mais ${TILES_JSON} introuvable." >&2
    exit 1
fi

PREFETCH_IMG_PID=""
PREFETCH_TOPO_PID=""

echo "Pipeline demarre le $(date)" > $LOG_FILE

# -------- Fonctions --------
update_tiles_list() {
    local dept=$1
    local key="D${dept}"

    python3 - "$CONFIG_FILE" "$TILES_JSON" "$key" <<'PYEOF'
import sys, json, re

config_file, tiles_json, key = sys.argv[1], sys.argv[2], sys.argv[3]

with open(tiles_json) as f:
    data = json.load(f)

if key not in data:
    print(f"[TILES] ATTENTION: cle {key} absente de {tiles_json}, tiles_list laisse vide.", file=sys.stderr)
    tiles = []
else:
    tiles = data[key]

lines = ["tiles_list:"]
for t in tiles:
    lines.append(f'  - "{t}"')
new_block = "\n".join(lines)

with open(config_file) as f:
    content = f.read()

pattern = re.compile(r"^tiles_list:.*?(?=^\S|\Z)", re.DOTALL | re.MULTILINE)
if pattern.search(content):
    content = pattern.sub(new_block + "\n", content, count=1)
else:
    content = content.rstrip("\n") + "\n\n" + new_block + "\n"

with open(config_file, "w") as f:
    f.write(content)

print(f"[TILES] {len(tiles)} tuiles injectees pour {key}.")
PYEOF
}

update_config() {
    local dept=$1
    local year=$2
    sed -i "s|source_images_dir : '.*'|source_images_dir : '../images/D${dept}_${year}'|" $CONFIG_FILE
    sed -i "s|source_topo_dir : '.*'|source_topo_dir : '../topo/D${dept}_${TOPO_YEAR}'|" $CONFIG_FILE

    if [ "$USE_TILES" -eq 1 ]; then
        update_tiles_list "$dept"
    fi
}

download_next() {
    local dept=$1
    local year=$2

    if [ -d "${IMAGES_BASE}/D${dept}_${year}" ]; then
        echo "[PREFETCH] Images D${dept}_${year} deja presentes, skip."
        PREFETCH_IMG_PID=""
    else
        echo "[PREFETCH] Images D${dept}_${year}..."
        (cd /workspace/images && python download_images.py $dept $year) > /tmp/prefetch_img_${dept}.log 2>&1 &
        PREFETCH_IMG_PID=$!
    fi

    if [ -d "${TOPO_BASE}/D${dept}_${TOPO_YEAR}" ]; then
        echo "[PREFETCH] Topo D${dept} deja presente, skip."
        PREFETCH_TOPO_PID=""
    else
        echo "[PREFETCH] Topo D${dept}..."
        (cd /workspace/topo && python download_topo.py $dept $TOPO_YEAR $TOPO_DATE) > /tmp/prefetch_topo_${dept}.log 2>&1 &
        PREFETCH_TOPO_PID=$!
    fi
}

wait_prefetch() {
    local dept=$1
    if [ -n "$PREFETCH_IMG_PID" ]; then
        echo "[PREFETCH] Attente fin telechargement D${dept}..."
        wait $PREFETCH_IMG_PID && echo "[PREFETCH] Images D${dept} OK" || { echo "[PREFETCH] ERREUR images D${dept} - voir /tmp/prefetch_img_${dept}.log"; echo "FAIL D${dept} - download images erreur - $(date)" >> $LOG_FILE; }
        wait $PREFETCH_TOPO_PID && echo "[PREFETCH] Topo D${dept} OK" || { echo "[PREFETCH] ERREUR topo D${dept} - voir /tmp/prefetch_topo_${dept}.log"; echo "FAIL D${dept} - download topo erreur - $(date)" >> $LOG_FILE; }
        PREFETCH_IMG_PID=""
        PREFETCH_TOPO_PID=""
    fi
}

cleanup() {
    local dept=$1
    local year=$2
    echo "[CLEANUP] Suppression D${dept}..."
    rm -rf "${IMAGES_BASE}/D${dept}_${year}" || true
    rm -rf "${TOPO_BASE}/D${dept}_${TOPO_YEAR}" || true
}

# -------- Pipeline --------
for i in "${!DEPTS_YEARS[@]}"; do
    PAIR=${DEPTS_YEARS[$i]}
    DEPT=${PAIR%%:*}
    YEAR=${PAIR##*:}

    NEXT_PAIR=${DEPTS_YEARS[$((i+1))]:-""}
    NEXT_DEPT=${NEXT_PAIR%%:*}
    NEXT_YEAR=${NEXT_PAIR##*:}

    echo "==============================="
    echo "[RUN] D${DEPT}_${YEAR}"
    echo "==============================="

    update_config $DEPT $YEAR

    if [ -n "$NEXT_DEPT" ]; then
        download_next $NEXT_DEPT $NEXT_YEAR
    fi

    cd /workspace/deeppvmapper
    if python main.py --dpt $((10#$DEPT)); then
        echo "[DONE] D${DEPT} termine."
        echo "OK   D${DEPT}_${YEAR} - $(date)" >> $LOG_FILE
    else
        echo "[ERREUR] main.py a echoue sur D${DEPT} - passage au suivant."
        echo "FAIL D${DEPT}_${YEAR} - main.py erreur - $(date)" >> $LOG_FILE
    fi

    wait_prefetch $NEXT_DEPT

    cleanup $DEPT $YEAR

done

echo "========================"
echo "Pipeline complet."
echo "========================"
echo "Pipeline termine le $(date)" >> $LOG_FILE