# Contributing to DeepPVMapper

Thanks for your interest in this project! There are two very different ways to contribute, depending on what you're into.

## 1. Correct or add PV installations on the map (no coding required)

The fastest way to improve the registry is via the [interactive map](https://gabrielkasmi.github.io/deeppvmapper/content/map.html):

- **Spot a false detection?** Click it and report it.
- **A PV system is missing?** Use "Add installation" and draw its outline.
- **An outline is wrong?** Redraw it.

Submissions are reviewed and merged into future releases of the registry. No setup, no code, just your local knowledge of rooftops.

## 2. Contribute to the pipeline (code)

Open an issue to discuss before submitting a large PR. Areas where help is especially welcome:

### Performance

- **Async tile prefetch** — classification reads tiles sequentially from disk; overlapping I/O with GPU inference (threading + Queue) would cut wall time on I/O-bound setups. Tile loading/decoding is currently the main bottleneck of the pipeline.
- **Multi-GPU segmentation** — `num_gpu > 1` via DataParallel is wired in but untested at scale.

### Imagery and models

- **Modular image loader** — preprocessing is hardcoded to IGN JP2 tiles; a pluggable loader interface would allow other aerial sources (SPOT, Sentinel-2, USGS, etc.) without touching the rest of the pipeline.
- **Additional model weights** — the classification and segmentation architectures are standard (InceptionV3, FCN/DeepLab); weights trained on other geographies or datasets can be dropped in via `config.yml` as long as they match the input/output format. Better-performing models would be a great addition to the project.

### Characteristics extraction

- **pypvroof coverage beyond France** — tilt estimation relies on a LUT built from French irradiance data; extending the LUT or adding a regression-based fallback would make the pipeline usable in other countries.
- **pypvroof refactor** — the library has known technical debt; a cleaner API and better test coverage would benefit both this pipeline and standalone users.

### Coverage

- **Overseas France (DOM-TOM)** — the pipeline assumes metropolitan France throughout (LAMB93 projection, department codes 01–95, metropolitan BDTOPO and commune shapefiles). Extending it to overseas departments (971–976) requires handling their local CRS (RGAF09, RGR92, etc.) and pointing to the right IGN/BDTOPO extracts. La Réunion is the suggested starting point.

### Building filter

- **Pluggable footprint sources** — the building filter is currently tied to BDTOPO; abstracting it behind a common interface would allow OpenStreetMap, Microsoft Building Footprints, or any polygon layer as a drop-in replacement.

## How to contribute code

1. Check the [open issues](https://github.com/gabrielkasmi/deeppvmapper/issues) — issues labelled `good first issue` are the best entry points — or open one to discuss your idea before starting work, so we can align on scope.
2. Fork the repo and create a branch for your change.
3. Set up following the README: `apt-get install gdal-bin libgdal-dev libopenjp2-7`, then `pip install "GDAL==$(gdal-config --version)" && pip install -r requirements.txt`. Model weights and runtime data are on [Zenodo](https://doi.org/10.5281/zenodo.7576814). GPU required (CUDA 11.8+, 8 GB VRAM minimum).
4. For local testing without a full department, use `tiles_list` in `config.yml` to run on a handful of tiles (see README → Usage).
5. Open a pull request with a clear description of what changed and why. Reference the related issue.

## Reporting bugs

Open an issue with: what you ran (exact command + config), what you expected, what happened instead, and your environment (OS, GPU, CUDA version).

## Questions

For anything else — data requests, feedback, collaboration ideas — [contact Gabriel](mailto:gabriel.kasmi.services@gmail.com).
