# AGENTS.md — HiTMicTools

## Project Overview

**HiTMicTools** is a deep-learning-based toolkit for automated high-throughput microscopy analysis (Boeck Lab, University Hospital of Basel). It processes time-lapse microscopy images through a configurable pipeline: focus restoration → segmentation → classification → cell tracking.

- **Package**: `hitmictools` v0.5.12 — Python 3.9+, `src/` layout
- **Entry point**: `hitmictools` CLI → `src/HiTMicTools/main.py` → `cli.py`
- **Config**: YAML file consumed by `ConfReader` → attribute-accessible dict
- **Models**: bundled as ZIP (model collection) or loaded individually

---

## Architecture Map

```
CLI (cli.py)
  └─ build_and_run_pipeline (build_pipeline.py)
       ├─ ConfReader              → parse YAML config
       ├─ get_pipeline(name)      → pipeline registry (pipelines/__init__.py)
       ├─ pipeline.load_config_dict()
       ├─ pipeline.load_model_bundle() / load_model_fromdict()
       ├─ pipeline.load_tracker()    (if tracking: true)
       └─ pipeline.process_folder() / process_folder_parallel()
```

### Pipelines (`src/HiTMicTools/pipelines/`)

| Name                     | Class                       | Use case                                     |
| ------------------------ | --------------------------- | -------------------------------------------- |
| `ASCT_focusrestore`    | `ASCT_focusrestore.py`    | Main: focus restore + seg + classify + track |
| `ASCT_ImageProcessing` | `ASCT_ImageProcessing.py` | Single-frame static analysis                 |
| `ASCT_scsegm`          | `ASCT_scsegm.py`          | RF-DETR-Segm instance segmentation           |
| `ASCT_zaslavier`       | `ASCT_zaslavier.py`       | Multi-channel GFP + PI analysis              |
| `oof_detection`        | `oof_detection.py`        | Out-of-focus detection only                  |

All inherit from `base_pipeline.py` which handles: model loading, file discovery, worklist support, parallel processing (joblib).

### Model Components (`src/HiTMicTools/model_components/`)

- `FocusRestorer` — NAFNet sliding-window inference (BF + FL channels)
- `SegmentationModel` — MonaiUNet sliding-window segmentation
- `ScSegmenter` — RF-DETR-Segm with temporal buffering + cross-frame tile batching
- `OofDetector` — RF-DETR tiling-based out-of-focus detector
- `CellClassifier` — FlexResNet/ONNX batch ROI classifier
- `PIClassifier` — sklearn/ONNX PI channel classifier
- `ImageScaler` — Multi-method normalization (range01, zscore, combined, fixed_range)

### Key Supporting Modules

- `img_processing/img_processor.py` — `ImagePreprocessor`: alignment, BaSiC/standard BG correction, well detection
- `roianalysis/roi_analyser.py` — Connected components + ROI measurements (CPU or GPU via RAPIDS)
- `tracking/cell_tracker.py` — `CellTracker`: btrack Bayesian multi-object tracking
- `resource_management/reserveresource.py` — `ReserveResource`: cross-process GPU booking via shared files
- `resource_management/sysutils.py` — device detection, cache clearing, system info
- `model_bundler.py` — creates ZIP model collections
- `utils.py` — metadata reading, unit conversion, btrack check

### Model Architectures (`src/HiTMicTools/model_arch/`)

- `nafnet.py` — NAFNet (image restoration)
- `basemodel.py` — UNet (segmentation)
- `flexresnet.py` — FlexResNet (classification)

### Tests & Docs

- `tests/` — test suites for models, workflows
- `docs/` — readthedocs source
- `Makefile` — `make test`, `make test-coverage`, `make test-model`, `make test-workflow`

---

## Development Notes

- **Version bumps are mandatory**: every code-changing PR/update that users may install must increment `pyproject.toml` before merge. Use normal PEP 440 ordering with no leading-zero patch numbers (`0.5.4`, then `0.5.5`, never `0.5.04`). This prevents `pip install --upgrade` from silently keeping stale Scicore/user installs.
- **Dependency changes are conservative**: keep runtime dependencies bounded in `pyproject.toml`; pin known fragile packages exactly; update `constraints/` when a known-good Scicore/Windows environment changes. For code-only updates inside an existing working environment, prefer `pip install -e . --no-deps` or `pip install --force-reinstall --no-deps ...` so pip does not unexpectedly upgrade the scientific stack.
- **GPU handling**: `ReserveResource` context manager must be used for any GPU-intensive work in multi-process contexts to avoid OOM. Check `resource_management/reserveresource.py`.
- **torch.compile**: Configurable per model via metadata. See recent commits for implementation pattern.
- **Model bundles**: ZIP files with standardized naming + `config.yml` manifest. Use `model_bundler.py`.
- **Pipeline registration**: Add new pipelines in `pipelines/__init__.py` with `@register_pipeline` decorator and `required_models` list.
- **Parallelism**: `process_folder_parallel()` uses joblib. Each worker loads its own model copy — memory-expensive.
- **btrack**: Optional dependency. Always gate tracking code with `check_btrack()` from `utils.py`.
- **File formats**: `.nd2`, `.tiff`, `.p.tiff` (jetraw). Jetraw requires valid licence.
