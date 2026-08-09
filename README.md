# Calibrated Risk-Bounded Regression Test Selection

Repository phục vụ tái lập chuyên đề tốt nghiệp và triển khai các thí nghiệm pilot cho luận văn thạc sĩ.

## Repository structure

- `legacy/`: Source code chuyên đề tốt nghiệp
- `data/raw/synthetic/`: Dữ liệu giả lập
- `data/raw/tcp_ci/`: Dữ liệu TCP-CI không được lưu trên Git
- `configs/`: Cấu hình chạy thực nghiệm
- `docs/`: Tài liệu và data dictionary
- `results/`: Kết quả được sinh từ pipeline


## Validate the synthetic dataset

Activate the Python environment:

```bash
source .venv/bin/activate

## Reproduce the legacy graduation-project experiment

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Validate the frozen synthetic dataset:

```bash
python src/validate_data.py
```

Run both the original and corrected temporal splitting modes:

```bash
python legacy/reproduce_legacy.py
```

Run only the row-level legacy splitting mode:

```bash
python legacy/reproduce_legacy.py --mode original
```

Run only the Sprint-level corrected splitting mode:

```bash
python legacy/reproduce_legacy.py --mode corrected
```

Generated results are written to:

```text
results/legacy/reproduction/
```

The `original` mode reproduces the row-level
`TimeSeriesSplit` found in the available legacy source.

The `corrected` mode applies temporal splitting to unique
Sprint identifiers so that one Sprint cannot occur in both
training and testing data.


## Legacy reproduction report

The complete legacy reproduction report is available at:

- `reports/legacy_reproduction_report.md`