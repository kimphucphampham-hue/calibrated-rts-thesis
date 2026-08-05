"""Regenerate the legacy synthetic dataset.

This script reproduces the synthetic-data generation logic
used in the original graduation project.

The regenerated dataset is written to:

    data/interim/synthetic_regenerated.csv

The frozen historical dataset under data/raw/ is never
overwritten by this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# REPOSITORY PATHS
# ============================================================

# File hiện tại nằm tại:
#   <repository>/legacy/generate_synthetic_data.py
#
# parents[0] là thư mục legacy/
# parents[1] là thư mục gốc repository.
REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    REPO_ROOT
    / "data"
    / "interim"
    / "synthetic_regenerated.csv"
)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_synthetic_data() -> pd.DataFrame:
    """Generate the legacy synthetic regression-testing dataset."""

    # 1. THIẾT LẬP THÔNG SỐ VÀ TÍNH TÁI LẬP
    np.random.seed(42)

    num_sprints = 50
    cases_per_sprint = 100
    total_samples = (
        num_sprints
        * cases_per_sprint
    )

    print(
        "--- Bắt đầu quy trình sinh "
        f"{total_samples} mẫu dữ liệu "
        f"qua {num_sprints} Sprint ---"
    )

    sprint_ids = np.repeat(
        np.arange(
            1,
            num_sprints + 1,
        ),
        cases_per_sprint,
    )

    test_case_ids = np.tile(
        np.arange(
            1,
            cases_per_sprint + 1,
        ),
        num_sprints,
    )

    # 2. PHÂN PHỐI PARETO CHO FAILURE HISTORY
    is_high_risk_cluster = np.random.binomial(
        1,
        0.2,
        total_samples,
    )

    failure_history = np.where(
        is_high_risk_cluster == 1,
        np.random.poisson(
            lam=6,
            size=total_samples,
        ),
        np.random.poisson(
            lam=0.5,
            size=total_samples,
        ),
    )

    # 3. SINH CÁC ĐẶC TRƯNG
    code_churn = np.random.poisson(
        lam=45,
        size=total_samples,
    )

    cyclomatic_complexity = np.random.poisson(
        lam=8,
        size=total_samples,
    )

    dependencies = np.random.poisson(
        lam=3,
        size=total_samples,
    )

    execution_time = np.random.normal(
        loc=1500,
        scale=350,
        size=total_samples,
    ).astype(int)

    execution_time = np.clip(
        execution_time,
        100,
        None,
    )

    test_age = np.random.randint(
        1,
        13,
        size=total_samples,
    )

    # 4. TÍNH ĐIỂM RỦI RO
    beta_churn = 0.015
    beta_complex = 0.12
    beta_depend = 0.08
    beta_fail_hist = 0.45
    beta_age = -0.18

    z_raw = (
        beta_churn * code_churn
        + beta_complex * cyclomatic_complexity
        + beta_depend * dependencies
        + beta_fail_hist * failure_history
        + beta_age * test_age
    )

    probabilities = (
        1
        / (
            1
            + np.exp(-z_raw)
        )
    )

    # Tạo đúng 10% nhãn 1 trước khi tiêm nhiễu.
    threshold_90 = np.percentile(
        probabilities,
        90,
    )

    labels = np.where(
        probabilities >= threshold_90,
        1,
        0,
    )

    positive_before_noise = int(
        labels.sum()
    )

    print(
        "\nPhân phối trước khi tiêm nhiễu:"
    )
    print(
        "  + Nhãn 0:",
        total_samples - positive_before_noise,
    )
    print(
        "  + Nhãn 1:",
        positive_before_noise,
    )

    # 5. TIÊM NHIỄU BẰNG CÁCH ĐẢO 3% NHÃN
    noise_rate = 0.03

    num_noise_samples = int(
        total_samples
        * noise_rate
    )

    noise_indices = np.random.choice(
        total_samples,
        size=num_noise_samples,
        replace=False,
    )

    original_noise_labels = labels[
        noise_indices
    ].copy()

    zero_to_one_count = int(
        (
            original_noise_labels == 0
        ).sum()
    )

    one_to_zero_count = int(
        (
            original_noise_labels == 1
        ).sum()
    )

    labels[noise_indices] = (
        1
        - labels[noise_indices]
    )

    print(
        "\nKết quả tiêm nhiễu:"
    )
    print(
        "  + Tổng số nhãn bị đảo:",
        num_noise_samples,
    )
    print(
        "  + Nhãn 0 chuyển thành 1:",
        zero_to_one_count,
    )
    print(
        "  + Nhãn 1 chuyển thành 0:",
        one_to_zero_count,
    )

    # 6. ĐÓNG GÓI DATAFRAME
    df_synthetic = pd.DataFrame(
        {
            "Sprint_ID": sprint_ids,
            "TestCase_ID": test_case_ids,
            "Code_Churn": code_churn,
            "Cyclomatic_Complexity": (
                cyclomatic_complexity
            ),
            "Dependencies": dependencies,
            "Failure_History": failure_history,
            "Execution_Time_ms": execution_time,
            "Test_Age": test_age,
            "Label": labels,
        }
    )

    return df_synthetic


def main() -> int:
    """Generate and save the regenerated dataset."""

    df_synthetic = generate_synthetic_data()

    # Tạo thư mục data/interim nếu chưa tồn tại.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Chỉ ghi vào data/interim.
    # Không ghi đè frozen CSV trong data/raw.
    df_synthetic.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    label_counts = (
        df_synthetic["Label"]
        .value_counts()
        .sort_index()
    )

    label_rates = (
        df_synthetic["Label"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        "\n--- Quy trình sinh dữ liệu hoàn tất ---"
    )

    print(
        "Kích thước:",
        f"{df_synthetic.shape[0]} dòng x "
        f"{df_synthetic.shape[1]} cột",
    )

    print(
        "\nPhân phối nhãn cuối cùng:"
    )

    print(
        "  + Nhãn 0:",
        int(label_counts.get(0, 0)),
        f"({label_rates.get(0, 0):.2f}%)",
    )

    print(
        "  + Nhãn 1:",
        int(label_counts.get(1, 0)),
        f"({label_rates.get(1, 0):.2f}%)",
    )

    print(
        "\nDataset đã được ghi tại:"
    )
    print(
        OUTPUT_PATH
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())