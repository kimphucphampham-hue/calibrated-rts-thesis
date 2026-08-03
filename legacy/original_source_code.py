import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Yêu cầu cài đặt imbalanced-learn nếu chạy ở môi trường cục bộ (Colab đã có sẵn)
# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# --- GIAI ĐOẠN 1: CHUẨN BỊ ĐẦU VÀO LUỒNG DỮ LIỆU ---
# Giả định dữ liệu 'df_synthetic' từ bước trước đã sẵn sàng trong bộ nhớ
# Sắp xếp nghiêm ngặt theo trình tự thời gian tăng dần của các Sprint
df_synthetic = df_synthetic.sort_values(by='Sprint_ID').reset_index(drop=True)

# Tách biệt tập đặc trưng biến độc lập (X) và nhãn mục tiêu (y)
# Loại bỏ các trường định danh không mang giá trị học máy để tránh gây nhiễu hệ thống
X = df_synthetic.drop(columns=['Sprint_ID', 'TestCase_ID', 'Label'])
y = df_synthetic['Label']

# --- GIAI ĐOẠN 2: THIẾT LẬP CƠ CHẾ KIỂM CHỨNG CHUỖI THỜI GIAN ---
# Thiết lập không gian chia cửa sổ cuốn chiếu (ví dụ: 5 vòng lặp tịnh tiến thời gian)
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Khởi tạo cấu trúc lưu trữ kết quả đánh giá cho từng mô hình qua các vòng lặp
results = {
    'Random Forest': {'precision': [], 'recall': [], 'f1': []},
    'SVM': {'precision': [], 'recall': [], 'f1': []},
    'Gradient Boosting': {'precision': [], 'recall': [], 'f1': []}
}

print(f"--- Bắt đầu quy trình huấn luyện cuốn chiếu với {n_splits} vòng lặp tịnh tiến ---")

# --- GIAI ĐOẠN 3: VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ CUỐN CHIẾU (ROLLING WINDOW) ---
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"\n[Vòng lặp {fold + 1}/{n_splits}]")

    # Phân chia dữ liệu theo chỉ số thời gian (Chặn đứng hoàn toàn hiện tượng Data Leakage)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Xác định phạm vi Sprint đang thực hiện kiểm thử trong vòng lặp này
    start_sprint_test = df_synthetic.iloc[test_index]['Sprint_ID'].min()
    end_sprint_test = df_synthetic.iloc[test_index]['Sprint_ID'].max()
    print(f"  + Huấn luyện trên các mẫu quá khứ từ mốc xuất phát đến trước Sprint {start_sprint_test}")
    print(f"  + Dự đoán 'mù' trên không gian tương lai: Từ Sprint {start_sprint_test} đến Sprint {end_sprint_test}")

    # 1. Chuẩn hóa thang đo đặc trưng số học (Feature Scaling)
    scaler = StandardScaler()
    # Chỉ học tham số scale (fit) trên tập Train quá khứ, sau đó ép thang đo lên cả hai tập
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Xử lý mất cân bằng dữ liệu bằng kỹ thuật SMOTE
    # TUYỆT ĐỐI chỉ áp dụng trên tập Train để thuật toán không học vẹt. Tập Test giữ nguyên bản chất 10/90.
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # 3. Cấu hình và Khởi tạo 3 Thuật toán phân lớp cốt lõi
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # 4. Thực thi quy trình Học và Trích xuất chỉ số hiệu năng
    for model_name, model in models.items():
        # Gọi hàm fit() để mô hình học các quy luật nhân quả
        model.fit(X_train_resampled, y_train_resampled)

        # Mô hình tiến hành suy diễn dự đoán trên tập kiểm thử (Testing set) độc lập
        y_pred = model.predict(X_test_scaled)

        # Tính toán các độ đo hiệu năng của vòng lặp hiện tại
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Lưu trữ kết quả của thuật toán
        results[model_name]['precision'].append(prec)
        results[model_name]['recall'].append(rec)
        results[model_name]['f1'].append(f1)

        # =====================================================================
        # BƯỚC MÔ PHỎNG TRIỂN KHAI THỰC TẾ: LỌC TẬP T' VÀ LẤY LẠI TESTCASE_ID
        # =====================================================================

        # 1. Lấy lại bản sao của dữ liệu GỐC (chứa ID) tương ứng với tập Test hiện tại
        df_test_original = df_synthetic.iloc[test_index]

        # 2. Dùng mảng y_pred làm mặt nạ (mask) để lọc ra các dòng mà AI đoán là 1
        T_prime_df = df_test_original[y_pred == 1]

        # 3. Trích xuất đúng cột TestCase_ID thành một danh sách (list)
        danh_sach_chay_thuc_te = T_prime_df['TestCase_ID'].tolist()

        # Chỉ in kết quả trích xuất ID ở Vòng lặp cuối cùng (để màn hình đỡ rối)
        if fold == n_splits - 1:
            print(f"    [Triển khai CI/CD] {model_name} đề xuất tập T' gồm {len(danh_sach_chay_thuc_te)}/{len(y_test)} Test Case.")
            print(f"    -> Danh sách ID đẩy lên máy chủ: {danh_sach_chay_thuc_te[:15]}... (hiển thị 15 ID đầu)")

        # THÊM DÒNG NÀY VÀO ĐỂ XUẤT FILE CÓ ĐẦY ĐỦ TESTCASE_ID NHÉ BẠN:
            T_prime_df.to_csv(f'T_prime_{model_name}.csv', index=False)

        # =====================================================================


# --- GIAI ĐOẠN 4: TỔNG HỢP VÀ KẾT XUẤT BẢNG HIỆU NĂNG TRUNG BÌNH (CHƯƠNG 4) ---
print("\n" + "="*50)
print("BẢNG TỔNG HỢP HIỆU NĂNG MÔ HÌNH SAU KHI CHẠY TIME-SERIES SPLIT")
print("="*50)

summary_data = []
for model_name in results.keys():
    # Tính giá trị trung bình (Mean) của các độ đo qua các vòng lặp cuốn chiếu
    avg_precision = np.mean(results[model_name]['precision']) * 100
    avg_recall = np.mean(results[model_name]['recall']) * 100
    avg_f1 = np.mean(results[model_name]['f1']) * 100

    summary_data.append({
        'Thuật toán / Mô hình': model_name,
        'Độ chính xác (Precision)': f"{avg_precision:.2f}%",
        'Độ thu hồi (Recall)': f"{avg_recall:.2f}%",
        'Điểm F1-Score': f"{avg_f1:.2f}%"
    })

# Chuyển đổi cấu trúc sang dạng bảng biểu hiển thị trực quan
df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("="*50)