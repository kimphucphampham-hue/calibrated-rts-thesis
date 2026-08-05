import numpy as np
import pandas as pd

# 1. THIẾT LẬP THÔNG SỐ BAN ĐẦU VÀ TÍNH TÁI LẬP (REPRODUCIBILITY)
# Thiết lập seed để đảm bảo mỗi lần chạy đều ra cùng một kết quả dữ liệu
np.random.seed(42)

num_sprints = 50
cases_per_sprint = 100
total_samples = num_sprints * cases_per_sprint  # Đúng 5.000 mẫu theo chuyên đề

print(f"--- Bắt đầu quy trình sinh {total_samples} mẫu dữ liệu qua {num_sprints} Sprint ---")

# Tạo mảng danh sách các Sprint tăng dần theo thời gian (Tuân thủ chuỗi thời gian)
sprint_ids = np.repeat(np.arange(1, num_sprints + 1), cases_per_sprint)
# Tạo mã định danh kịch bản kiểm thử (Test Case ID) chạy trong mỗi Sprint
test_case_ids = np.tile(np.arange(1, cases_per_sprint + 1), num_sprints)


# 2. PHÂN PHỐI PARETO (ĐỊNH LUẬT 80/20) CHO ĐẶC TRƯNG "LỊCH SỬ LỖI"
# Giả định 20% số ca kiểm thử nằm trong module nhạy cảm (High-risk cluster) và chiếm 80% tỷ lệ lỗi
is_high_risk_cluster = np.random.binomial(1, 0.2, total_samples)

# Nếu thuộc cụm rủi ro cao, số lần fail trong quá khứ tuân theo phân phối Poisson biên độ lớn (Lam=6)
# Nếu thuộc cụm an toàn, số lần fail tuân theo phân phối Poisson biên độ nhỏ (Lam=0.5)
failure_history = np.where(
    is_high_risk_cluster == 1,
    np.random.poisson(lam=6, size=total_samples),
    np.random.poisson(lam=0.5, size=total_samples)
)


# 3. SINH CÁC ĐẶC TRƯNG TĨNH VÀ ĐỘNG THEO QUY LUẬT PHẦN MỀM (MỤC 3.2)
# Khối lượng mã thay đổi (Code Churn): Sinh bằng phân phối Poisson (Lam=45 dòng code)
code_churn = np.random.poisson(lam=45, size=total_samples)

# Độ phức tạp Cyclomatic: Mã nguồn thông thường có độ phức tạp trung bình là 6-8 nhánh rẽ
cyclomatic_complexity = np.random.poisson(lam=8, size=total_samples)

# Độ phụ thuộc (Dependencies): Số lượng module liên kết, sinh theo phân phối Poisson (Lam=3)
dependencies = np.random.poisson(lam=3, size=total_samples)

# Thời gian thực thi (ms): Sinh theo phân phối chuẩn (Normal Distribution) quanh mức 1500ms
execution_time = np.random.normal(loc=1500, scale=350, size=total_samples).astype(int)
execution_time = np.clip(execution_time, 100, None)  # Đảm bảo thời gian không bị âm

# Tuổi của ca kiểm thử (Test Age): Số chu kỳ chưa đổi, sinh ngẫu nhiên từ 1 đến 12 Sprint
test_age = np.random.randint(1, 13, size=total_samples)


# 4. CƠ CHẾ ÁNH XẠ BIẾN VÀ TÍNH ĐIỂM RỦI RO (HÀM LOGISTIC / SIGMOID)
# Định nghĩa các trọng số Beta thể hiện mối tương quan nhân - quả logic (Mục 3.1.2)
beta_churn = 0.015       # Code thay đổi càng nhiều -> Rủi ro càng tăng (Dương)
beta_complex = 0.12      # Độ phức tạp cao -> Rủi ro tăng mạnh (Dương)
beta_depend = 0.08       # Ít ảnh hưởng hơn nhưng vẫn làm tăng rủi ro (Dương)
beta_fail_hist = 0.45    # Lịch sử thường xuyên fail -> Khả năng cao lại tái diễn (Dương mạnh)
beta_age = -0.18         # Test case lâu đời không bắt được lỗi -> Thường dư thừa/an toàn (Âm)

# Tính giá trị Logit (z) dựa trên tổ hợp tuyến tính các đặc trưng
z_raw = (beta_churn * code_churn +
         beta_complex * cyclomatic_complexity +
         beta_depend * dependencies +
         beta_fail_hist * failure_history +
         beta_age * test_age)

# Tính xác suất lỗi P(Y=1) thông qua hàm kích hoạt Sigmoid
probabilities = 1 / (1 + np.exp(-z_raw))

# Để đảm bảo phân phối nhãn nền tảng đạt chính xác tỷ lệ 10/90 (Mục 3.1.1)
# Ta lấy ngưỡng quyết định (Threshold) tại phân vị thứ 90 (90th percentile) của xác suất
threshold_90 = np.percentile(probabilities, 90)
labels = np.where(probabilities >= threshold_90, 1, 0)


# 5. KỸ THUẬT TIÊM NHIỄU NGẪU NHIÊN GIẢ LẬP HÀNH VI CON NGƯỜI (NOISE INJECTION)
# Chủ động lấy ngẫu nhiên 3% số lượng mẫu để đảo nhãn (Flip Labels)
noise_rate = 0.03
num_noise_samples = int(total_samples * noise_rate)
noise_indices = np.random.choice(total_samples, size=num_noise_samples, replace=False)

# Tiến hành đảo nhãn: 1 thành 0, 0 thành 1 để phá vỡ ranh giới toán học cứng nhắc
labels[noise_indices] = 1 - labels[noise_indices]


# 6. ĐÓNG GÓI THÀNH DANH SÁCH CẤU TRÚC (DATAFRAME) VÀ KIỂM CHỨNG TRỰC QUAN
df_synthetic = pd.DataFrame({
    'Sprint_ID': sprint_ids,
    'TestCase_ID': test_case_ids,
    'Code_Churn': code_churn,
    'Cyclomatic_Complexity': cyclomatic_complexity,
    'Dependencies': dependencies,
    'Failure_History': failure_history,
    'Execution_Time_ms': execution_time,
    'Test_Age': test_age,
    'Label': labels
})

# Kiểm tra phân phối nhãn thực tế sau khi đã tiêm nhiễu
label_counts = df_synthetic['Label'].value_counts(normalize=True) * 100

print("\n--- Quy trình sinh dữ liệu hoàn tất thành công! ---")
print(f"Tổng số bản ghi thu được: {df_synthetic.shape[0]} dòng, {df_synthetic.shape[1]} cột.")
print(f"Tỷ lệ phân phối Nhãn thực tế trong Dataset:")
print(f"  + Nhãn 0 (Ca kiểm thử An toàn): {label_counts[0]:.2f}%")
print(f"  + Nhãn 1 (Ca kiểm thử Có lỗi): {label_counts[1]:.2f}%")

# Xuất thử 5 dòng đầu tiên để kiểm tra cấu trúc dữ liệu
print("\nXem trước 5 bản ghi đầu tiên của Hệ thống:")
print(df_synthetic.head())

# (Tùy chọn) Xuất dữ liệu ra file CSV để nạp vào bước huấn luyện mô hình ở mục sau
df_synthetic.to_csv('synthetic_regression_testing_data.csv', index=False)