# BÁO CÁO TÁI LẬP CHUYÊN ĐỀ TỐT NGHIỆP

## 1. Thông tin chung

* **Tên artifact:** Legacy Graduation Project Reproduction
* **Repository:** `calibrated-rts-thesis`
* **Dữ liệu đầu vào:** `data/raw/synthetic/synthetic_data.csv`
* **Source code gốc:** `legacy/original_source_code.py`
* **Script tái lập:** `legacy/reproduce_legacy.py`
* **File cấu hình:** `configs/legacy.yaml`
* **Ngày thực hiện:** 2026-08-05
* **Người thực hiện:** Phạm Thị Kim Phúc
* **Commit:** [[Git commit hash]]

## 2. Mục tiêu tái lập

Mục tiêu của hoạt động này là đóng gói lại chuyên đề tốt nghiệp thành một repository có khả năng chạy độc lập, kiểm tra được và tái lập được. Quá trình tái lập tập trung vào bốn mục tiêu:

1. Xác minh đặc điểm thực tế của bộ dữ liệu giả lập.
2. Chạy lại ba mô hình Random Forest, Support Vector Machine và Gradient Boosting.
3. So sánh kết quả tái lập với các chỉ số được công bố trong báo cáo chuyên đề.
4. Kiểm tra và sửa cách chia dữ liệu theo thời gian để một Sprint không xuất hiện đồng thời trong tập huấn luyện và tập kiểm tra.

Hoạt động này chỉ có mục đích tái lập và kiểm tra proof-of-concept cũ. Dữ liệu giả lập không được sử dụng làm bằng chứng chính cho thực nghiệm luận văn thạc sĩ.

## 3. Artifact đầu vào

Các artifact được sử dụng gồm:

| Artifact                | Đường dẫn                               | Vai trò                                       |
| ----------------------- | --------------------------------------- | --------------------------------------------- |
| Báo cáo chuyên đề       | `docs/previous_graduation_report.docx`  | Nguồn mô tả phương pháp và kết quả đã công bố |
| Source code gốc         | `legacy/original_source_code.py`        | Nguồn xác định hành vi triển khai ban đầu     |
| Dữ liệu giả lập         | `data/raw/synthetic/synthetic_data.csv` | Frozen dataset dùng để tái lập                |
| Script kiểm tra dữ liệu | `src/validate_data.py`                  | Xác minh schema và chất lượng dữ liệu         |
| Script tái lập          | `legacy/reproduce_legacy.py`            | Chạy lại protocol original và corrected       |
| Cấu hình                | `configs/legacy.yaml`                   | Lưu seed, số fold, model và đường dẫn output  |
| Dependency lock         | `requirements-lock.txt`                 | Ghi lại phiên bản thư viện                    |

Source code gốc không phải là một chương trình độc lập hoàn toàn vì nó giả định biến `df_synthetic` đã được tạo và tồn tại trong bộ nhớ. Script tái lập mới khắc phục điểm này bằng cách đọc trực tiếp file CSV từ đường dẫn được khai báo trong cấu hình.

## 4. Môi trường thực nghiệm

Thực nghiệm được thực hiện trong Python virtual environment `.venv`.

| Thành phần       | Phiên bản            |
| ---------------- | -------------------- |
| Python           | 3.9.6   |
| NumPy            | 2.0.2    |
| Pandas           | 2.3.3   |
| scikit-learn     | 1.6.1  |
| imbalanced-learn | 0.12.4 |
| PyYAML           | 6.0.3   |
| Hệ điều hành     | macOS-15.7.2         |

Thông tin đầy đủ của môi trường được lưu tại:

```text
results/legacy/reproduction/run_metadata.json
```

## 5. Kiểm tra frozen dataset

Script sau được dùng để kiểm tra dữ liệu:

```bash
python src/validate_data.py
```

Kết quả kiểm tra:

| Thuộc tính            |                              Giá trị |
| --------------------- | -----------------------------------: |
| Trạng thái            | PASS_WITH_WARNINGS |
| Số dòng               | 500 |
| Số cột                | 9 |
| Số Sprint             | 50 |
| Số test case duy nhất | 100 |
| Số missing value      | 0 |
| Số dòng trùng         | 0 |
| Số mẫu nhãn 0         | 4386 |
| Số mẫu nhãn 1         | 614 |
| Tỷ lệ nhãn 1          | 12.28% |

Báo cáo chuyên đề mô tả bộ dữ liệu gồm 5.000 mẫu trải qua 50 Sprint và phân phối nhãn xấp xỉ 10% nhãn 1, 90% nhãn 0. Frozen CSV được kiểm tra trực tiếp thay vì mặc định xem các mô tả này là giá trị thực tế.

### 5.1. Kết quả kiểm tra schema

Bộ dữ liệu chứa các trường:

* `Sprint_ID`
* `TestCase_ID`
* `Code_Churn`
* `Cyclomatic_Complexity`
* `Dependencies`
* `Failure_History`
* `Execution_Time_ms`
* `Test_Age`
* `Label`

Kết quả kiểm tra cho thấy frozen dataset có đầy đủ 9 cột theo schema dự kiến, gồm `Sprint_ID`, `TestCase_ID`, `Code_Churn`, `Cyclomatic_Complexity`, `Dependencies`, `Failure_History`, `Execution_Time_ms`, `Test_Age` và `Label`. Không phát hiện cột bắt buộc bị thiếu hoặc cột ngoài schema.

Toàn bộ 5.000 dòng dữ liệu không chứa giá trị thiếu và không có dòng trùng lặp hoàn toàn. Không phát hiện cặp `Sprint_ID`–`TestCase_ID` bị lặp trong cùng một Sprint, nghĩa là mỗi test case chỉ xuất hiện một lần trong một chu kỳ Sprint.

Các cột được kỳ vọng là dữ liệu số đều chứa giá trị số hợp lệ và không có giá trị vô hạn. Cột `Label` chỉ chứa hai giá trị nhị phân `0` và `1`; không phát hiện nhãn ngoài miền giá trị cho phép.

Toàn bộ giá trị trong cột `Execution_Time_ms` đều lớn hơn 0. Các cột `Code_Churn`, `Dependencies`, `Failure_History` và `Test_Age` không chứa giá trị âm. Các giá trị trong cột `Cyclomatic_Complexity` cũng đều lớn hơn 0. Ngoài ra, các giá trị `Sprint_ID` và `TestCase_ID` đều là số dương.

Như vậy, frozen dataset đạt các yêu cầu cơ bản về schema, tính đầy đủ, tính duy nhất và miền giá trị. Không phát hiện lỗi chất lượng dữ liệu nghiêm trọng có thể ngăn cản quá trình tái lập mô hình. Trạng thái `PASS_WITH_WARNINGS` của bước kiểm tra không xuất phát từ lỗi schema mà chủ yếu do tỷ lệ nhãn 1 thực tế khác với tỷ lệ xấp xỉ 10% được mô tả trong báo cáo chuyên đề.


### 5.2. Sai khác giữa tài liệu và frozen artifact

Tỷ lệ nhãn 1 quan sát được là 12,28%, cao hơn mức khoảng 10% được mô tả trong báo cáo.

Sai khác này được ghi nhận như một thuộc tính của frozen artifact. Không thực hiện chỉnh sửa hoặc tái cân bằng dữ liệu gốc để ép dữ liệu khớp với mô tả trong báo cáo.

## 6. Cấu hình mô hình

Ba mô hình được tái lập:

| Mô hình           | Cấu hình chính                                                   |
| ----------------- | ---------------------------------------------------------------- |
| Random Forest     | `n_estimators=100`, `class_weight="balanced"`, `random_state=42` |
| SVM               | `kernel="rbf"`, `class_weight="balanced"`, `random_state=42`     |
| Gradient Boosting | `n_estimators=100`, `random_state=42`                            |

Các feature được sử dụng:

```text
Code_Churn
Cyclomatic_Complexity
Dependencies
Failure_History
Execution_Time_ms
Test_Age
```

Các trường định danh `Sprint_ID`, `TestCase_ID` và nhãn `Label` không được sử dụng làm feature.

`StandardScaler` chỉ được fit trên dữ liệu huấn luyện. Tập kiểm tra chỉ được transform bằng scaler đã fit từ tập train. SMOTE cũng chỉ được áp dụng trên tập huấn luyện sau khi chuẩn hóa.

## 7. Các protocol đánh giá

### 7.1. Protocol `original`

Protocol `original` tái hiện hành vi của source code cũ:

1. Sắp xếp toàn bộ DataFrame theo `Sprint_ID`.
2. Gọi `TimeSeriesSplit(n_splits=5)` trực tiếp trên các dòng dữ liệu.
3. Fit scaler trên tập train.
4. Áp dụng SMOTE trên tập train.
5. Huấn luyện và đánh giá ba mô hình.

Do đơn vị được chia là từng dòng, ranh giới fold có thể cắt ngang một Sprint. Khi điều này xảy ra, cùng một `Sprint_ID` xuất hiện trong cả train và test.

Kết quả ranh giới temporal:

| Fold | Sprint train | Sprint test | Sprint trùng |
| ---: | ------------ | ----------- | ------------ |
|    1 | 1-9     | 9-17    | 9    |
|    2 | 1-17      | 17-26    | 17      |
|    3 | 1-26     | 26-34    | 26      |
|    4 | 1-34     | 34-42    | 34     |
|    5 | 1-42      | 42-50    | 42      |

Kết quả cho thấy protocol original có Sprint overlap. Vì có overlap, protocol này không bảo đảm tính độc lập hoàn toàn giữa dữ liệu quá khứ và dữ liệu tương lai ở cấp Sprint.

### 7.2. Protocol `corrected`

Protocol corrected thay đổi đơn vị temporal split từ dòng dữ liệu thành Sprint:

1. Lấy danh sách `Sprint_ID` duy nhất.
2. Sắp xếp danh sách Sprint tăng dần.
3. Gọi `TimeSeriesSplit` trên danh sách Sprint.
4. Ánh xạ Sprint train và Sprint test về toàn bộ các dòng tương ứng.
5. Kiểm tra hai tập Sprint không giao nhau.
6. Kiểm tra Sprint cuối của train nhỏ hơn Sprint đầu của test.

Kết quả ranh giới temporal:

| Fold | Sprint train | Sprint test | Sprint trùng |
| ---: | ------------ | ----------- | ------------ |
|    1 | 1–10    | 11–18   | Không        |
|    2 | 1–18     | 19–26   | Không        |
|    3 | 1–26     | 27–34   | Không        |
|    4 | 1–34    | 35–42   | Không        |
|    5 | 1–42    | 43–50   | Không        |

Ở protocol corrected, toàn bộ các dòng của một Sprint được giữ trong cùng một partition. Không có Sprint nào xuất hiện đồng thời trong train và test.

### 7.3. Phân biệt với protocol được mô tả trong tài liệu

Báo cáo chuyên đề mô tả một ví dụ huấn luyện trên Sprint 1–40 và kiểm tra trên Sprint 41–50. Trong khi đó:

* Source code gốc sử dụng năm fold theo từng dòng.
* Protocol corrected sử dụng năm fold expanding window theo Sprint.
* Protocol 1–40/41–50 là một temporal holdout duy nhất.

Vì vậy, kết quả corrected không được mô tả là sự tái lập chính xác protocol holdout trong báo cáo. Nó là phiên bản sửa của phương pháp năm fold hiện diện trong source code.

## 8. Kết quả tái lập

### 8.1. Kết quả được công bố trong báo cáo cũ

| Mô hình           | Precision | Recall |     F1 |
| ----------------- | --------: | -----: | -----: |
| Random Forest     |    83,61% | 74,11% | 78,57% |
| SVM               |    74,09% | 76,09% | 74,99% |
| Gradient Boosting |    79,81% | 74,75% | 77,18% |

### 8.2. Kết quả protocol `original`

| Mô hình           | Precision trung bình | Recall trung bình | F1 trung bình | Độ lệch chuẩn F1 | Selection Ratio | Runtime Ratio |
| ----------------- | -------------------: | ----------------: | ------------: | ---------------: | --------------: | ------------: |
| Random Forest     |              83,62% |           73,11% |       78,01% |          4,13% |         10,73% |       10,83% |
| SVM               |              77,27% |           76,3% |       76,76% |          3,19% |         12,1% |       12,12% |
| Gradient Boosting |              81,56% |           75,17% |       78,18% |          3,44% |         11,28% |       11,33% |

### 8.3. Kết quả protocol `corrected`

| Mô hình           | Precision trung bình | Recall trung bình | F1 trung bình | Độ lệch chuẩn F1 | Selection Ratio | Runtime Ratio |
| ----------------- | -------------------: | ----------------: | ------------: | ---------------: | --------------: | ------------: |
| Random Forest     |              83.16% |           71,81% |       77,04% |          5% |         10,65% |       10,64% |
| SVM               |              77,18% |           75,94% |       76,53% |          3,72% |         12,1% |       12,08% |
| Gradient Boosting |              80,63% |           75,45% |       77,95% |          4,15% |         11,52% |       11,54% |

## 9. So sánh với kết quả được công bố

### 9.1. Protocol original so với báo cáo

| Mô hình           | Chênh lệch Precision | Chênh lệch Recall | Chênh lệch F1 |
| ----------------- | -------------------: | ----------------: | ------------: |
| Random Forest     |           0,01 điểm phần trăm |        -0,99 điểm phần trăm |    -0,55 điểm phần trăm |
| SVM               |           3,18 điểm phần trăm |        0,21 điểm phần trăm |    1,77 điểm phần trăm |
| Gradient Boosting |           1,75 điểm phần trăm |        0,42 điểm phần trăm |    1,00 điểm phần trăm |

Kết quả so sánh cho thấy protocol `original` tái tạo tương đối gần các chỉ số đã được công bố trong báo cáo chuyên đề, tuy nhiên chưa đạt mức tái lập chính xác hoàn toàn.

Random Forest có mức sai lệch thấp nhất. Precision chỉ chênh lệch 0,01 điểm phần trăm, trong khi Recall và F1 lần lượt chênh lệch 0,99 và 0,55 điểm phần trăm. Điều này cho thấy kết quả chạy lại của Random Forest gần như nhất quán với số liệu được công bố.

Đối với SVM, Recall chỉ chênh lệch 0,21 điểm phần trăm, nhưng Precision chênh lệch 3,18 điểm phần trăm và F1 chênh lệch 1,77 điểm phần trăm. Đây là mô hình có mức sai lệch Precision và F1 lớn nhất trong ba mô hình. Kết quả này cho thấy khả năng nhận diện các mẫu nhãn 1 của SVM tương đối ổn định về Recall, nhưng tỷ lệ dự đoán dương tính chính xác có sự khác biệt đáng kể hơn so với báo cáo cũ.

Gradient Boosting có mức chênh lệch trung bình, với Precision chênh lệch 1,75 điểm phần trăm, Recall chênh lệch 0,42 điểm phần trăm và F1 chênh lệch 1,00 điểm phần trăm. Các sai lệch này nhỏ hơn SVM nhưng lớn hơn Random Forest.

Nhìn chung, Recall của cả ba mô hình đều có mức chênh lệch dưới 1 điểm phần trăm. Điều này cho thấy khả năng phát hiện các test case mang nhãn 1 được tái tạo khá ổn định. Precision có mức biến động lớn hơn, đặc biệt đối với SVM và Gradient Boosting.

Các kết quả trên cho phép kết luận rằng protocol `original` đã tái lập gần đúng xu hướng và hiệu năng của ba mô hình trong báo cáo chuyên đề. Tuy nhiên, do vẫn tồn tại sai lệch về metric, khác biệt về phiên bản thư viện và sự thiếu nhất quán giữa protocol được mô tả trong báo cáo với source code gốc, kết quả này nên được phân loại là **tái lập gần đúng**, không phải **tái lập chính xác hoàn toàn**.


### 9.2. Protocol corrected so với original

| Mô hình           | F1 original | F1 corrected | Chênh lệch |
| ----------------- | ----------: | -----------: | ---------: |
| Random Forest     |     78,01% |      77,04% | -0,96 điểm phần trăm |
| SVM               |     76,76% |      76,53% | -0,23 điểm phần trăm |
| Gradient Boosting |     78,18% |      77,95% | -0,22 điểm phần trăm |

Kết quả so sánh cho thấy F1 của cả ba mô hình đều giảm nhẹ sau khi thay protocol chia theo từng dòng bằng protocol chia theo `Sprint_ID`. Random Forest có mức giảm lớn nhất, từ 78,01% xuống 77,04%, tương ứng giảm khoảng 0,97 điểm phần trăm. SVM giảm từ 76,76% xuống 76,53%, trong khi Gradient Boosting giảm từ 78,18% xuống 77,95%; cả hai mô hình đều chỉ giảm khoảng 0,23 điểm phần trăm.

Mức giảm F1 của protocol `corrected` là tương đối nhỏ, dưới 1 điểm phần trăm đối với cả ba mô hình. Điều này cho thấy hiệu năng tổng thể của các mô hình tương đối ổn định khi thay đổi đơn vị chia dữ liệu từ từng dòng sang từng Sprint. Gradient Boosting vẫn đạt F1 cao nhất trong protocol corrected, tiếp theo là Random Forest và SVM.

Việc F1 giảm không đồng nghĩa protocol `corrected` kém hơn protocol `original`. Protocol original có thể cắt ngang ranh giới của một Sprint, khiến cùng một `Sprint_ID` xuất hiện trong cả tập huấn luyện và tập kiểm tra. Điều này có thể tạo ra lợi thế đánh giá không mong muốn do dữ liệu train và test thuộc cùng một chu kỳ phát triển. Ngược lại, protocol corrected giữ toàn bộ dữ liệu của một Sprint trong cùng một partition và bảo đảm các Sprint dùng để huấn luyện xảy ra trước các Sprint dùng để kiểm tra.

Do đó, sự suy giảm nhỏ của F1 có thể được xem là chi phí hợp lý để đổi lấy một quy trình đánh giá nghiêm ngặt và đáng tin cậy hơn về mặt thời gian. Kết quả của protocol corrected phản ánh sát hơn khả năng của mô hình khi dự đoán trên các Sprint tương lai chưa từng được quan sát. Vì vậy, protocol corrected nên được ưu tiên cho các phân tích tiếp theo, mặc dù metric của nó thấp hơn nhẹ so với protocol original.


## 10. Các điểm không nhất quán được phát hiện

Quá trình tái lập phát hiện các điểm sau:

### 10.1. Source code không tự tải dữ liệu

Source code gốc giả định `df_synthetic` đã có trong bộ nhớ. Vì vậy, chỉ riêng file source không đủ để tái chạy trong một môi trường mới.

Script mới bổ sung bước đọc frozen CSV từ đường dẫn cấu hình.

### 10.2. Đơn vị chia temporal chưa đúng ở cấp Sprint

Source gốc gọi `TimeSeriesSplit` trên từng dòng. Cách này có thể khiến cùng một Sprint xuất hiện ở cả train và test.

Script corrected gọi `TimeSeriesSplit` trên danh sách Sprint duy nhất và giữ toàn bộ các dòng của một Sprint trong cùng một partition.

### 10.3. Khác biệt giữa protocol trong báo cáo và source

Báo cáo mô tả ví dụ Sprint 1–40 dùng để train và Sprint 41–50 dùng để test. Source lại triển khai năm fold expanding window theo từng dòng.

Do đó, không thể khẳng định source hiện có là implementation chính xác của protocol được mô tả trong tài liệu.

### 10.4. Khác biệt về phân phối nhãn

Báo cáo chuyên đề mô tả bộ dữ liệu giả lập có phân phối xấp xỉ 10% nhãn 1 và 90% nhãn 0. Kết quả kiểm tra frozen dataset cho thấy trong tổng số 5.000 mẫu có 614 mẫu mang nhãn 1 và 4.386 mẫu mang nhãn 0, tương ứng với tỷ lệ 12,28% nhãn 1 và 87,72% nhãn 0.

Việc kiểm tra source code sinh dữ liệu cho thấy sự khác biệt này xuất phát trực tiếp từ bước tiêm nhiễu nhãn. Trước khi tiêm nhiễu, code sử dụng phân vị thứ 90 của xác suất rủi ro làm ngưỡng quyết định. Kết quả của bước này tạo ra đúng 500 mẫu nhãn 1 và 4.500 mẫu nhãn 0, tương ứng với phân phối 10%–90%.

Sau đó, code chọn ngẫu nhiên 3% trong tổng số 5.000 mẫu, tương đương 150 mẫu, và đảo giá trị nhãn của các mẫu được chọn. Với seed cố định bằng 42, tập mẫu được chọn để đảo nhãn gồm 132 mẫu ban đầu mang nhãn 0 và 18 mẫu ban đầu mang nhãn 1. Sau khi đảo nhãn, 132 mẫu được chuyển từ nhãn 0 sang nhãn 1, trong khi 18 mẫu được chuyển từ nhãn 1 sang nhãn 0.

Do đó, số lượng nhãn 1 cuối cùng được tính như sau:

```text
500 + 132 - 18 = 614 mẫu
```

Số lượng nhãn 0 cuối cùng là:

```text
4.500 - 132 + 18 = 4.386 mẫu
```

Như vậy, tỷ lệ 12,28% nhãn 1 trong frozen dataset là kết quả xác định và có thể tái lập của source code sinh dữ liệu khi sử dụng `np.random.seed(42)`. Đây không phải là lỗi của file CSV và cũng không phải bằng chứng cho thấy frozen dataset được tạo từ một phiên bản source khác.

Nguyên nhân về mặt phương pháp là thao tác đảo nhãn được thực hiện bằng cách lấy mẫu ngẫu nhiên trên toàn bộ dữ liệu mà không bảo toàn phân phối lớp. Do nhãn 0 chiếm 90% dữ liệu ban đầu, phần lớn các mẫu được chọn để đảo nhãn cũng thuộc nhãn 0. Vì vậy, số mẫu chuyển từ nhãn 0 sang nhãn 1 lớn hơn đáng kể số mẫu chuyển theo chiều ngược lại, làm tỷ lệ nhãn 1 tăng từ 10% lên 12,28%.

Do đó, phát biểu “bộ dữ liệu đạt chính xác tỷ lệ 10/90” chỉ đúng tại thời điểm trước khi tiêm nhiễu. Sau bước đảo nhãn ngẫu nhiên 3%, phân phối cuối cùng không còn giữ nguyên tỷ lệ 10/90. Trong quá trình tái lập, frozen dataset được giữ nguyên với phân phối thực tế 12,28%–87,72% để bảo đảm kết quả phù hợp với artifact ban đầu và có thể tái lập đầy đủ.


### 10.5. Không nhất quán về Recall của SVM

Phần tóm tắt của báo cáo đề cập Recall của SVM là 88%, trong khi Bảng 4.1 ghi Recall là 76,09%. Báo cáo tái lập sử dụng Bảng 4.1 làm nguồn tham chiếu định lượng vì bảng này cung cấp đầy đủ Precision, Recall và F1 cho cả ba mô hình.

## 11. Đánh giá khả năng tái lập

## 11. Đánh giá khả năng tái lập

Kết quả tái lập được phân loại như sau:

* **Khả năng chạy độc lập:** Đạt
* **Khả năng tái tạo preprocessing:** Đạt
* **Khả năng tái tạo model configuration:** Đạt
* **Khả năng tái tạo chính xác metric:** Đạt một phần
* **Khả năng kiểm tra temporal split:** Đạt
* **Khả năng ghi lại môi trường:** Đạt

### 11.1. Khả năng chạy độc lập

Repository có thể chạy độc lập từ thư mục gốc mà không yêu cầu người dùng phải tạo trước biến `df_synthetic` trong bộ nhớ. Script tái lập đọc trực tiếp frozen dataset từ file CSV, đọc tham số thực nghiệm từ file cấu hình YAML và tự động tạo các file kết quả trong thư mục output.

Quy trình tái lập có thể được thực hiện bằng các lệnh:

```bash
python src/validate_data.py
python legacy/reproduce_legacy.py --mode both
```

Do đó, khả năng chạy độc lập được đánh giá là **Đạt**, với điều kiện người dùng đã cài đặt đầy đủ các thư viện trong `requirements-lock.txt` và frozen dataset vẫn được lưu đúng tại đường dẫn đã cấu hình.

### 11.2. Khả năng tái tạo preprocessing

Các bước preprocessing chính của source code chuyên đề đã được tái tạo, bao gồm:

1. Sắp xếp dữ liệu theo `Sprint_ID`.
2. Loại bỏ `Sprint_ID`, `TestCase_ID` và `Label` khỏi tập feature.
3. Chuẩn hóa dữ liệu bằng `StandardScaler`.
4. Chỉ fit scaler trên tập huấn luyện.
5. Sử dụng scaler đã fit để transform tập kiểm tra.
6. Chỉ áp dụng SMOTE trên tập huấn luyện.
7. Không sử dụng dữ liệu kiểm tra trong quá trình chuẩn hóa hoặc lấy mẫu lại.

Các bước này phù hợp với logic được triển khai trong source code cũ. Vì vậy, khả năng tái tạo preprocessing được đánh giá là **Đạt**.

Tuy nhiên, đánh giá này chỉ áp dụng cho pipeline huấn luyện mô hình. Source code sinh dữ liệu giả lập được xem là một artifact đầu vào riêng và không thuộc trực tiếp bước preprocessing của từng fold.

### 11.3. Khả năng tái tạo model configuration

Repository đã tái tạo ba họ mô hình được sử dụng trong chuyên đề:

* Random Forest với 100 cây quyết định và `class_weight="balanced"`.
* SVM với kernel RBF và `class_weight="balanced"`.
* Gradient Boosting với 100 vòng lặp học.

Random seed được cố định để giảm biến động giữa các lần chạy. Các tham số chính được lưu trong file cấu hình thay vì khai báo phân tán trong source code.

Do đó, khả năng tái tạo model configuration được đánh giá là **Đạt**.

Đánh giá này áp dụng cho các tham số được thể hiện trong source code và tài liệu hiện có. Những tham số sử dụng giá trị mặc định của thư viện nhưng không được ghi rõ trong báo cáo cũ có thể thay đổi giữa các phiên bản `scikit-learn`.

### 11.4. Khả năng tái tạo chính xác metric

Kết quả của protocol `original` gần với các chỉ số được công bố nhưng không trùng khớp hoàn toàn.

Chênh lệch F1 so với báo cáo là:

* Random Forest: giảm 0,55 điểm phần trăm.
* SVM: tăng 1,77 điểm phần trăm.
* Gradient Boosting: tăng 1,00 điểm phần trăm.

Chênh lệch F1 của cả ba mô hình đều dưới 2 điểm phần trăm. Điều này cho thấy repository đã tái tạo được xu hướng và mức hiệu năng tổng thể của các mô hình. Tuy nhiên, do kết quả không hoàn toàn giống các giá trị trong báo cáo, khả năng tái tạo chính xác metric được đánh giá là **Đạt một phần**.

Các nguyên nhân có thể dẫn đến sai lệch gồm:

* Phiên bản Python và thư viện của lần chạy ban đầu không được ghi lại.
* Một số tham số mặc định của thư viện có thể khác giữa các phiên bản.
* Báo cáo cũ không cung cấp metric chi tiết của từng fold.
* Protocol được mô tả trong báo cáo không hoàn toàn thống nhất với protocol được triển khai trong source code.
* Không có log thực nghiệm ban đầu để đối chiếu từng bước.

Vì vậy, kết quả hiện tại được phân loại là **tái lập gần đúng**, không phải tái lập chính xác tuyệt đối.

### 11.5. Khả năng kiểm tra temporal split

Repository cho phép chạy và đối chiếu hai protocol:

* `original`: áp dụng `TimeSeriesSplit` trực tiếp trên từng dòng dữ liệu.
* `corrected`: áp dụng `TimeSeriesSplit` trên danh sách `Sprint_ID` duy nhất.

Các file kết quả lưu lại phạm vi Sprint của train và test trong từng fold, đồng thời ghi rõ các Sprint xuất hiện ở cả hai partition.

Kết quả kiểm tra cho thấy protocol original có thể cắt ngang một Sprint, trong khi protocol corrected bảo đảm:

* Một Sprint không xuất hiện đồng thời trong train và test.
* Toàn bộ Sprint trong tập train xảy ra trước các Sprint trong tập test.
* Toàn bộ các dòng của cùng một Sprint được giữ trong cùng một partition.

Do đó, khả năng kiểm tra temporal split được đánh giá là **Đạt**.

### 11.6. Khả năng ghi lại môi trường

Repository tạo file:

```text
results/legacy/reproduction/run_metadata.json
```

File này ghi lại các thông tin của lần chạy tái lập, bao gồm:

* Phiên bản Python.
* Hệ điều hành.
* Phiên bản NumPy.
* Phiên bản Pandas.
* Phiên bản scikit-learn.
* Phiên bản imbalanced-learn.
* Phiên bản PyYAML.
* Đường dẫn cấu hình.
* Đường dẫn dữ liệu.
* Protocol được thực thi.

Ngoài ra, các phiên bản dependency của môi trường hiện tại được lưu trong `requirements-lock.txt`. Vì vậy, khả năng ghi lại môi trường tái lập hiện tại được đánh giá là **Đạt**.

Tuy nhiên, môi trường của thực nghiệm ban đầu trong chuyên đề không được ghi lại đầy đủ. Do đó, repository có thể xác định và tái tạo môi trường hiện tại nhưng chưa thể khôi phục chính xác môi trường lịch sử đã tạo ra các metric được công bố.

### 11.7. Nhận xét tổng hợp

Repository đã tái tạo thành công pipeline chính của chuyên đề tốt nghiệp dưới dạng một artifact có thể chạy độc lập, bao gồm kiểm tra dữ liệu, tiền xử lý, chia dữ liệu theo thời gian, huấn luyện ba mô hình, tính metric, lưu kết quả từng fold và ghi lại môi trường thực thi.

Source code sinh dữ liệu giả lập ban đầu cũng giải thích được phân phối nhãn của frozen dataset. Trước bước tiêm nhiễu, dữ liệu có đúng 10% nhãn 1 và 90% nhãn 0. Sau khi đảo ngẫu nhiên 3% số nhãn với seed bằng 42, phân phối cuối cùng trở thành 12,28% nhãn 1 và 87,72% nhãn 0. Vì vậy, sự khác biệt về phân phối nhãn đã được giải thích bằng source code và không còn được xem là một nguyên nhân chưa xác định.

Kết quả protocol original gần với các metric trong báo cáo cũ, nhưng vẫn tồn tại sai lệch nhỏ. Vì vậy, repository đạt mức **tái lập gần đúng** về kết quả định lượng.

Ngoài ra, quá trình tái lập phát hiện sự không thống nhất giữa tài liệu và source code về cách chia dữ liệu theo thời gian. Source code cũ sử dụng năm fold trên từng dòng dữ liệu, trong khi báo cáo mô tả việc huấn luyện trên Sprint 1–40 và kiểm tra trên Sprint 41–50. Protocol corrected đã khắc phục vấn đề Sprint overlap nhưng được xem là một phiên bản sửa về phương pháp luận, không phải là kết quả nguyên bản của chuyên đề.

Tổng thể, repository đáp ứng yêu cầu của một gói tái lập ở mức kỹ thuật: có thể chạy lại, kiểm tra được, lưu lại output và giải thích được các sai khác chính. Tuy nhiên, không thể phân loại là exact reproduction vì metric chưa khớp tuyệt đối, môi trường thực nghiệm ban đầu chưa được lưu và protocol trong báo cáo không hoàn toàn thống nhất với source code.


## 12. Hạn chế

### 12.1. Khả năng tái tạo dữ liệu giả lập

Source code sinh dữ liệu giả lập ban đầu đã được bổ sung vào repository tại `legacy/original_source/generate_synthetic_data_original.py`. Đây là phiên bản được lưu nguyên trạng nhằm bảo tồn artifact lịch sử của chuyên đề tốt nghiệp.

Repository cũng cung cấp phiên bản có thể chạy độc lập tại `legacy/generate_synthetic_data.py`. Script này sử dụng cùng seed, thứ tự sinh đặc trưng, công thức tính điểm rủi ro, ngưỡng phân vị thứ 90 và cơ chế đảo ngẫu nhiên 3% số nhãn như source code ban đầu. Dữ liệu sinh lại được ghi vào `data/interim/synthetic_regenerated.csv` và không ghi đè frozen dataset tại `data/raw/synthetic/synthetic_data.csv`.

Kết quả đối chiếu cho thấy dataset sinh lại có 5.000 mẫu, 50 Sprint, 100 test case trong mỗi Sprint, 4.386 mẫu nhãn 0 và 614 mẫu nhãn 1. Toàn bộ giá trị và thứ tự các dòng trong dataset sinh lại khớp với frozen CSV.

Do đó, việc thiếu source code sinh dữ liệu không còn được xem là hạn chế của repository. Frozen CSV vẫn được duy trì như artifact lịch sử bất biến, trong khi file trong `data/interim` được sử dụng làm output tạm thời để kiểm chứng khả năng tái tạo dữ liệu.

### 12.2. Môi trường thực nghiệm lịch sử

Thực nghiệm ban đầu của chuyên đề tốt nghiệp được thực hiện trên Google Colab. Tuy nhiên, tại thời điểm thực nghiệm ban đầu, môi trường chưa được đóng băng bằng `pip freeze`, dependency lock file hoặc container image. Vì vậy, phiên bản chính xác của Python và toàn bộ các thư viện tại thời điểm tạo ra kết quả được công bố chưa được xác nhận đầy đủ.

Trong quá trình xây dựng artifact tái lập, notebook được mở lại trên Google Colab và môi trường hiện tại được kiểm tra. Lần kiểm tra này ghi nhận Python 3.12.13, NumPy 2.0.2, Pandas 2.2.2, scikit-learn 1.6.1 và imbalanced-learn 0.14.2. Tuy nhiên, các phiên bản này thuộc runtime Colab tại thời điểm kiểm tra tái lập và không được xem là bằng chứng rằng cùng các phiên bản đã được sử dụng trong thực nghiệm ban đầu.

Repository lưu snapshot của môi trường Colab hiện tại và môi trường tái lập cục bộ để các lần chạy mới có thể được kiểm tra và so sánh. Đồng thời, notebook Colab gốc được lưu dưới dạng artifact lịch sử để tìm kiếm các output, lệnh cài đặt package hoặc metadata còn sót lại có thể giúp khôi phục một phần môi trường ban đầu.

Do chưa có bằng chứng đầy đủ về toàn bộ dependency của runtime lịch sử, repository chưa thể bảo đảm môi trường tái lập hiện tại hoàn toàn giống môi trường đã tạo ra các metric trong báo cáo chuyên đề. Đây vẫn được xem là một hạn chế của quá trình tái lập.


### 12.3. Sự không thống nhất của temporal evaluation

Báo cáo chuyên đề mô tả ví dụ huấn luyện trên Sprint 1–40 và kiểm tra trên Sprint 41–50. Trong khi đó, source code gốc triển khai `TimeSeriesSplit` với năm fold trực tiếp trên từng dòng dữ liệu.

Sự khác biệt này khiến protocol thực tế trong source code không hoàn toàn tương ứng với protocol được mô tả trong tài liệu. Repository giữ lại protocol `original` để tái hiện source code và bổ sung protocol `corrected` để chia dữ liệu theo đơn vị Sprint.

### 12.4. Dữ liệu là dữ liệu giả lập và không đại diện đầy đủ cho dữ liệu CI thực tế.

Mặc dù source sinh dữ liệu đã được cung cấp và có thể tái chạy, dataset vẫn được xây dựng bằng các phân phối xác suất và trọng số được thiết kế thủ công. Dữ liệu này không phản ánh đầy đủ các yếu tố của môi trường CI thực tế như thay đổi mã nguồn theo commit, quan hệ giữa test và file, flaky test, build failure, dependency graph và concept drift theo thời gian.

Do đó, kết quả trên dữ liệu giả lập chỉ được sử dụng để sanity-check pipeline và tái lập chuyên đề, không được sử dụng làm bằng chứng thực nghiệm chính cho luận văn thạc sĩ.

### 12.5. Kết quả corrected là một phân tích phương pháp luận bổ sung, không phải kết quả được công bố trong chuyên đề cũ.

Protocol `corrected` là một phân tích phương pháp luận bổ sung nhằm loại bỏ Sprint overlap giữa tập huấn luyện và tập kiểm tra. Kết quả này không phải là kết quả được công bố trong chuyên đề tốt nghiệp và không được trình bày như một phần của thực nghiệm lịch sử.

Protocol corrected được sử dụng để chỉ ra ảnh hưởng của cách chia dữ liệu và làm cơ sở thiết kế temporal evaluation nghiêm ngặt hơn cho các thực nghiệm tiếp theo.


## 13. Hướng dẫn tái chạy

Từ thư mục gốc repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Kiểm tra dữ liệu:

```bash
python src/validate_data.py
```

Chạy protocol original:

```bash
python legacy/reproduce_legacy.py --mode original
```

Chạy protocol corrected:

```bash
python legacy/reproduce_legacy.py --mode corrected
```

Chạy cả hai:

```bash
python legacy/reproduce_legacy.py --mode both
```

Chạy test:

```bash
python -m pytest tests/test_temporal_split.py -v
```

Các kết quả được ghi tại:

```text
results/legacy/reproduction/
```

## 14. Kết luận

Repository tái lập đã chuyển proof-of-concept của chuyên đề tốt nghiệp từ một source code phụ thuộc vào trạng thái bộ nhớ thành một pipeline có cấu hình, có kiểm tra dữ liệu, có ghi lại môi trường và có output theo từng fold.

Protocol original giúp tái hiện hành vi của source code hiện có. Protocol corrected khắc phục vấn đề chia dữ liệu theo từng dòng bằng cách chia theo đơn vị Sprint, qua đó bảo đảm một Sprint không xuất hiện đồng thời trong train và test.

Kết quả tái lập [[khớp gần đúng/không khớp hoàn toàn/khớp]] với các metric được công bố. Các sai khác được ghi nhận minh bạch và không bị điều chỉnh thủ công để làm cho kết quả mới giống báo cáo cũ.

Artifact này hoàn thành vai trò sanity-check và tạo nền tảng kỹ thuật ban đầu cho luận văn thạc sĩ. Các thực nghiệm chính của luận văn sẽ sử dụng dữ liệu CI thực, đánh giá theo thời gian và protocol được định nghĩa trước.
