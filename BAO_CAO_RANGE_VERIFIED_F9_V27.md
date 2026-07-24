# Báo cáo sửa Range Verification F9 v27

## Hiện tượng sau v26

Truy vấn thực tế sau khi triển khai v26 đã lấy được 43 candidate và có đủ
bằng chứng ở các vùng 1936, 1940 và 1945. Tuy nhiên:

- `evolutionPeriods` vẫn rỗng;
- strategy có `_timeline_window_coverage` nhưng không có `_dynamic_periods`;
- câu trả lời tiếp tục tự gộp 1930–1945.

Một lần chạy chẩn đoán trực tiếp ghi nhận lý do chính xác:

`F9 period plan rejected (discovery): periods do not cover requested boundaries`

Planner đã trả các period không bắt đầu/kết thúc đúng tuyệt đối tại hai năm do
người dùng chọn. Validator v26 loại toàn bộ kế hoạch dù evidence của period
đầu/cuối vẫn bao phủ 1897 và 1945.

## Thay đổi v27

Runtime revision:

`range-verified-evolution-planning-f9-v27`

### 1. Phân biệt range cut và natural boundary

- Năm đầu/cuối câu hỏi là biên cắt do người dùng chọn, không nhất thiết là
  bước ngoặt tự nhiên.
- Nếu evidence của period đầu thực sự bao phủ năm đầu, backend được phép mở
  period đầu tới biên hỏi.
- Nếu evidence của period cuối bao phủ năm cuối, backend được phép mở period
  cuối tới biên hỏi.
- Nếu evidence không bao phủ biên, plan vẫn bị loại và log nêu rõ đầu hay
  cuối đang thiếu.

Cơ chế này không hard-code 1897 hoặc 1945 và áp dụng cho mọi khoảng F9.

### 2. Ground excerpt từ nguồn

Backend không còn phụ thuộc hoàn toàn vào chuỗi excerpt do model chép lại.
Sau khi model chọn evidence ID và boundary year, backend lấy chính câu chứa
năm đó từ `heading + text`. Semantic verifier vì vậy luôn nhận đoạn nguồn
thật, giảm false rejection do OCR, dấu câu hoặc model diễn đạt lại.

### 3. Làm sạch subject F9

Trước đây subject bị nhận thành:

`Chính sách cai trị của Pháp thay đổi`

V27 tách thành:

`Chính sách cai trị của Pháp`

Phần “thay đổi như thế nào” và khoảng năm được giữ ở intent/range, không còn
lẫn vào entity/exact phrase. Subject sạch được dùng nhất quán cho retrieval,
grounding và query recovery.

### 4. Kiểm soát subject type

Nếu model trả nhầm tên facet như `historical_evolution` vào `subject_type`,
backend tự quy về loại đối tượng theo domain, ví dụ `governance_policy`, thay
vì truyền nhãn sai xuống period planner.

## Kiểm thử

- 110 backend unit tests pass.
- Python compile pass.
- Có test riêng cho làm sạch subject.
- Có test period plan bắt đầu/kết thúc lệch biên nhưng evidence vẫn phủ toàn
  khoảng.
- Có test nâng prompt runtime từ v25/v26 lên v27.

