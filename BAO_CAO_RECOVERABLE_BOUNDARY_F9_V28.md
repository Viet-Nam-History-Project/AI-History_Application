# Báo cáo cải tiến F9 v28 – ranh giới có thể phục hồi

## Hiện tượng

Với câu hỏi “Chính sách cai trị của Pháp thay đổi như thế nào từ 1897 đến
1945?”, retrieval đã tìm được bằng chứng ở toàn khoảng nhưng câu trả lời vẫn
quay về cách phân kỳ thô. Chẩn đoán truy vấn thật cho thấy period planner đã
tạo nhiều kỳ, nhưng validator loại toàn bộ kế hoạch chỉ vì một ranh giới thiếu
trường actor/action hoặc được biểu diễn ở cuối kỳ thay vì đầu kỳ.

## Nguyên nhân

Luồng cũ là all-or-nothing:

1. Planner đề xuất period plan và các mốc chuyển đổi.
2. Validator yêu cầu mọi mốc phải có đủ actor, action, cause và mặc định mốc
   phải nằm sát đầu kỳ.
3. Chỉ một mốc không qua kiểm tra làm toàn bộ `evolutionPeriods` rỗng.
4. Model sinh câu trả lời bằng fallback nên lại gom các giai đoạn, dù dữ liệu
   chi tiết đã tồn tại trong Neo4j.

## Thay đổi v28

- Subject F9 tiếp tục được chuẩn hóa thành đúng đối tượng lịch sử, không chứa
  cụm hỏi “thay đổi như thế nào” hoặc khoảng năm.
- Chấp nhận bước ngoặt có bằng chứng nằm sát đầu **hoặc** cuối kỳ.
- Mốc giữa kỳ bị gắn nhầm vào kỳ đầu được bỏ khỏi claim thay vì làm hỏng cả
  timeline.
- Actor/action/cause do planner đề xuất không còn là điều kiện loại sớm.
  Semantic-role verifier sẽ sửa chúng từ đoạn trích đã khóa.
- Nếu verifier không đủ chắc chắn về vai nghĩa, chỉ ba claim actor/action/cause
  của ranh giới đó bị để trống. Năm, trạng thái và đoạn bằng chứng vẫn được
  giữ; model bị cấm tự gán nguyên nhân hoặc chủ thể.
- Sửa serializer của `AIQueryLog`: mảng có phần tử `null`, chẳng hạn
  `[null, 1911, 1930, 1939]`, được lưu dưới dạng JSON để phù hợp kiểu property
  của Neo4j.

## Kết quả kiểm chứng thật

Truy vấn kiểm chứng đã tạo được period plan:

- 1897–1911
- 1911–1930
- 1930–1939
- 1939–1945

Retrieval strategy có các cờ `dynamic_periods`,
`periodization_coverage_verified` và `period_coverage`, chứng tỏ câu trả lời
đã dùng period plan động thay vì fallback.

## Phạm vi

Cơ chế này chỉ chạy khi intent là `historical_evolution` (F9 – tiếp nối và
thay đổi). Nó không thay đổi planner của câu hỏi so sánh, diễn biến, nhân vật,
nguyên nhân–kết quả hoặc các dạng câu hỏi khác.

Runtime revision: `recoverable-boundary-evolution-planning-f9-v28`.

## Kiểm thử

- 112 unit tests backend: đạt.
- Python compile: đạt.
- Web Admin TypeScript typecheck: đạt.
