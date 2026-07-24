# Báo cáo Dynamic Evolution Planner F9 v24

## 1. Phạm vi

Phiên bản này chỉ áp dụng khi `QueryPlan.intent == historical_evolution`, tức
nhóm F9 về tiếp nối và thay đổi. Comparison, event phases, overview, kết
quả–ý nghĩa và các planner khác không đi qua bước phân kỳ động này.

## 2. Vấn đề của v23

V23 đã cân bằng retrieval theo thời gian nhưng còn lưu một số bộ giai đoạn
theo câu hỏi mẫu. Cùng khoảng 1965–1976, tiến trình quân sự, ngoại giao, kinh
tế và tổ chức nhà nước cần những ranh giới khác nhau. Khoảng năm chỉ là biên
tìm kiếm; subject và bước ngoặt mới quyết định periodization.

## 3. Luồng sáu lớp

1. Nhận diện subject và `subject_type`.
2. Nhận diện subtype F9 và domain.
3. Truy xuất rộng ứng viên bước ngoặt trong Graph/chunks, bao gồm trạng thái
   đầu–cuối, thay đổi chiến lược/chính sách, hiệp định, cải cách, khủng hoảng,
   thay đổi chính quyền, giải phóng và thống nhất.
4. Lấy mẫu evidence theo toàn khoảng hỏi rồi chấm bước ngoặt theo mức liên
   quan subject, quy mô thay đổi, vai trò nhân quả, độ mạnh ranh giới, độ phủ
   bằng chứng, ý nghĩa lịch sử và trùng lặp.
5. Tạo 2–7 giai đoạn động. Mỗi giai đoạn bắt buộc có evidence ID hợp lệ; tên
   PDF và khoảng năm của tập sách không được dùng làm ranh giới lịch sử.
6. Chọn facet theo domain rồi mở rộng retrieval theo period plan vừa tạo.

## 4. Bảo vệ độ chính xác

- Period plan phải bao phủ đúng khoảng người dùng hỏi và không có khoảng trống.
- Không chấp nhận period không có evidence ID hoặc kế hoạch có confidence dưới
  ngưỡng.
- Nếu khoảng hỏi vượt vòng đời subject, planner đánh dấu `range_mismatch` và
  yêu cầu giải thích quá trình kế tiếp thay vì kéo dài subject.
- Subject quá rộng được phân loại `multi_domain`; câu trả lời phải nói rõ các
  phương diện được xét và giữ độ sâu tổng quan.
- Khi không đủ bằng chứng để phân kỳ, hệ thống giữ retrieval theo facet và
  không tự tạo mốc.

## 5. Quan sát trong Web Admin

Diagnostics trả thêm:

- `evolutionSubjectType`;
- `evolutionPeriods`;
- `evolutionPeriodLabels`;
- `evolutionBoundaryCauses`;
- `evolutionSubjectLifetime`;
- `evolutionRangeMismatch`;
- `evolutionRangeResolution`.

Mục **Period plan động F9** trong Kết quả & bằng chứng cho phép Admin kiểm tra
tên từng chặng, nguyên nhân tạo bước ngoặt và trường hợp khoảng hỏi vượt vòng
đời đối tượng.

## 6. Phiên bản và kiểm thử

- Runtime revision: `dynamic-evolution-planning-f9-v24`.
- Toàn bộ 101 unit test backend đã đạt; Web Admin đạt TypeScript type-check
  và ESLint.
- Có regression test bảo đảm query parser không định sẵn period cho hai subject
  dùng cùng khoảng năm.
- Có test hai pha retrieval, cân bằng evidence theo period động và xử lý câu
  hỏi vượt vòng đời “Chiến tranh đặc biệt”.
