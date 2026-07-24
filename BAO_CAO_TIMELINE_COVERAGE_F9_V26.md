# Báo cáo cải tiến Timeline Coverage cho F9 v26

## 1. Phạm vi

Phiên bản này chỉ áp dụng cho dạng câu hỏi F9 “tiếp nối và thay đổi”, ví dụ:

- một chính sách thay đổi như thế nào qua một khoảng thời gian;
- một tổ chức, phong trào, chiến lược hoặc tiến trình phát triển qua các chặng;
- đâu là yếu tố tiếp nối, điểm đảo chiều và trạng thái cuối kỳ.

Comparison, event phases, nhân vật, nguyên nhân–kết quả và các intent khác
không sử dụng quy tắc phân kỳ của F9.

Runtime revision:

`timeline-coverage-evolution-planning-f9-v26`

## 2. Nguyên nhân câu trả lời v25 phân kỳ chưa tốt

Log của chính truy vấn “Chính sách cai trị của Pháp thay đổi như thế nào từ
1897 đến 1945?” cho thấy:

- `evolutionPeriods`, nhãn giai đoạn, trạng thái và boundary đều rỗng;
- strategy không có `_dynamic_periods`;
- mô hình sinh câu trả lời chỉ nhận 8 chunk cuối cùng.

Như vậy ba giai đoạn 1897–1918, 1919–1930 và 1930–1945 không phải period plan
đã được verifier chấp nhận. Đó là phân kỳ do mô hình sinh tự suy ra từ tập
bằng chứng còn thiếu độ phủ.

Hai nguyên nhân kỹ thuật chính:

1. Hybrid retrieval toàn cục ưu tiên các chunk có điểm cao, nhưng không bảo đảm
   mọi vùng trong khoảng 1897–1945 đều có bằng chứng.
2. Planner cho phép năm trong heading, nhưng bước xác nhận excerpt trước đây
   chỉ đối chiếu với thân `text`. Boundary dựa trên tiêu đề mục/chương có thể
   bị loại âm thầm.

## 3. Luồng F9 v26

### Vòng 1 — Timeline discovery

1. Nhận diện intent F9, subject, domain và khoảng năm do người dùng hỏi.
2. Chia khoảng hỏi thành 4–9 cửa sổ recall liên tục, trung bình khoảng 7 năm.
3. Neo4j full-text lấy tối đa 4 chunk cho mỗi cửa sổ.
4. Ghép kết quả cửa sổ với vector/full-text/entity retrieval ban đầu.
5. Các cửa sổ chỉ dùng bảo đảm độ phủ truy xuất; không phải giai đoạn lịch sử.

Không có mốc lịch sử nào được hard-code. Cùng thuật toán này dùng được cho
chính sách thuộc địa, một chiến lược quân sự, một phong trào hoặc một quá
trình kinh tế trong khoảng năm khác.

### Vòng 2 — Periodization và coverage revision

1. Planner tạo period sơ bộ từ evidence card.
2. Retrieval mở rộng theo các period sơ bộ để lấy thêm state/boundary evidence.
3. Planner chạy lại ở chế độ kiểm tra độ phủ:
   - tách khi có state vector trước/sau thực sự khác và ảnh hưởng kéo dài;
   - gộp khi chỉ có sự kiện nhỏ nhưng trạng thái cốt lõi không đổi;
   - không để một period dài che mất lần đảo chiều có bằng chứng;
   - không biến cửa sổ recall thành period.
4. Semantic-role verifier kiểm tra lại chủ thể, hành động và nguyên nhân của
   từng boundary.
5. Reranker khóa boundary evidence và giữ quota bằng chứng cho từng period.

## 4. Mức bằng chứng thời gian

Evidence card nay phân biệt:

1. `text_years`: năm xuất hiện trực tiếp trong nội dung;
2. `heading_years`: năm xuất hiện trong tiêu đề mục/chương;
3. `metadata_year_start/end`: khoảng thời gian của chunk/nguồn.

Hai mức đầu có thể hỗ trợ boundary nếu nội dung thực sự chứng minh thay đổi
trạng thái. Metadata chỉ dùng recall, không bao giờ đủ để lập ranh giới.

Excerpt verifier đối chiếu trên cả `heading + text`, khắc phục lỗi false
rejection của v25.

## 5. Điều kiện split/merge tổng quát

Một mốc chỉ được dùng để tách period nếu đồng thời:

- tác động trực tiếp đến subject;
- làm thay đổi trạng thái cốt lõi;
- có thể mô tả trạng thái trước và sau khác nhau;
- có ảnh hưởng kéo dài, không chỉ là một biện pháp hay sự kiện cục bộ;
- phù hợp domain của câu hỏi;
- có evidence ID và excerpt kiểm chứng được.

State vector được chọn theo domain. Với chính sách cai trị có thể xét mức tập
quyền, đàn áp/kiểm soát/nhượng bộ, vai trò bộ máy bản xứ và thời bình/thời
chiến. Domain khác phải tự chọn phương diện tương ứng, không dùng checklist
của chính sách cai trị.

## 6. Fail-safe

Nếu period plan vẫn bị verifier từ chối:

- hệ thống ghi rõ lý do rejection vào backend log;
- mô hình sinh không được tự đặt tên các giai đoạn và giả vờ đã phủ toàn kỳ;
- chỉ được trình bày các thay đổi đã có bằng chứng và nêu một lần giới hạn độ
  phủ.

Fail-safe này ngăn trường hợp planner rỗng nhưng câu trả lời vẫn tự tạo ba
giai đoạn như lần thử v25.

## 7. Kiểm chứng

Truy vấn chỉ đọc trên dữ liệu Neo4j hiện tại cho khoảng 1897–1945:

- 7 cửa sổ được kiểm tra;
- lấy được 28 chunk, 4 chunk/cửa sổ;
- cửa sổ 1932–1938 lấy được mục về Chính phủ Mặt trận Nhân dân Pháp;
- cửa sổ 1939–1945 lấy được các chunk có range 1939–1945 và mục Nhật tiến
  chiếm Đông Dương.

Điều này xác nhận evidence cuối thập niên 1930 và thời chiến không còn bị
loại khỏi đầu vào chỉ vì các chunk 1897–1918/1930 có điểm toàn cục cao hơn.

Kết quả tự động:

- Backend: 108 unit tests pass.
- Python compile: pass.
- Web Admin TypeScript: `tsc --noEmit` pass.

## 8. Chi phí và vận hành

Timeline-window recall chỉ đọc Neo4j, không gọi embedding và không tốn token
OpenAI. Với F9 có period plan hợp lệ, có thêm một lượt planner coverage và
semantic verification; chi phí token tăng riêng cho câu hỏi F9 đổi lại độ phủ
và tính đúng đắn cao hơn. Các intent khác giữ nguyên số lượt xử lý.
