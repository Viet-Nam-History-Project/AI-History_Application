# Báo cáo Evolution Planner F9 v23

## 1. Mục tiêu

F9 là nhóm câu hỏi về sự tiếp nối và thay đổi theo thời gian. Phiên bản trước
nhận diện được `historical_evolution`, nhưng chỉ tìm một facet rất rộng. Vì vậy,
RAG có thể lấy nhiều đoạn sâu của một giai đoạn mà bỏ sót bước ngoặt khác.

Ví dụ câu hỏi:

> Chính sách cai trị của Pháp thay đổi như thế nào từ 1897 đến 1945?

Câu trả lời cũ có chi tiết về bộ máy, báo chí hoặc thuế khóa nhưng mất cân đối:
gộp 1897–1918, thiên lệch giai đoạn 1919–1930, bỏ 1936–1939 và chưa kết thúc
đúng tại cuộc đảo chính ngày 9/3/1945.

## 2. Taxonomy yêu cầu F9

Planner phân loại trước khi retrieval:

| Mã | Trọng tâm |
| --- | --- |
| `evolution_over_time` | Trình bày tiến trình thay đổi |
| `continuity_and_change` | Điều tiếp nối và điều thay đổi |
| `turning_points` | Các bước ngoặt |
| `extent_of_change` | Mức độ thay đổi |
| `cause_of_change` | Nguyên nhân thay đổi |
| `before_after` | Trước và sau một mốc |
| `short_long_term` | Thay đổi ngắn hạn và dài hạn |
| `acceleration_slowdown` | Tốc độ thay đổi |
| `reversal` | Sự đảo chiều |
| `inheritance_development` | Kế thừa và phát triển |
| `periodization` | Chia giai đoạn |
| `decisive_change` | Thay đổi có ý nghĩa quyết định |

Planner tiếp tục nhận diện domain như chính trị–hành chính, kinh tế, chiến lược
quân sự, chiến dịch, phong trào cách mạng hoặc nhà nước–triều đại. Facet được
chọn từ taxonomy của đúng domain; danh mục không phải checklist.

## 3. Luồng xử lý

```text
Câu hỏi
  → chuẩn hóa lỗi gõ/alias
  → nhận diện historical_evolution
  → phân loại evolutionIntent
  → phân loại evolutionDomain
  → bảo vệ khoảng thời gian người dùng nêu
  → chọn 3–6 facet phù hợp
  → tạo truy vấn facet chuyên biệt
  → tạo truy vấn cho từng chặng đã biết
  → hybrid retrieval và rerank
  → lọc chunk nằm ngoài khoảng thời gian
  → giữ bằng chứng cân bằng giữa các chặng
  → sinh câu trả lời: luận đề → tiến trình → nhận xét F9
```

Model planner chỉ chọn ID facet trong danh sách cho phép. Nó không được thêm
sự kiện lịch sử, sửa mốc thời gian hoặc trả lời ở bước planning. Nếu model lỗi,
kế hoạch deterministic vẫn hoạt động.

## 4. Cân bằng thời gian cho trường hợp 1897–1945

Backend tạo truy vấn riêng cho:

1. 1897–1914.
2. 1914–1918.
3. 1919–1929.
4. 1930–1935.
5. 1936–1939.
6. 1939–1945.

Các chặng này là lịch truy xuất, không phải dữ kiện được chèn thẳng vào đáp án.
Một mốc chỉ được model trình bày nếu chunk đã chọn có bằng chứng hỗ trợ.

Selector ưu tiên một chunk mạnh cho mỗi chặng dựa trên:

- năm ở metadata hoặc trong nội dung chunk;
- độ cụ thể của khoảng thời gian;
- tên riêng, biện pháp, mốc hoặc số liệu;
- độ liên quan của retrieval;
- dấu hiệu chuyển biến và bước ngoặt.

Sau đó selector mới bổ sung các chunk tốt theo thứ hạng. Nhờ vậy một giai đoạn
có nhiều đoạn tương tự không lấn át toàn bộ tiến trình.

## 5. Facet mới

Hai facet được bổ sung cho câu hỏi về chính sách cai trị:

- `governance_methods`: đàn áp, khủng bố, kiểm soát báo chí, cảnh sát, tư pháp,
  nhượng bộ dân chủ và cải cách mang tính mị dân.
- `wartime_mobilization`: huy động nhân lực, tài chính, vật lực, công phiếu,
  trưng dụng, quan hệ Pháp–Nhật và bước ngoặt thời chiến.

Không bắt buộc index lại. `text_supports_facet()` vẫn nhận diện hai facet bằng
nội dung chunk cũ; lần index sau sẽ ghi facet trực tiếp vào metadata như bình
thường.

## 6. Kiểm tra dữ liệu thực tế

Kiểm tra chỉ đọc trên Neo4j cho ba tập `1897-1918`, `1919-1930` và `1930-1945`
cho thấy kho dữ liệu có đủ tín hiệu quan trọng để cải thiện câu trả lời:

| Từ khóa kiểm tra | Số chunk |
| --- | ---: |
| Chiến tranh thế giới thứ nhất | 12 |
| Khai thác thuộc địa lần thứ hai | 32 |
| Xô viết Nghệ… | 33 |
| Mặt trận Nhân dân | 84 |
| Nhượng bộ | 42 |
| Phát xít hóa | 3 |
| Pháp – Nhật | 25 |
| Nhật đảo chính Pháp | 11 |

Kết luận: thiếu sót của câu trả lời chủ yếu đến từ planning và retrieval chưa
cân bằng giai đoạn, không phải do kho dữ liệu hoàn toàn thiếu nội dung.

## 7. Quy tắc tạo câu trả lời

Chỉ dẫn runtime `ADAPTIVE_EVOLUTION_PLANNING_F9_V23` yêu cầu:

1. Mở đầu bằng luận đề về điều thay đổi và điều tiếp nối.
2. Phân kỳ theo bước ngoặt có bằng chứng, không chia đều máy móc.
3. Mỗi giai đoạn ưu tiên thời gian, động lực, đặc điểm mới và minh họa cụ thể.
4. Chọn facet đúng domain; không tạo mục không liên quan.
5. Kết luận đúng subtype F9, không chỉ kể dòng thời gian.
6. Không dùng kiến thức ngoài nguồn để lấp chỗ trống.

Chỉ dẫn bắt buộc được ghép vào prompt active ở runtime và được trả về trang
“Phiên bản prompt”, nên nội dung Web Admin luôn phản ánh đúng logic backend.

## 8. Quan sát trên Web Admin

Phần “Kết quả & bằng chứng” hiển thị thêm:

- dạng F9;
- miền tiến trình;
- các chặng cần cân bằng bằng chứng;
- facet đã có hoặc còn thiếu bằng chứng;
- chiến lược retrieval có hậu tố `_evolution_timeline`,
  `_adaptive_f9`, `_evolution_period_gate` và `_period_coverage`.

Nguồn và số trang vẫn chỉ xuất hiện trong Web Admin, không hiển thị trên app
người dùng.

## 9. Phiên bản và kiểm thử

- RAG revision: `adaptive-evolution-planning-f9-v23`.
- Toàn bộ 97 unit test backend đã đạt; Web Admin cũng đạt TypeScript
  type-check và ESLint.
- Có regression test cho subtype F9, taxonomy đúng domain, truy vấn từng chặng
  và selector giữ bằng chứng trải đều theo thời gian.
