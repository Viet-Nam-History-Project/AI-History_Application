# Báo cáo thiết kế Comparison Planner ba tầng v22

## 1. Bài toán

Câu hỏi so sánh lịch sử không chỉ có dạng “nêu điểm giống và khác”. Người dùng
có thể chỉ hỏi một khác biệt quyết định, một nhóm tiêu chí cụ thể, nguyên nhân,
kết quả, ý nghĩa, mức độ hiệu quả, vai trò hoặc tính kế thừa. Nếu hệ thống dùng
một checklist chung cho tất cả câu hỏi, retrieval sẽ tìm những nội dung không
liên quan và model tạo ra các hàng như “chưa có dữ liệu về giao thông vận tải”
trong một câu hỏi về chiến lược quân sự.

Phiên bản v22 giải quyết vấn đề này bằng planning ba tầng:

1. Nhận diện kiểu yêu cầu so sánh.
2. Nhận diện loại đối tượng lịch sử.
3. Chọn tiêu chí động, sau đó kiểm chứng riêng cho từng đối tượng.

Đây là cải tiến retrieval và answer planning, không phải fine-tune model và
không thay đổi dữ liệu gốc trong PDF.

## 2. Kiến trúc tổng quát

```text
Câu hỏi người dùng
  → chuẩn hóa lỗi gõ/alias an toàn
  → trích Object A và Object B
  → nhận diện comparisonIntent
  → nhận diện comparisonDomain/objectType
  → trích explicitFacets do người dùng nêu
  → model planner chọn facet trong taxonomy an toàn
  → tạo truy vấn Object × Facet
  → hybrid retrieval: vector + full-text + graph/alias
  → gán từng chunk cho Object A/B
  → rerank và tạo ma trận bằng chứng Object × Facet
  → chọn bố cục theo comparisonIntent
  → sinh câu trả lời có căn cứ
  → trả citations/diagnostics riêng cho Web Admin
```

Planner chạy trước retrieval để không bị một vài kết quả đầu tiên làm lệch phạm
vi câu hỏi. Model planner chỉ được phân loại và chọn ID facet; nó không được trả
lời lịch sử hoặc thêm sự kiện ở bước này.

## 3. Tầng 1 — Comparison intent

| Mã | Ý nghĩa | Bố cục mặc định |
| --- | --- | --- |
| `similarities_differences` | So sánh toàn diện | Bảng chọn lọc, điểm giống và khác |
| `main_similarity` | Điểm giống chủ yếu | Trả lời trực tiếp điểm chung và căn cứ |
| `main_difference` | Khác biệt chính | Trả lời trực tiếp khác biệt và ý nghĩa |
| `focused_facets` | Người dùng chỉ định tiêu chí | Chỉ trình bày các tiêu chí được nêu |
| `cause_comparison` | So sánh nguyên nhân | Đối chiếu nguyên nhân và giải thích |
| `consequence_comparison` | So sánh kết quả/tác động | Kết quả trực tiếp và tác động lâu dài |
| `effectiveness_comparison` | Đánh giá hiệu quả | Nêu thước đo, đối chiếu và phán đoán |
| `significance_comparison` | So sánh ý nghĩa | Ý nghĩa của từng đối tượng và khác biệt |
| `continuity_change` | Tiếp nối và thay đổi | Yếu tố kế thừa, yếu tố mới, nguyên nhân |
| `role_comparison` | So sánh vai trò | Đóng góp, hành động, phạm vi ảnh hưởng |
| `interpretation_comparison` | So sánh tư liệu/quan điểm | Luận điểm, bằng chứng, góc nhìn, giới hạn |
| `extent_comparison` | So sánh mức độ | Tiêu chí đánh giá và kết luận có điều kiện |
| `judgement` | Kiểm tra nhận định | Kết luận, bằng chứng ủng hộ/phản bác |

Các cờ `answerRequirements` được sinh từ intent:

```json
{
  "includeSimilarities": true,
  "includeDifferences": true,
  "includeExplanation": true,
  "includeJudgement": false
}
```

Nhờ đó, câu hỏi “Khác biệt cơ bản giữa A và B là gì?” không còn bị biến thành
một bảng toàn diện.

## 4. Tầng 2 — Domain và object type

v22 hỗ trợ các domain ưu tiên sau:

| Domain | Đối tượng tiêu biểu |
| --- | --- |
| `economic_policy` | Cuộc khai thác thuộc địa, chính sách kinh tế |
| `military_strategy` | Chiến tranh đặc biệt, Chiến tranh cục bộ |
| `military_campaign` | Chiến dịch, trận đánh |
| `diplomatic_agreement` | Hiệp định, hội nghị, đàm phán |
| `historical_person` | Nhân vật và chủ trương hoạt động |
| `movement_revolution` | Phong trào, khởi nghĩa, cách mạng |
| `political_administration` | Bộ máy cai trị, cải cách hành chính |
| `state_dynasty` | Nhà nước, triều đại, chế độ |
| `source_interpretation` | Hai tư liệu hoặc hai cách đánh giá |
| `historical_event` | Fallback cho sự kiện lịch sử khác |

Bộ quy tắc cục bộ tạo domain dự phòng. Model planner có thể sửa domain trong
danh sách cho phép nếu tên đối tượng không có marker rõ ràng. Khi model lỗi,
hệ thống vẫn tiếp tục bằng domain dự phòng.

## 5. Tầng 3 — Chọn facet động

Mỗi domain có một taxonomy ứng viên riêng. Ví dụ:

```text
economic_policy
  → bối cảnh
  → người tổ chức
  → mục tiêu/hệ quả
  → quy mô/vốn
  → nông nghiệp
  → công nghiệp
  → thương nghiệp
  → giao thông
  → tác động xã hội

military_strategy
  → bối cảnh
  → mục tiêu
  → lực lượng
  → phạm vi
  → âm mưu/biện pháp
  → quy mô/mức độ
  → thắng lợi tiêu biểu
  → kết quả

diplomatic_agreement
  → bối cảnh
  → các bên tham gia
  → mục tiêu
  → nội dung/điều khoản
  → cơ chế thực hiện
  → kết quả
  → ý nghĩa
  → hạn chế
```

Thứ tự ưu tiên:

```text
explicitFacets người dùng nêu
  → facet bắt buộc của comparisonIntent
  → facet cốt lõi của domain
  → facet có giá trị phân biệt
  → facet bổ sung có đủ bằng chứng
```

Nếu `explicitFacets` không rỗng, model không được tự mở rộng ngoài chúng. Câu
hỏi toàn diện chọn tối đa 4–7 facet; câu hỏi khác biệt/đánh giá chọn 2–4; câu
hỏi nguyên nhân, kết quả, ý nghĩa hoặc vai trò chọn 1–4.

## 6. QueryPlan nội bộ

Ví dụ plan cho câu hỏi quân sự toàn diện:

```json
{
  "intent": "comparison",
  "comparisonIntent": "similarities_differences",
  "comparisonDomain": "military_strategy",
  "comparisonSubjects": [
    "Chiến tranh đặc biệt",
    "Chiến tranh cục bộ"
  ],
  "comparisonObjectTypes": [
    "military_strategy",
    "military_strategy"
  ],
  "explicitFacets": [],
  "requiredFacets": [
    "comparison_context",
    "forces",
    "geographic_scope",
    "strategy_methods",
    "representative_events",
    "result"
  ],
  "answerStructure": "comparison_matrix_then_similarities_differences"
}
```

Ví dụ người dùng chỉ định:

```text
“So sánh Chiến tranh đặc biệt và Chiến tranh cục bộ
 về lực lượng và phạm vi”
```

```json
{
  "comparisonIntent": "focused_facets",
  "explicitFacets": ["forces", "geographic_scope"],
  "requiredFacets": ["forces", "geographic_scope"],
  "answerStructure": "comparison_selected_facets"
}
```

## 7. Retrieval Object × Facet

Backend tạo một truy vấn chi tiết cho mỗi cặp:

```text
Object A × Facet 1
Object B × Facet 1
Object A × Facet 2
Object B × Facet 2
...
```

Mỗi truy vấn gồm:

- mô tả chi tiết của facet;
- tên chuẩn và alias của đối tượng;
- khoảng thời gian nếu đã xác định chắc chắn.

Các tập kết quả được hợp nhất và loại trùng trước khi rerank. Với câu hỏi chỉ có
ít facet, số truy vấn và `top_k` giảm theo, giúp tiết kiệm embedding token và
giảm latency so với việc luôn chạy một checklist lớn.

## 8. Gán chunk cho từng đối tượng

Một chunk được gán cho A/B theo thứ tự:

1. So khớp tên/alias có ý nghĩa trong heading, text và related entities.
2. Nếu tên không đủ rõ, dùng độ giao nhau của khoảng thời gian.
3. Với nguồn có khoảng năm rộng, chọn phía có số năm giao nhau lớn nhất.

Quy tắc này tránh dùng đoạn của đối tượng A để suy diễn nội dung cho B.

## 9. Ma trận bằng chứng

Sau rerank, backend tạo `comparisonEvidence`:

```json
{
  "forces": {
    "Chiến tranh đặc biệt": ["chunk-a", "chunk-b"],
    "Chiến tranh cục bộ": ["chunk-c"]
  },
  "result": {
    "Chiến tranh đặc biệt": ["chunk-d"],
    "Chiến tranh cục bộ": []
  }
}
```

Từ ma trận này hệ thống tính:

- `balancedFacets`: facet có bằng chứng cho tất cả đối tượng;
- `missingComparisonCells`: ô `facet::object` còn thiếu;
- facet bổ sung thiếu một phía sẽ không được dùng để tạo hàng;
- facet do người dùng yêu cầu nhưng thiếu một phía vẫn được báo rõ.

Web Admin hiển thị ma trận và số chunk trong từng ô để quản trị viên kiểm tra.
Ứng dụng mobile không hiển thị metadata quản trị này.

## 10. Theo dõi vận hành và phục vụ báo cáo

Mỗi lượt hỏi được ghi vào `AIQueryLog`. Ngoài các chỉ số retrieval cũ, v22 lưu
thêm:

- `comparisonIntent`, `comparisonDomain`;
- `comparisonSubjects`, `comparisonObjectTypes`;
- `explicitFacets`;
- `comparisonEvidence`;
- `balancedFacets`, `missingComparisonCells`;
- `answerStructure`, `answerRequirements`;
- số candidate, số chunk được chọn, độ bao phủ và thời gian xử lý.

Nhờ đó có thể thống kê riêng các câu hỏi so sánh, biết planner đã chọn gì và
phân biệt được ba nguyên nhân chất lượng kém:

1. planner chọn sai intent/domain/facet;
2. retrieval không tìm đủ bằng chứng cho một phía;
3. model sinh câu trả lời chưa tuân thủ plan dù bằng chứng đã đủ.

Log không lưu nguyên nội dung PDF vào Firestore. Dữ liệu chẩn đoán và citation
được quản lý ở backend/Neo4j và chỉ phục vụ Admin.

## 11. Sinh câu trả lời

Prompt sinh câu trả lời nhận:

- comparison intent và domain;
- hai đối tượng;
- explicit facets;
- answer requirements;
- balanced facets và ô còn thiếu;
- các chunk đã chọn;
- citations tách riêng.

Quy tắc trình bày:

- chỉ dùng bảng khi có nhiều tiêu chí;
- không viết tên tài liệu, số trang hoặc ký hiệu nguồn trong nội dung app;
- không tạo hàng không liên quan;
- không âm thầm bỏ facet người dùng yêu cầu;
- câu hỏi đánh giá phải nêu tiêu chí đánh giá trước khi phán đoán;
- câu hỏi khác biệt chính phải trả lời trực tiếp, không kể toàn bộ lịch sử A/B.

Nguồn và trang vẫn được trả trong `citations` và chỉ hiển thị tại Web Admin.

## 12. Fallback và an toàn

- Chuẩn hóa câu hỏi chỉ sửa lỗi có độ chắc chắn cao.
- Planner model chỉ chọn enum/domain/facet đã cho phép.
- Nếu planner lỗi, dùng plan cục bộ dự phòng.
- Nếu reranker lỗi, dùng hybrid ranking và bộ chọn coverage xác định.
- Không bổ sung sự kiện ngoài nguồn truy xuất.
- Explicit facet không được tự ý thay đổi.
- Kết quả cuối vẫn đi qua period gate và kiểm tra độ liên quan.

## 13. Chi phí token

Câu hỏi so sánh có thêm một chat call nhỏ để lập plan. Ngoài ra:

- một embedding cho câu hỏi;
- embedding cho các truy vấn Object × Facet;
- một chat call rerank;
- một chat call sinh câu trả lời.

Việc nhận diện explicit facets và giới hạn 1–7 facet giúp tránh tạo hàng chục
truy vấn không cần thiết. Các call này xảy ra lúc hỏi chatbot, không phải lúc
Index PDF. Token Index và token hỏi đáp được theo dõi riêng.

## 14. Kiểm thử

Các nhóm regression cần duy trì:

1. So sánh hai cuộc khai thác thuộc địa toàn diện.
2. So sánh chiến lược quân sự không xuất hiện facet kinh tế.
3. Câu hỏi chỉ định “lực lượng và phạm vi” không bị mở rộng.
4. Câu hỏi “khác biệt cơ bản” không tạo bảng toàn diện.
5. Câu hỏi so sánh nguyên nhân chỉ truy xuất facet nguyên nhân.
6. Hiệp định được phân loại `diplomatic_agreement`.
7. Ma trận bằng chứng phát hiện đúng ô thiếu của A/B.
8. Planner lỗi vẫn dùng fallback.
9. Prompt Web Admin và runtime dùng cùng revision.

Revision triển khai: `three-layer-comparison-planning-v22`.

Kết quả kiểm thử tại thời điểm triển khai:

- 92/92 unit và regression test backend đạt;
- `tsc --noEmit` của Web Admin đạt;
- health runtime trả revision v22 và không có job Index/chat đang treo;
- ca “về lực lượng và phạm vi” chỉ chọn `forces`,
  `geographic_scope`, cả hai facet đều cân bằng và không có ô thiếu;
- ca “điểm khác biệt cơ bản” chọn `main_difference`, dùng bố cục trả lời trực
  tiếp, không đưa các facet kinh tế như nông nghiệp, công nghiệp hoặc giao thông.

## 15. Vị trí code

- `SourceCode/src/api/query_analysis.py`: intent, explicit facets và fallback plan.
- `SourceCode/src/api/history_facets.py`: taxonomy facet/domain.
- `SourceCode/src/api/rag_service.py`: model planner, retrieval, evidence matrix,
  prompt và answer requirements.
- `SourceCode/src/api/models.py`: diagnostics trả về Web Admin.
- `Web-Admin-for-managing-History-App/src/components/ai/EvaluationWorkbench.tsx`:
  màn hình kiểm tra plan và ma trận bằng chứng.

## 16. Hướng phát triển tiếp

- Nhận diện tốt hơn hai nhân vật chỉ dựa trên tên riêng.
- Hỗ trợ phép so sánh khác loại, ví dụ nhân vật với tổ chức.
- Hiện tại recovery và gán phía đã dùng bảng alias lịch sử đã kiểm soát; bước
  tiếp theo là đọc thêm alias động từ Entity trong Neo4j ngay khi phân loại object.
- Chấm điểm bằng chứng theo từng ô thay vì chỉ đếm chunk.
- Tạo bộ benchmark riêng cho từng comparison intent và domain.
- Thêm cache plan cho câu hỏi lặp lại để giảm một chat call.
