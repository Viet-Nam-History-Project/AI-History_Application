# Báo cáo refactor RAG strict-corpus v29

## 1. Vấn đề trước khi refactor

`rag_service.py` đã tăng lên hơn 5.200 dòng và riêng `answer()` khoảng 1.006
dòng. Một request chat phải đi qua chuẩn hóa, planning, retrieval, recovery,
coverage, sinh câu trả lời và logging trong cùng một method. Ngoài khó kiểm
thử, thiết kế cũ còn có nhánh best-effort cho phép model trả lời bằng kiến thức
riêng khi retrieval không đủ bằng chứng.

Hai vấn đề này liên quan trực tiếp với nhau: khi planning, retrieval và
generation không có hợp đồng dữ liệu rõ ràng, một retrieval miss rất dễ bị hiểu
nhầm thành “kho không có dữ liệu”, hoặc bị lấp bằng kiến thức không truy vết
được.

## 2. Nguyên tắc của v29

- Chỉ sinh nội dung lịch sử từ evidence đã chọn trong kho PDF.
- Không dùng model knowledge fallback.
- Không hardcode dữ kiện của một sự kiện cụ thể trong control flow hay prompt
  fallback.
- Mỗi yêu cầu trong câu hỏi được theo dõi như một `AtomicRequirement`.
- Phân biệt rõ retrieval miss với corpus gap đã được kiểm chứng.
- Giữ `RagService` làm API tương thích trong lúc tách dần pipeline, tránh một
  lần viết lại lớn gây regression cho comparison, F9 và multi-facet event.

## 3. Các module đã tách

```text
src/api/rag/
├── contracts.py              # AtomicRequirement và EvidenceLedger
├── conversation_resolver.py  # giải tham chiếu câu hỏi nối tiếp trước retrieval
├── question_decomposer.py    # QueryPlan → các yêu cầu độc lập
├── coverage_gate.py          # đánh giá trạng thái evidence theo requirement
├── diagnostics.py            # ánh xạ trạng thái pipeline ra API diagnostics
├── citation_mapper.py        # context cho model và citation cho Web Admin
├── grounded_generator.py     # sinh câu trả lời ở chế độ STRICT_CORPUS
├── observability.py          # payload AIQueryLog nhất quán
└── presentation.py           # làm sạch citation/meta language trong câu trả lời
```

Nhờ đó `answer()` còn khoảng 718 dòng ở v29. Mục tiêu 100–250 dòng chưa được
tuyên bố hoàn tất: phần retrieval/recovery cũ vẫn đang được tách theo từng
stage để không làm hỏng các planner đã kiểm thử. Tuy nhiên, generation,
coverage, citations, diagnostics, conversation và logging không còn nằm trong
God Method.

## 4. Evidence ledger

Mỗi requirement có một trong các trạng thái:

| Trạng thái | Ý nghĩa |
| --- | --- |
| `not_searched` | Chưa chạy truy xuất cho yêu cầu |
| `retrieval_miss` | Đã truy xuất nhưng chưa chọn được evidence |
| `partial_evidence` | Có evidence nhưng chưa đủ chi tiết để xác nhận hoàn chỉnh |
| `verified` | Evidence đáp ứng yêu cầu |
| `source_conflict` | Các nguồn được chọn mâu thuẫn |
| `verified_corpus_gap` | Đã audit corpus và xác nhận kho thật sự thiếu |

Quy tắc quan trọng: danh sách evidence rỗng mặc định là `retrieval_miss`, tuyệt
đối không tự đổi thành `verified_corpus_gap`. Vì vậy chatbot không còn nói
“kho không có dữ liệu” chỉ vì một truy vấn tìm kiếm chưa tốt.

## 5. Luồng mới

```text
Câu hỏi + lịch sử hội thoại
  → ConversationResolver tạo câu hỏi độc lập
  → normalize + QueryPlan
  → retrieval/rerank/recovery hiện hành
  → QueryPlan được tách thành AtomicRequirement
  → EvidenceLedger đối chiếu requirement ↔ chunk
  → CoverageGate
      ├─ không có evidence: trả trạng thái không đủ bằng chứng, không gọi model
      └─ có evidence: GroundedAnswerGenerator ở STRICT_CORPUS
  → CitationMapper giữ nguồn cho Web Admin
  → Observability lưu diagnostics và requirement statuses
```

App vẫn chỉ nhận phần trả lời tự nhiên. Citation, số trang, requirement status
và chẩn đoán retrieval nằm trong response/log để Web Admin kiểm tra.

## 6. Câu hỏi nối tiếp

Trước đây câu hỏi như “Còn lực lượng phía Pháp?” có thể được planning như một
câu độc lập và mất chủ thể của lượt trước. `ConversationResolver` hiện chạy
trước normalize, planning và embedding. Nó chỉ kế thừa chủ thể đã xuất hiện
trong tin nhắn người dùng trước đó, không tự thêm ngày tháng, nhân vật hay dữ
kiện lịch sử.

## 7. Thay đổi hành vi khi thiếu evidence

Nhánh `_answer_event_phases_best_effort()` và các chiến lược
`model_knowledge_fallback` đã bị loại bỏ. Khi không requirement nào có evidence
được chọn:

- không gọi chat model để tự điền kiến thức;
- không tạo citation giả;
- confidence bằng 0;
- log ghi `strictCorpus=true`, `grounded=false` và trạng thái từng requirement.

Khi chỉ một số requirement có evidence, generator được phép trả lời các phần
có căn cứ và phải thể hiện đúng phần chưa đủ bằng chứng; nó không được tự bù
chi tiết từ trí nhớ model.

## 8. Khả năng kiểm thử

Các phần thuần logic không phụ thuộc Neo4j/OpenAI đã có unit test riêng:

- retrieval miss không bị đổi thành corpus gap;
- partial evidence vẫn luôn gắn với chunk thật;
- chỉ corpus audit rõ ràng mới tạo `verified_corpus_gap`;
- decomposer hoạt động theo facet, không theo tên sự kiện;
- câu hỏi nối tiếp được giải tham chiếu trước retrieval;
- source code không còn prompt best-effort và dữ kiện trả lời hardcode.

Toàn bộ **124/124 test** trong virtualenv dự án hiện đạt. `py_compile` đạt cho
`rag_service.py`, `models.py` và toàn bộ `src/api/rag/`; `git diff --check`
không phát hiện lỗi khoảng trắng hay patch hỏng.

## 9. Revision và triển khai

Revision mới là `strict-corpus-modular-rag-v29`. FastAPI phải nạp lại process
để import các module mới. Cơ chế revision watcher hiện có sẽ chỉ restart khi
không còn chat request hay index job đang chạy; cũng có thể restart thủ công
bằng service/script của dự án.

## 10. Phần refactor tiếp theo

Để đạt mục tiêu `answer()` 100–250 dòng mà vẫn giữ hành vi ổn định, phần còn
lại sẽ được tách theo thứ tự:

1. `retrieval_orchestrator.py`: initial, planned và subject/timeline retrieval;
2. `recovery_orchestrator.py`: rewrite, missing-facet retry và evidence fallback;
3. `comparison_planner.py` và `evolution_planner.py`: chuyển logic planner lớn
   ra khỏi service;
4. protocol cho repository/model để unit test từng stage bằng fake;
5. regression fixtures cho từng intent trước khi xóa lớp tương thích cũ.

Đây là refactor theo đường nối tiếp có kiểm chứng, không phải viết lại toàn bộ
pipeline trong một lần.
