# Báo cáo cải tiến câu hỏi sự kiện nhiều ý M2

## 1. Mục tiêu và phạm vi

Một câu như “Chiến dịch Điện Biên Phủ diễn biến thế nào, ai chỉ huy, lực
lượng nào tham gia và kết quả ra sao?” thực chất chứa nhiều yêu cầu độc lập.
Trước M2, hybrid retrieval có thể ưu tiên nhiều chunk đầu sách hoặc nhiều
chunk về cùng một hệ quả, khiến câu trả lời bỏ mất diễn biến, chỉ huy, lực
lượng hay kết quả trực tiếp.

M2 chỉ áp dụng cho intent `multi_facet_event`. Nó không thay đổi planner của
`comparison`, `event_phases` hay F9 `historical_evolution`.

## 2. Hai nguồn mới đã được kiểm tra

Hai PDF mới đều ở trạng thái sẵn sàng và được RAG sử dụng trực tiếp:

| Nguồn | Trang nội dung | Chunks | Entities | Quan hệ |
| --- | ---: | ---: | ---: | ---: |
| 1945–1950 | 21–599 | 1.012 | 4.720 | 4.208 |
| 1951–1954 | 23–481 | 824 | 2.639 | 2.293 |

Việc kiểm tra trực tiếp nguồn `1951–1954` cho thấy kho đã có đủ bằng chứng:

- trang 393: Võ Nguyên Giáp và vai trò chỉ huy;
- trang 397: các đơn vị, lực lượng tham gia;
- trang 408–409: đợt 1, từ 13 đến 17/3/1954, Him Lam, Độc Lập, Bản Kéo;
- trang 412: đợt 2, từ 30/3 đến 26/4/1954, C1, D1, E1, D2, A1;
- trang 419–422: đợt 3, từ 1 đến 7/5/1954;
- trang 422: De Castries và bộ tham mưu bị bắt tại sở chỉ huy;
- trang 423: 56 ngày đêm, 16.200 quân và kết quả tổng hợp.

Vì vậy, câu trả lời thiếu trước đây chủ yếu do khâu chọn bằng chứng, không phải
do kho hoàn toàn thiếu dữ liệu.

## 3. Luồng M2

```text
Câu hỏi
  → chuẩn hóa câu chữ
  → nhận diện sự kiện và các facet được hỏi
  → QueryPlan multi_facet_event
  → hybrid retrieval thông thường
  → quét có giới hạn các chunk nhắc trực tiếp đúng sự kiện
  → chấm điểm và giữ bằng chứng riêng cho từng facet
  → giữ từng đợt/chặng có số thứ tự
  → kiểm tra facet còn thiếu và truy xuất bù
  → sinh câu trả lời theo đúng các ý người dùng hỏi
```

QueryPlan của ví dụ:

```json
{
  "intent": "multi_facet_event",
  "subject": "Điện Biên Phủ",
  "requiredFacets": ["progress", "forces", "leadership", "result"],
  "answerStructure": "multi_facet_event_sections"
}
```

## 4. Quét bằng chứng theo chủ thể

Các kênh hybrid đều có giới hạn số kết quả. Với sách dài, các chunk đầu tiên
nhắc “Điện Biên Phủ” có thể che mất các trang diễn biến nằm rất xa phía sau.
M2 bổ sung `retrieve_subject_evidence`: lấy một tập hữu hạn các chunk thực sự
chứa đúng tên sự kiện hoặc alias, sau đó xếp hạng facet tại backend.

Cơ chế này:

- không gọi thêm model OpenAI;
- không phát sinh token OpenAI cho bước quét;
- không đưa toàn bộ sách vào prompt;
- chỉ đưa các chunk tốt nhất sau khi đã chọn theo facet.

## 5. Chọn bằng chứng linh hoạt theo facet

- `progress`: giữ tối đa ba bằng chứng diễn biến trọng tâm. Nếu nguồn ghi các
  đợt/chặng bằng số thứ tự, hệ thống ưu tiên một chunk mạnh nhất cho mỗi đợt.
  Không giả định mọi sự kiện đều có đúng ba đợt.
- `leadership`: ưu tiên tên người đi cùng vai trò như tư lệnh, chỉ huy trưởng,
  tham mưu trưởng; tránh dùng câu nhận định chung thay cho người chỉ huy.
- `forces`: ưu tiên đơn vị hoặc lực lượng thực sự tham chiến và lực lượng hậu
  cần khi có dữ liệu.
- `result`: giữ hai loại bằng chứng nếu có: kết thúc quyết định tại sở chỉ huy
  (bị bắt, đầu hàng, tử trận) và kết quả tổng hợp (quân số, toàn thắng).
- `cause`, `significance`: chỉ được thêm khi người dùng thực sự hỏi.

Các đoạn chỉ nói về chuẩn bị chiến dịch không còn được dùng để thay thế toàn bộ
phần diễn biến. Các đoạn nói về hiệp định hoặc hệ quả ngoại giao cũng không
được dùng thay cho kết quả trực tiếp trên chiến trường.

## 6. Prompt, App và Web Admin

`MULTI_FACET_EVENT_PLANNING_M2` được tự động ghép vào Answer Planning đang
hoạt động. API quản lý prompt trả về chính chỉ dẫn hiệu lực này nên nội dung
Web Admin và quy tắc FastAPI thực thi vẫn đồng bộ.

App chỉ hiển thị câu trả lời lịch sử. Nguồn PDF, số trang, facet coverage và
diagnostics được trả riêng để Web Admin kiểm tra trong phần kết quả và bằng
chứng; chúng không được chèn vào câu trả lời cho người dùng.

## 7. Chi phí và cập nhật dữ liệu

Hai PDF đã index ở trạng thái “Sẵn sàng” nên không phải index lại để dùng M2.
Khi FastAPI khởi động lại, phiên bản
`multi-facet-event-subject-evidence-m2-f9-v28` có hiệu lực và tự dùng cả nguồn
cũ lẫn nguồn mới.

M2 chỉ tăng truy vấn Neo4j có giới hạn. Token OpenAI chỉ dùng cho embedding,
rerank/analysis và sinh câu trả lời như pipeline hiện có; subject scan không
gọi thêm OpenAI.

## 8. Kiểm thử hồi quy

Kiểm thử bao phủ:

- nhận diện đúng intent và subject của câu hỏi nhiều ý;
- mở rộng truy vấn theo từng facet;
- giữ các đợt diễn biến khác nhau thay vì chunk chuẩn bị;
- giữ cả kết thúc tại sở chỉ huy và kết quả tổng hợp;
- đồng bộ prompt M2;
- không làm thay đổi các planner comparison và F9.

Kết quả tự động: **117/117 test đạt**, `compileall` và `git diff --check`
không phát hiện lỗi.

## 9. Đối chiếu pipeline thật

Câu hỏi đã chạy lại trực tiếp qua Neo4j và OpenAI:

> Điện Biên Phủ diễn ra thế nào, ai chỉ huy, lực lượng nào tham gia và kết quả
> ra sao?

Diagnostics ghi nhận:

- intent `multi_facet_event`;
- đủ 4/4 facet: `progress`, `forces`, `leadership`, `result`;
- không còn facet thiếu;
- strategy có `subject_evidence_scan` và `facet_quota_gate`;
- citations được chọn ở các trang 393, 408, 412, 419, 422 và 423 của nguồn
  `1951-1954`.

Câu trả lời thực tế đã nêu đủ ba đợt và mốc ngày, Him Lam–Độc Lập–Bản Kéo,
Võ Nguyên Giáp, các đại đoàn tham chiến, 56 ngày đêm, con số 16.200 và việc
De Castries cùng bộ tham mưu bị bắt sống khi chiến dịch kết thúc.

## 10. Đồng bộ revision tự động

FastAPI, bộ kiểm tra `check-ai.sh`, script khởi động `start-ai.sh` và bộ tự
restart cùng đọc revision từ một nguồn duy nhất:
`SourceCode/src/api/rag_revision.txt`.

Web Admin không lưu bản sao revision và không so sánh bằng chuỗi hardcode nữa.
Trước mỗi lượt kiểm thử, Web gọi `/health/revision`, xác nhận FastAPI đang công
bố một revision hợp lệ rồi lưu revision thực tế vào bản ghi đánh giá. Endpoint
công bố cả revision của tiến trình đang chạy và revision hiện có trên ổ đĩa.
Nếu hai giá trị lệch nhau, Web chỉ yêu cầu restart FastAPI; nếu giống nhau,
kiểm thử tiếp tục bình thường. Vì vậy, khi backend nâng cấp về sau, không cần
cập nhật `AI_REQUIRED_RAG_REVISION` hoặc khởi động Web chỉ để đồng bộ một chuỗi
phiên bản.
