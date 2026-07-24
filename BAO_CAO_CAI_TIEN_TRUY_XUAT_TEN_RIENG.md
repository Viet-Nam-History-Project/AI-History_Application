# Báo cáo cải tiến truy xuất tên riêng cho trợ lý Lịch sử Việt Nam

Tài liệu này dành cho thành viên mới của nhóm. Bạn không cần biết sâu về AI hoặc
Neo4j trước khi đọc. Mục tiêu là hiểu vì sao hệ thống từng trả lời yếu với câu
`Nguyễn Thị Định là ai?`, phần nào đã được sửa và cách tự kiểm tra sau khi pull
code.

## 1. Tóm tắt thay đổi

Trước đây backend chủ yếu tìm tài liệu bằng hai cách:

1. Vector search: tìm đoạn có ý nghĩa gần với câu hỏi.
2. Full-text search: tìm các từ xuất hiện trong đoạn văn.

Hai cách này hữu ích với câu hỏi dài, nhưng có thể bỏ sót một tên riêng. Ví dụ
PDF `1945-1965` có đoạn trực tiếp nói về Nguyễn Thị Định ở trang 259, nhưng kết
quả chung chung vẫn có thể đứng cao hơn vì chứa nhiều từ như "nhân vật", "lịch
sử", "kháng chiến". Reranker sau đó chỉ nhìn tập ứng viên chưa tốt nên không thể
tự lấy lại đoạn đã bị bỏ sót.

Phiên bản mới bổ sung một làn truy xuất chính xác cho tên riêng:

- Nhận ra chủ thể chính trong câu hỏi, ví dụ `Nguyễn Thị Định`.
- Tìm cụm tên đầy đủ trong `heading + text` của chunk.
- Chạy đồng thời exact phrase, vector và full-text search.
- Ưu tiên bằng chứng trực tiếp khi gộp thứ hạng.
- Không cho reranker vô tình loại hết bằng chứng trực tiếp.
- Tính độ tin cậy từ nhiều tín hiệu thay vì chỉ tin một con số của model.

Đây là cải tiến RAG, không phải huấn luyện lại mô hình ngôn ngữ. PDF và dữ liệu
Neo4j hiện có vẫn được giữ nguyên.

## 2. Luồng hoạt động mới

```text
Câu hỏi người dùng
        |
        v
Phân tích câu hỏi, lấy tên/chủ thể chính
        |
        v
+----------------+----------------+----------------+
| Exact phrase   | Vector search  | Full-text      |
| Nguyễn Thị Định| Gần nghĩa      | Khớp từ khóa   |
+----------------+----------------+----------------+
        |                |                |
        +----------------+----------------+
                         |
                         v
             Gộp hạng có trọng số
                         |
                         v
       Giữ lại bằng chứng trực tiếp rồi rerank
                         |
                         v
       Tính confidence và tạo câu trả lời có nguồn
```

### 2.1. Phân tích câu hỏi

File `SourceCode/src/api/query_analysis.py` nhận biết các mẫu hỏi thường gặp mà
không cần gọi thêm model AI:

- `Nguyễn Thị Định là ai?` -> chủ thể `Nguyễn Thị Định`.
- `Hãy cho tôi biết Nguyễn Thị Định là ai` -> vẫn lấy đúng tên.
- `Chiến dịch Điện Biên Phủ diễn ra như thế nào?` -> chủ thể
  `Chiến dịch Điện Biên Phủ`, dù câu hỏi không ghi năm 1954.
- `Ai là nhân vật lịch sử?` -> không coi một cụm chung chung là tên riêng.

Kết quả phân tích được lưu trong `QuerySignals` gồm:

- `primary_entity`: chủ thể chính.
- `exact_phrases`: các cụm cần tìm nguyên vẹn.
- `is_identity_query`: có phải dạng câu hỏi "X là ai" hay không.

### 2.2. Ba kênh truy xuất

File `SourceCode/src/api/neo4j_repository.py` chạy ba kênh:

| Kênh | Dùng để làm gì | Trọng số khi gộp |
| --- | --- | ---: |
| Exact phrase | Giữ đoạn có nguyên cụm tên/chủ thể | 4 |
| Vector | Tìm đoạn gần nghĩa với câu hỏi | 1 |
| Full-text | Tìm đoạn khớp từ khóa | 1 |

Exact phrase dùng full-text index để lấy ứng viên trước, sau đó xác nhận lại bằng
điều kiện `CONTAINS` trên nội dung. Vì vậy một đoạn chỉ chứa các từ rời rạc
`Nguyễn`, `Thị`, `Định` ở vị trí không liên quan sẽ không được coi là bằng chứng
trực tiếp.

Mỗi kết quả có thêm metadata:

- `channels`: kết quả đến từ kênh nào.
- `directEvidence`: có chứa cụm bằng chứng trực tiếp hay không.
- `entityMatch`: có khớp đúng chủ thể chính hay không.
- `matchedPhrases`: danh sách cụm đã khớp.

### 2.3. Không làm mất bằng chứng khi rerank

File `SourceCode/src/api/rag_service.py` vẫn cho model rerank các ứng viên. Tuy
nhiên, nếu model chỉ chọn các đoạn chung chung, backend tự đưa bằng chứng trực
tiếp phù hợp trở lại danh sách cuối. Đây là hàng rào bảo vệ, không phải ép model
phải chọn một câu trả lời cố định.

### 2.4. Độ tin cậy mới

Confidence được tính từ sáu yếu tố:

| Yếu tố | Trọng số | Ý nghĩa |
| --- | ---: | --- |
| Chất lượng truy xuất | 30% | Điểm của các đoạn sau rerank |
| Độ phủ bằng chứng | 30% | Bằng chứng có trả lời đủ ý hỏi không |
| Bằng chứng trực tiếp | 20% | Có đoạn chứa tên/chủ thể đầy đủ không |
| Khớp đúng thực thể | 10% | Đoạn có nói trực tiếp về chủ thể không |
| Độ tin cậy nguồn | 5% | Tỷ lệ nguồn chính thống/được duyệt |
| Đa dạng trang | 5% | Bằng chứng có đến từ nhiều trang không |

Với câu hỏi "X là ai", nếu không có `directEvidence` và `entityMatch`, confidence
bị giới hạn tối đa 45%. Nhờ vậy hệ thống không còn báo tin cậy 98% trong khi chỉ
trả lời rằng nhân vật "được nhắc đến trong bối cảnh lịch sử".

## 3. Các file liên quan

| File | Vai trò |
| --- | --- |
| `SourceCode/src/api/query_analysis.py` | Tách chủ thể/tên riêng từ câu hỏi |
| `SourceCode/src/api/neo4j_repository.py` | Truy vấn exact, vector, full-text và gộp kết quả |
| `SourceCode/src/api/rag_service.py` | Mở rộng query, rerank, confidence và sinh câu trả lời |
| `SourceCode/src/api/models.py` | Khai báo diagnostics trả về web-admin |
| `SourceCode/tests/test_query_analysis.py` | Kiểm thử nhận diện chủ thể |
| `SourceCode/tests/test_rag_service.py` | Kiểm thử giữ bằng chứng và confidence |

## 4. Cách chạy sau khi pull code

### Bước 1: chuẩn bị `.env`

Mỗi thành viên tự nhận file `.env` từ trưởng nhóm và đặt tại
`History-Chatbot/SourceCode/.env`. Không commit file này vì nó chứa khóa Neo4j,
model và Firebase.

### Bước 2: khởi động backend

Từ thư mục gốc `History-Chatbot`:

```bash
./start-ai.sh
```

Backend chạy tại `http://127.0.0.1:8000`. Kiểm tra nhanh bằng:

```bash
./check-ai.sh
```

Nếu dùng Windows và không chạy được file `.sh`, mở PowerShell trong thư mục
`SourceCode`, kích hoạt môi trường Python rồi chạy:

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Bước 3: chạy kiểm thử tự động

Trên Linux/macOS:

```bash
PYTHONPATH=SourceCode .venv/bin/python -m unittest discover -s SourceCode/tests
```

Kết quả đúng tại thời điểm bàn giao:

```text
Ran 20 tests
OK
```

### Bước 4: kiểm thử trên web-admin

Vào `Prompt & Kiểm thử`, lần lượt hỏi:

```text
Nguyễn Thị Định là ai?
```

```text
Chiến dịch Điện Biên Phủ diễn ra như thế nào, gồm lực lượng gì,
ai là người lãnh đạo và có kết quả ra sao?
```

Câu thứ hai cố ý không ghi năm. Hệ thống vẫn phải nhận ra chủ thể chính và tìm
được tài liệu liên quan.

## 5. Cách đọc diagnostics

API debug và lịch sử đánh giá có thể trả các trường sau:

| Trường | Kết quả tốt |
| --- | --- |
| `primary_entity` | Đúng tên người hoặc sự kiện đang hỏi |
| `exact_match_count` | Lớn hơn 0 nếu PDF có nhắc trực tiếp |
| `channels` | Có `exact`, thường kèm `vector` hoặc `lexical` |
| `confidence_factors.direct_evidence` | Bằng 1 khi có bằng chứng trực tiếp |
| `confidence_factors.entity_match` | Bằng 1 khi đúng chủ thể |
| `confidence` | Cao khi bằng chứng vừa đúng vừa đủ |

Nếu PDF chắc chắn có tên nhưng `exact_match_count = 0`:

1. Mở **Kho tri thức PDF** và kiểm tra tài liệu đã ở trạng thái **Sẵn sàng**.
2. Mở **Kiểm tra** và tìm tên trong nội dung chunk.
3. Nếu không tìm thấy, PDF có thể OCR sai tên; cần sửa nguồn/OCR rồi Index lại.
4. Nếu tìm thấy trong chunk mà exact vẫn bằng 0, kiểm tra full-text index và log
   của FastAPI.

## 6. Dữ liệu Neo4j có bị thay đổi không?

Thay đổi này chỉ bổ sung cách đọc và xếp hạng dữ liệu. Nó không tự xóa node,
không đổi relationship và không tạo bản sao PDF.

- Đổi metadata như tiêu đề tài liệu không tự đổi nội dung chunk.
- Bấm **Index lại** sẽ tạo lại trang/chunk của đúng nguồn đó theo quy trình hiện
  có.
- Exact phrase hoạt động trên `AIChunk.heading` và `AIChunk.text` đã index.

## 7. Giới hạn còn lại và hướng nâng cấp

Bản sửa hiện tại giải quyết lỗi bỏ sót bằng chứng trực tiếp mà không cần migration
toàn bộ graph. Các bước tiếp theo nên làm theo thứ tự:

1. Chuẩn hóa `Entity` và `aliases`, ví dụ Nguyễn Thị Định không bị tách thành
   nhiều node do khác cách viết.
2. Tạo quan hệ `AIChunk -> MENTIONS -> Entity` để truy xuất theo thực thể nhanh
   và dễ giải thích hơn.
3. Chuẩn hóa relationship đồng nghĩa bằng một từ điển quan hệ có kiểm duyệt.
4. Tách các mệnh đề quan trọng thành `Claim` và nối `SUPPORTED_BY` tới trang
   nguồn.
5. Tạo bộ câu hỏi vàng cho từng thời kỳ, chấm Recall@K, độ chính xác nguồn và
   mức đầy đủ của câu trả lời trước mỗi lần release.

Không nên tự động gộp/xóa hàng loạt Entity chỉ bằng tên gần giống. Lịch sử có
nhiều người, địa danh và tổ chức trùng hoặc gần tên; mọi migration graph phải có
bản xem trước và khả năng rollback.

## 8. Checklist trước khi merge

- [ ] Không commit `.env`, khóa Firebase, khóa model hoặc mật khẩu Neo4j.
- [ ] Chạy đủ unit test và thấy `OK`.
- [ ] `./check-ai.sh` báo backend và Neo4j hoạt động.
- [ ] Hỏi thử một tên người có trong PDF và một sự kiện không ghi năm.
- [ ] Kiểm tra nguồn trả về đúng tài liệu và đúng trang.
- [ ] Kiểm tra câu hỏi không có bằng chứng không được báo confidence cao.
- [ ] Ghi lại thay đổi retrieval/prompt trong PR để thành viên khác biết có cần
  Index lại hay không.

## 9. Kết luận ngắn cho người mới

Model không thể trả lời từ một đoạn mà hệ thống không đưa cho nó. Vì vậy chất
lượng chatbot phụ thuộc trước hết vào retrieval. Bản sửa này giúp retrieval hiểu
"đang hỏi về ai/cái gì", ưu tiên đoạn nhắc trực tiếp chủ thể và chỉ báo tin cậy
cao khi có bằng chứng tương ứng. Prompt vẫn quan trọng cho cách trình bày, nhưng
prompt không thể thay thế dữ liệu và truy xuất đúng.
