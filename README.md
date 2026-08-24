# Vietnam History AI — PDF GraphRAG

Backend FastAPI dùng chung cho app mobile và web-admin. Đây là pipeline tri thức
duy nhất của hệ thống:

```text
Upload PDF
  → đọc text / OCR
  → làm sạch và tách theo trang
  → chia AIChunk
  → OpenAI embedding
  → OpenAI trích Entity + relationship
  → lưu Neo4j
  → hybrid retrieval + trả lời có nguồn
```

## Thành phần

- `SourceCode/src/api/main.py`: API, job Index nền và endpoint quản trị.
- `SourceCode/src/api/pdf_ingestion.py`: đọc PDF, OCR và chia chunk.
- `SourceCode/src/api/graph_extraction.py`: OpenAI Structured Outputs cho entity
  và relationship.
- `SourceCode/src/api/neo4j_repository.py`: schema, lưu graph, tìm kiếm và usage log.
- `SourceCode/src/api/rag_service.py`: lớp tương thích và điều phối retrieval
  trong giai đoạn chuyển đổi.
- `SourceCode/src/api/rag/planning/`: phân rã trực tiếp câu hỏi thành các
  requirement nguyên tử; facet cũ chỉ là lớp tương thích.
- `SourceCode/src/api/rag/retrieval/`: lập truy vấn bù theo cụm field và giới
  hạn budget.
- `SourceCode/src/api/rag/evidence/`: kiểm định field hai tầng, ràng buộc chủ
  thể/vai trò và dựng Evidence Ledger.
- `SourceCode/src/api/rag/generation/`: sinh JSON claim–evidence nội bộ rồi
  render Markdown cho App/Web.
- `SourceCode/src/api/rag/verification/`: kiểm tra số liệu theo
  subject/predicate và loại claim không được hỗ trợ.
- `SourceCode/src/api/rag/orchestration/`: các use case điều phối từng giai
  đoạn; xem sơ đồ tại `SourceCode/src/api/rag/README.md`.
- `SourceCode/tests/`: kiểm thử chuẩn hoá entity, retrieval và serialization Neo4j.

Các script import JSON, Gemini, Streamlit và chatbot terminal cũ đã được gỡ bỏ.
Firestore chỉ giữ metadata/trạng thái PDF; Neo4j chỉ được cập nhật bởi backend này.

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/pip install -r SourceCode/requirements.txt
cp SourceCode/.env.example SourceCode/.env
```

Các biến quan trọng:

```env
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_ENTITY_MODEL=gpt-5.4-nano
AI_PIPELINE_VERSION=pdf-openai-v2

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j

AI_ADMIN_API_KEY=

# Kho JSON lịch sử đã được Admin duyệt và xuất bản qua Firebase Hosting.
AI_PUBLISHED_CONTENT_ENABLED=true
AI_PUBLISHED_CONTENT_MANIFEST_PATH=
AI_PUBLISHED_CONTENT_MANIFEST_URL=
AI_PUBLISHED_CONTENT_CACHE_TTL_SECONDS=300
```

`AI_PIPELINE_VERSION` quyết định khả năng tái sử dụng kết quả. Nếu PDF,
embedding model, entity model và version không đổi, lần Index lại dùng lại
embedding/entity theo `contentHash`, không gọi OpenAI lại cho chunk đó.

Chatbot truy xuất đồng thời hai kho có kiểm soát: PDF đã index trong Neo4j và
JSON lịch sử đã xuất bản. JSON chỉ được đọc qua `content/manifest.json`; backend
kiểm tra SHA-256 trước khi dùng, tự làm mới theo TTL và giữ snapshot hợp lệ gần
nhất nếu Hosting tạm thời không truy cập được. Cơ chế này giúp những sự kiện đã
có trên App (ví dụ giai đoạn chưa có PDF tương ứng) vẫn trở thành bằng chứng
RAG mà không hardcode dữ kiện trong mã nguồn.

## Chạy

Từ thư mục gốc:

```bash
./start-ai.sh
```

Hoặc:

```bash
cd SourceCode
../.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Android Emulator: `http://10.0.2.2:8000`

### Tự nạp revision RAG mới khi hệ thống rảnh

Chạy một lần để cài FastAPI service và timer theo dõi revision:

```bash
./install-ai-service.sh
```

Timer kiểm tra mỗi 30 giây. Revision được đọc từ nguồn duy nhất
`SourceCode/src/api/rag_revision.txt`. Khi code yêu cầu một revision mới, timer chỉ
restart FastAPI sau khi không còn job Index ở trạng thái
`queued/running/stopping`, không còn request Chat đang xử lý và trạng thái rảnh
được giữ ít nhất 45 giây. Kiểm tra bằng:

```bash
systemctl --user status history-chatbot-ai-update.timer
./check-ai.sh
```

Revision của tiến trình và revision trên ổ đĩa được trả trong `/health` qua
`rag_revision`, `source_rag_revision` và `restart_required`. Web Admin đọc trực
tiếp các trường này, không giữ một bản revision hardcode riêng.

## Luồng Index và token

Khi quản trị viên bấm **Index**:

1. Backend đọc PDF; nếu không có text thì dùng OCRmyPDF hoặc
   Poppler + Tesseract.
2. Nội dung được làm sạch, tách theo trang và chia `AIChunk`.
3. `text-embedding-3-small` tạo vector cho chunk mới hoặc đã thay đổi.
4. `gpt-5.4-nano` trích các loại entity:
   `PERSON`, `LOCATION`, `ORGANIZATION`, `EVENT`, `PERIOD`, `DOCUMENT`,
   `STATE`, `WEAPON`.
5. Bộ resolver hợp nhất alias đã kiểm chứng, loại ngày/số lượng/mention chung,
   đồng thời giữ `surfaceForms` theo từng chunk để truy vết cách viết gốc.
6. Backend chỉ giữ quan hệ thuộc ontology cố định và có hai đầu entity hợp lệ;
   predicate gốc được lưu trong `rawPredicate`.
7. Một transaction thay thế dữ liệu của PDF trong Neo4j:
   `KnowledgeSource → KnowledgePage → AIChunk → Entity`.
8. `AIUsageLog` lưu model, input/output token, số chunk gọi OpenAI, số chunk
   tái sử dụng, số entity và số relationship.

Web-admin hiển thị token của lần Index gần nhất trong **Kho tri thức PDF** và
tổng token Index trong **Quản lý Graph**. Dashboard OpenAI vẫn là nguồn đối
soát chi phí chính thức.

Index lần đầu cho các PDF cũ sẽ tạo lại entity/relationship và phát sinh token.
Hệ thống không tự chạy hàng loạt; quản trị viên chủ động bấm Index cho từng PDF.

## Bổ sung facet cho chunk đã Index

Pipeline gắn facet lịch sử (chính trị, kinh tế, văn hóa, xã hội, mục tiêu/hệ quả,
diễn biến...) ngay khi tạo chunk. Với dữ liệu đã Index trước khi có cơ chế này,
có thể cập nhật metadata trực tiếp từ nội dung đang lưu trong Neo4j:

```bash
cd SourceCode
../.venv/bin/python scripts/backfill_chunk_facets.py --dry-run
../.venv/bin/python scripts/backfill_chunk_facets.py
```

Backfill chỉ chạy bộ quy tắc cục bộ và cập nhật thuộc tính `facets` của
`AIChunk`; không tạo embedding, không gọi model OpenAI và không tốn token.

## Kiểm thử

```bash
PYTHONPATH=SourceCode .venv/bin/python -m unittest discover -s SourceCode/tests -v
```

## Tài liệu

- [Báo cáo cải tiến truy xuất tên riêng](BAO_CAO_CAI_TIEN_TRUY_XUAT_TEN_RIENG.md)
- [Báo cáo Comparison Planner ba tầng v22](BAO_CAO_COMPARISON_PLANNER_V22.md)
- [Báo cáo Evidence-Verified Evolution Planner F9 v25](BAO_CAO_EVIDENCE_VERIFIED_EVOLUTION_F9_V25.md)
- [Báo cáo refactor RAG strict-corpus v29](BAO_CAO_RAG_REFACTOR_STRICT_CORPUS_V29.md)

Báo cáo kiến trúc tổng hợp và lộ trình v30 được lưu tại `../BaoCao.md` trong
workspace đồ án để dùng chung khi viết báo cáo hệ thống.
