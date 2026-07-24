# Báo cáo Evidence-Verified Evolution Planner F9 v25

## 1. Phạm vi

Phiên bản này chỉ áp dụng cho nhóm câu hỏi F9 **“tiếp nối và thay đổi”**:

- thay đổi qua thời gian;
- tiếp nối và biến đổi;
- bước ngoặt;
- mức độ thay đổi;
- nguyên nhân chuyển đổi;
- trước – sau;
- tăng tốc, chậm lại hoặc đảo chiều;
- kế thừa và phát triển;
- phân kỳ một tiến trình.

Các planner dành cho câu hỏi so sánh, diễn biến sự kiện, nhân vật, nguyên
nhân – kết quả và các intent khác không bị thay đổi.

## 2. Vấn đề được phát hiện

Trong lượt hỏi “Chính sách cai trị của Pháp thay đổi như thế nào từ 1897 đến
1945?”, period planner v24 tạo bốn chặng:

1. 1897–1918;
2. 1919–1930;
3. 1930–1936;
4. 1936–1945.

Chặng cuối đã gộp hai trạng thái trái chiều: nhượng bộ hạn chế trong thời kỳ
Mặt trận Nhân dân và sự chuyển sang siết chặt, huy động thời chiến. Một lỗi
nghiêm trọng hơn xuất hiện ở khâu sinh câu trả lời: AI viết rằng Chính phủ Mặt
trận Nhân dân xóa bỏ cải cách và hủy tuần làm 40 giờ.

Đoạn dữ liệu thực tế lại viết:

> Chính phủ Pháp ngả hẳn sang hữu, xóa bỏ dần những “cải cách” mà Mặt trận
> Nhân dân tiến hành...

Như vậy, dữ liệu đúng nhưng câu trả lời đã đảo **chủ thể hành động** với
**đối tượng bị xóa bỏ**.

## 3. Kết quả rà soát dữ liệu

Neo4j đang có dữ liệu cho các chặng quan trọng, gồm:

- cải cách và tuần làm 40 giờ thời Mặt trận Nhân dân;
- Chính phủ Pháp ngả sang hữu và xóa bỏ dần các cải cách;
- cuộc khai thác thuộc địa lần thứ hai;
- huy động nhân lực và kinh tế cho chiến tranh;
- Nhật đảo chính Pháp ngày 9/3/1945.

Vấn đề chính là truy xuất và phân kỳ chưa chọn đồng thời đủ các bằng chứng đối
lập. Planner cũ còn có thể xem khoảng năm của sách/PDF là tín hiệu bước ngoặt,
trong khi đó chỉ là metadata của nguồn.

## 4. Nguyên nhân kỹ thuật

1. Evidence card trộn năm xuất hiện trong nội dung với `yearStart/yearEnd` của
   nguồn, nên một sách bao quát 1897–1945 có thể bị hiểu nhầm là bằng chứng cho
   ranh giới lịch sử.
2. Mỗi giai đoạn chỉ cần một evidence ID. Một đoạn hẹp về hội đồng, nghị định
   hoặc thiết chế có thể bị nâng thành đặc trưng của toàn thời kỳ.
3. Planner chưa bắt buộc mô hình hóa đầy đủ:
   - thời điểm;
   - chủ thể;
   - hành động;
   - trạng thái trước và sau.
4. `boundary_cause` là câu do model tự tóm tắt, không bị khóa vào một đoạn
   nguyên văn. Vì vậy có thể đảo sai vai nghĩa dù các từ đều có trong chunk.
5. Không có quy tắc tách chặng khi xuất hiện một đảo chiều chính sách.
6. Ngân sách mặc định tám chunk có thể không đủ để giữ cả bằng chứng bước ngoặt
   và bằng chứng mô tả trạng thái cho năm hoặc sáu chặng.

## 5. Thiết kế v25

### 5.1. Phân biệt năm nội dung và năm metadata

Evidence card có hai nhóm riêng:

- `explicit_content_years`: năm thực sự xuất hiện trong heading/text;
- `metadata_year_start`, `metadata_year_end`: khoảng của nguồn.

Chỉ `explicit_content_years` được dùng để xác minh một bước ngoặt. Metadata
chỉ hỗ trợ lấy mẫu ứng viên trên toàn khoảng hỏi.

### 5.2. Đánh giá evidence card

Mỗi card có:

- `subject_relevance`;
- `boundary_strength`;
- `boundary_eligible`;
- điểm chi tiết cụ thể;
- điểm retrieval ban đầu.

Card chỉ đủ điều kiện làm bằng chứng ranh giới khi có năm nội dung, có tín
hiệu chuyển trạng thái và liên quan trực tiếp đến subject.

### 5.3. Cấu trúc bước ngoặt có kiểm chứng

Mỗi period mới mang các trường:

- `dominant_state`;
- `boundary_year`;
- `boundary_actor`;
- `boundary_action`;
- `boundary_evidence_id`;
- `boundary_excerpt`;
- `evidence_ids`.

`boundary_excerpt` phải là đoạn nguyên văn nằm trong chính chunk đã khai báo.
Backend kiểm tra lại quan hệ substring sau chuẩn hóa. Với mọi chặng sau chặng
đầu, `boundary_year` phải xuất hiện trực tiếp trong chunk và gần ranh giới bắt
đầu của chặng.

Sau kiểm tra hình thức, một verifier riêng dùng model nhỏ
`OPENAI_ENTITY_MODEL` đọc đồng thời claim và excerpt để kiểm tra semantic
role. Nếu claim đảo chủ thể, verifier sửa actor/action/cause từ chính excerpt;
nếu excerpt không đủ căn cứ thì period plan bị từ chối thay vì suy đoán.

### 5.4. Quy tắc tách đảo chiều tổng quát

Planner phải tách chặng khi có bằng chứng về hai trạng thái chủ đạo đối lập,
ví dụ:

- nhượng bộ/nới lỏng ↔ siết chặt/đàn áp;
- trực tiếp ↔ gián tiếp;
- tập trung ↔ phân quyền;
- mở rộng ↔ thu hẹp;
- thời bình ↔ thời chiến;
- tăng trưởng ↔ khủng hoảng/suy giảm.

Đây là quy tắc theo **trạng thái của domain**, không phải danh sách mốc năm
được viết riêng cho lịch sử thuộc địa Pháp. Vì vậy nó dùng được cho tiến trình
quân sự, chính trị, kinh tế, ngoại giao, xã hội và phong trào.

### 5.5. Phân biệt bằng chứng vĩ mô và minh họa

Nhãn giai đoạn và `dominant_state` phải dựa trên bằng chứng có phạm vi vĩ mô.
Hội đồng, sắc lệnh, trận đánh, thiết chế hoặc biện pháp riêng chỉ được dùng làm
minh họa trừ khi chính nó tạo ra bước ngoặt của subject.

### 5.6. Truy xuất hai lớp

F9 thực hiện:

1. tìm subject, trạng thái đầu/cuối, dấu hiệu chuyển đổi và các cặp trạng thái
   đối lập;
2. tạo period plan có bằng chứng;
3. truy xuất lần hai theo từng period, `dominant_state`, actor và action;
4. khóa trước các boundary chunk đã xác minh;
5. bổ sung state evidence cho từng giai đoạn;
6. rerank và sinh câu trả lời.

Sau khi biết số period, `top_k` F9 được điều chỉnh tối đa 12 chunk để có thể
giữ cả hai loại bằng chứng.

## 6. Cách ngăn lỗi đảo chủ thể

Model sinh câu trả lời nhận trực tiếp:

- năm chuyển đổi;
- chủ thể;
- hành động;
- nguyên nhân;
- đoạn bằng chứng nguyên văn.

Prompt bắt buộc bảo toàn quan hệ chủ thể – hành động – đối tượng. Ví dụ, từ
đoạn “Chính phủ Pháp ngả sang hữu, xóa bỏ các cải cách mà Mặt trận Nhân dân
tiến hành”, AI phải hiểu:

- chủ thể xóa bỏ: Chính phủ Pháp đã ngả sang hữu;
- đối tượng bị xóa bỏ: các cải cách;
- bên từng tiến hành cải cách: Mặt trận Nhân dân.

Việc này không chỉ dựa vào prompt sinh câu trả lời: bước kiểm tra semantic
role chạy trước generation và khóa lại actor/action đã được sửa.

## 7. Khả năng áp dụng cho câu hỏi khác

Hệ thống không chứa điều kiện kiểu “nếu câu hỏi nói về Pháp thì tách năm 1936
và 1939”. Period vẫn do model tạo từ evidence. Backend chỉ kiểm tra các thuộc
tính tổng quát:

- ranh giới có năm trực tiếp hay không;
- có liên quan đến subject hay không;
- actor/action/excerpt có đầy đủ hay không;
- excerpt có thật trong chunk hay không;
- các period có liên tục, không chồng lấn lớn và bao phủ khoảng hỏi hay không;
- có đảo chiều trạng thái đáng kể bị gộp hay không.

## 8. Quan sát trên Web Admin

Diagnostics và `AIQueryLog` bổ sung:

- `evolutionPeriodStates`;
- `evolutionBoundaryYears`;
- `evolutionBoundaryActors`;
- `evolutionBoundaryActions`;
- `evolutionBoundaryEvidenceIds`.

Admin có thể đối chiếu period plan với danh sách chunk được chọn để xác định
lỗi nằm ở retrieval, planner hay generation.

## 9. Kiểm thử

- 104 kiểm thử backend: đạt.
- Web Admin TypeScript typecheck: đạt.
- Web Admin ESLint: đạt.
- FastAPI health:
  `evidence-verified-evolution-planning-f9-v25`.
- Neo4j: kết nối thành công.

Hai regression test mới bảo đảm:

1. khoảng năm trong tên/metadata của nguồn không thể tự trở thành bằng chứng
   bước ngoặt;
2. verifier sửa được lỗi đảo chủ thể trong claim bước ngoặt;
3. prompt active tự loại bỏ rule F9 v24 và chỉ giữ một rule F9 v25.
