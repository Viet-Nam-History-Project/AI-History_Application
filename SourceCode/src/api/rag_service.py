import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import replace

from openai import OpenAI

from .config import get_settings
from .entity_resolution import (
    expand_known_entity_aliases,
    known_entity_year_range,
)
from .graph_extraction import TokenUsage
from .models import ChatRequest, ChatResponse, Citation, RetrievalDiagnostics
from .neo4j_repository import Neo4jKnowledgeRepository
from .history_facets import (
    COMPARISON_DOMAIN_FACETS,
    EVOLUTION_DOMAIN_FACETS,
    HISTORY_FACETS,
    comparison_facets_for_domain,
    evolution_facets_for_domain,
    facet_detail_search_text,
    facet_label,
    text_supports_facet,
)
from .query_analysis import (
    QueryNormalization,
    QueryPlan,
    QuerySignals,
    analyze_query,
    build_query_plan,
    normalize_query,
)


logger = logging.getLogger(__name__)


PRESENTATION_INSTRUCTION = """
Tự nhận biết yêu cầu trình bày trong câu hỏi. Nếu người dùng yêu cầu ngắn gọn,
hãy cô đọng; nếu yêu cầu chi tiết, hãy giải thích đầy đủ; nếu yêu cầu kể lại,
hãy trình bày như một câu chuyện lịch sử theo trình tự. Nếu không có yêu cầu rõ
ràng, trả lời vừa đủ, dễ đọc và ưu tiên trực tiếp vào câu hỏi. Không sáng tác
lời thoại, cảm xúc, con số hay tình tiết không có trong nguồn.

QUY TẮC TRÌNH BÀY:
- Trả lời trực tiếp và chọn lọc, không biến mọi tên riêng xuất hiện trong nguồn thành đáp án.
- Với câu hỏi liệt kê nhân vật hoặc sự kiện, hãy phân nhóm và nêu vai trò thay vì chép lại nguồn.
- Dùng Markdown chuẩn: tiêu đề ngắn, đoạn văn thoáng, danh sách hoặc bảng khi giúp so sánh.
- Dùng bảng chỉ khi có nhiều mục cùng trường thông tin và vẫn dễ đọc trên màn hình hẹp.
- Không dùng một khối văn bản quá dài; ưu tiên phân nhóm thông tin.
- Không mở đầu bằng lời xã giao dài dòng và không lặp lại toàn bộ câu hỏi.
- Kết thúc bằng một nhận định ngắn nếu nó giúp người học ghi nhớ.
""".strip()

OUTPUT_CONTRACT = """
QUY TẮC NGUỒN THAM KHẢO BẮT BUỘC:
- Dùng các nguồn được đánh số để kiểm chứng nội bộ, nhưng không chèn số nguồn vào văn bản trả lời.
- Tuyệt đối không viết các ký hiệu như [1], [2], [1][3], [1, 3] hoặc chú thích đánh số.
- Không tự tạo mục "Nguồn tham khảo" trong câu trả lời. Ứng dụng sẽ hiển thị
  danh sách nguồn riêng từ dữ liệu citations do backend cung cấp.
""".strip()

QUERY_NORMALIZATION_INSTRUCTION = """
QUY TẮC HIỂU CÂU HỎI:
- Trả lời theo câu hỏi đã chuẩn hóa bên dưới, không lặp lại lỗi gõ của người dùng
  trong tiêu đề hoặc nội dung.
- Âm thầm sửa lỗi chính tả, từ lặp và viết tắt khi cách hiểu đã rõ.
- Không tự ý đổi tên người, địa danh, tổ chức hoặc mốc thời gian.
- Câu hỏi chuẩn hóa chỉ được sửa cách diễn đạt; không thêm phạm vi,
  mốc thời gian hoặc kiến thức suy ra. Ghi phần suy ra vào trường phạm vi.
- Nếu một từ viết tắt hoặc lỗi gõ có thể làm thay đổi ý nghĩa lịch sử, nêu rõ
  điểm mơ hồ và hỏi lại thay vì đoán.
""".strip()

ANSWER_PLANNING_INSTRUCTION = """
QUY TẮC LẬP CÂU TRẢ LỜI:
- Bám theo kế hoạch facet đã cung cấp; không tự thu hẹp câu hỏi tổng quan thành
  một dòng thời gian hoặc một giai đoạn chỉ vì nguồn đầu tiên nói về giai đoạn đó.
- Trình bày các phương diện bắt buộc trước. Phần thay đổi theo thời gian chỉ đặt
  sau phần nội dung chính, trừ khi intent là historical_evolution.
- Chỉ viết nội dung được hỗ trợ bởi NGUỒN TRUY XUẤT. Facet còn thiếu phải được
  nói rõ là kho tài liệu chưa đủ bằng chứng, không tự bổ sung kiến thức bên ngoài.
- Không ghi tên tài liệu, số trang hoặc danh sách nguồn vào phần văn bản trả lời;
  metadata này được trả riêng để Web Admin hiển thị khi đánh giá.
- Không dùng ngôn ngữ hậu trường như "theo nguồn tài liệu", "nguồn
  cho thấy", "tài liệu nêu rõ" hoặc "các nguồn được cung cấp". Hãy
  trình bày trực tiếp thành kiến thức lịch sử tự nhiên, ví dụ: "Có nhiều
  loại thuế...". Việc kiểm chứng và hiển thị nguồn do backend/Web Admin
  thực hiện riêng.
- Không mở đầu bằng "Nếu hiểu câu hỏi..." khi kế hoạch xác định không mơ hồ.
- Dùng sắc thái thận trọng: nhượng bộ hoặc nới lỏng hạn chế không đồng nghĩa với
  bình đẳng hay chấm dứt nền thống trị thuộc địa.
- Với câu hỏi giải thích, tổng quan, phân tích hoặc so sánh: mỗi ý lớn
  nên gắn với 1–2 minh họa cụ thể nếu NGUỒN TRUY XUẤT có. Minh họa có
  thể là tên nhân vật, sự kiện, địa danh, cơ quan, biện pháp, mốc thời
  gian hoặc số liệu. Không biến câu trả lời thành danh sách ví dụ rời rạc.
- Linh hoạt theo dạng câu hỏi: diễn biến cần mốc thời gian/địa điểm; nhân vật
  cần hành động/vai trò; chính sách cần biện pháp/ví dụ/tác động; nguyên
  nhân–kết quả cần dẫn chứng thể hiện quan hệ. Không ép mọi câu trả lời vào
  một bố cục cố định.
- Với câu trả lời có nhiều mục, hãy ưu tiên minh họa khác loại thay vì
  lặp lại một kiểu: ít nhất một tên riêng hoặc thiết chế, một mốc thời gian
  và một biện pháp/số liệu nếu các chi tiết đó rõ ràng trong nguồn.
- Chỉ nói kho tài liệu thiếu chi tiết sau khi kế hoạch cho biết phương diện đó
  thực sự không có bằng chứng cụ thể; không dùng câu này để thay cho việc
  khai thác các nguồn chi tiết đã được cung cấp.
- Với intent event_phases hoặc cấu trúc phases_with_dates_then_result:
  trình bày từng đợt/giai đoạn theo đúng thứ tự; mỗi mục cần nêu khoảng thời
  gian bắt đầu–kết thúc, mục tiêu hoặc diễn biến chính, địa điểm/bước ngoặt
  và nhân vật then chốt nếu NGUỒN TRUY XUẤT có bằng chứng. Sau các đợt phải
  có phần kết quả chung, ngày kết thúc và số phận/hành động của người chỉ huy
  hoặc nhân vật quyết định nếu nguồn có. Không bỏ mốc cụ thể để chỉ viết
  "mở đầu", "sau đó", "cuối cùng". Nếu nguồn thiếu ngày hoặc nhân vật, nói
  ngắn gọn phần còn thiếu thay vì tự điền bằng kiến thức ngoài nguồn.
- Với intent comparison hoặc cấu trúc comparison_matrix_then_similarities_differences:
  tuân theo comparisonIntent và answerRequirements. Tiêu chí người dùng nêu rõ
  có ưu tiên cao nhất; không tự mở rộng thành một bảng toàn diện khi người dùng
  chỉ hỏi một điểm giống, một khác biệt hoặc vài phương diện cụ thể. Chỉ đưa
  facet bổ sung vào câu trả lời khi có bằng chứng cho cả hai phía. Facet do
  người dùng yêu cầu nhưng thiếu một phía phải được báo ngắn gọn, không âm thầm
  loại bỏ. Chỉ dùng bảng cho so sánh nhiều tiêu chí; câu hỏi về khác biệt chính,
  mức độ, hiệu quả hoặc nhận định phải trả lời trực tiếp và có lập luận.
""".strip()

MANDATORY_COMPARISON_PLANNING_INSTRUCTION = """
- ADAPTIVE_COMPARISON_PLANNING_V22:
- Quy tắc này thay thế mọi quy tắc comparison cũ: tuân theo comparisonIntent,
  explicitFacets, answerRequirements và ma trận bằng chứng được backend cung cấp.
- Không coi taxonomy là checklist. Không thêm phương diện ngoài yêu cầu chỉ để
  tạo bảng dài; không tạo hàng không liên quan hoặc hàng chỉ báo thiếu dữ liệu.
- Facet bổ sung chỉ được trình bày khi có bằng chứng cho cả hai đối tượng. Facet
  người dùng nêu rõ không được âm thầm bỏ; nếu thiếu một phía, báo đúng phần thiếu.
- Chỉ dùng bảng cho phép so sánh nhiều tiêu chí. main_similarity,
  main_difference, extent_comparison, effectiveness_comparison và judgement
  cần trả lời trực tiếp theo đúng trọng tâm, có giải thích/phán đoán khi kế hoạch yêu cầu.
- Nguồn và số trang chỉ được trả qua citations/diagnostics cho Web Admin,
  không viết vào câu trả lời hiển thị trên ứng dụng.
""".strip()

OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V24 = """
- DYNAMIC_EVOLUTION_PLANNING_F9_V24:
- Với intent historical_evolution, tuân theo evolutionIntent,
  evolutionDomain, evolutionSubjectType và period plan động do backend cung
  cấp. Quy tắc này chỉ áp dụng cho F9, không áp dụng sang dạng câu hỏi khác.
- Không dùng một bộ giai đoạn cố định theo khoảng năm. Phân kỳ phải được tạo
  từ subject + domain + khoảng hỏi + các bước ngoặt có ID bằng chứng trong
  Graph/chunk. Ranh giới PDF không phải ranh giới lịch sử.
- Mở đầu bằng luận đề phân biệt điều thay đổi và điều tiếp nối. Phân kỳ theo
  bước ngoặt có bằng chứng, không chia đều thời gian một cách máy móc và không
  để một giai đoạn có nhiều chunk lấn át toàn bộ tiến trình.
- Mỗi giai đoạn được chọn nên nêu ngắn gọn: khoảng thời gian; bối cảnh hoặc
  động lực chuyển đổi; đặc điểm/chính sách chính; một minh họa cụ thể nếu có.
- Chỉ chọn các phương diện phù hợp với domain của đối tượng. Chính sách cai
  trị ưu tiên bộ máy, phương thức đàn áp/kiểm soát/nhượng bộ, quan hệ với bộ
  máy bản xứ và tác động chiến tranh; tiến trình kinh tế, quân sự hoặc phong
  trào phải dùng taxonomy tương ứng.
- Kết luận phải trả lời đúng subtype: yếu tố giữ nguyên/thay đổi, bước ngoặt,
  nguyên nhân, mức độ, đảo chiều, kế thừa hoặc trạng thái cuối kỳ. Không chỉ
  kể một dòng thời gian rồi bỏ phần nhận xét.
- Nếu khoảng hỏi vượt vòng đời đối tượng, phải nói rõ thời điểm đối tượng kết
  thúc và quá trình/chính sách kế tiếp; không kéo dài vòng đời để lấp khoảng.
- Nếu subject quá rộng, trả lời tổng quan có kiểm soát và nói rõ các phương
  diện được xét; không âm thầm biến câu hỏi rộng thành riêng miền quân sự.
- Chỉ khẳng định mốc, chính sách và quan hệ nhân quả có trong nguồn truy xuất.
  Nếu một chặng quan trọng thiếu bằng chứng, nêu giới hạn một lần ngắn gọn;
  không tự điền bằng kiến thức ngoài nguồn và không tạo nhiều mục “chưa có dữ liệu”.
""".strip()

MANDATORY_EVOLUTION_PLANNING_INSTRUCTION = """
- RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28:
- Với intent historical_evolution, tuân theo evolutionIntent,
  evolutionDomain, evolutionSubjectType và period plan động do backend cung
  cấp. Quy tắc này chỉ áp dụng cho F9, không áp dụng sang dạng câu hỏi khác.
- Backend quét toàn khoảng hỏi theo các cửa sổ thời gian để tìm bằng chứng,
  sau đó mới phân kỳ và kiểm tra lại độ phủ. Cửa sổ truy xuất không phải giai
  đoạn lịch sử và tuyệt đối không được dùng làm tiêu đề câu trả lời.
- Không dùng một bộ giai đoạn cố định theo khoảng năm. Phân kỳ phải được tạo
  từ subject + domain + khoảng hỏi + các bước ngoặt có ID bằng chứng trong
  Graph/chunk. Khoảng năm của sách/PDF chỉ là metadata, không phải bằng chứng
  về một bước ngoặt lịch sử.
- Mỗi bước ngoặt phải giữ đúng bộ bốn: thời điểm, chủ thể, hành động và trạng
  thái trước/sau. Không đảo chủ thể với đối tượng của hành động; phải ưu tiên
  đoạn trích khóa bằng chứng do backend cung cấp khi diễn đạt nguyên nhân đổi kỳ.
- Nếu backend giữ mốc kỳ nhưng để trống boundary actor/action/cause, đoạn trích
  chỉ đủ chứng minh sự chuyển trạng thái chứ chưa đủ quy trách nhiệm. Khi đó
  vẫn trình bày kỳ tương ứng nhưng không tự gán chủ thể, hành động hay nguyên
  nhân cho bước ngoặt.
- boundary year có thể đánh dấu lúc bước vào kỳ hoặc lúc kết thúc kỳ. Hãy đọc
  nó cùng khoảng năm và đoạn trích, không mặc định mọi boundary đều là mốc mở
  đầu rồi làm sai trật tự chuyển đổi.
- Nếu trong cùng một chặng xuất hiện hai trạng thái chủ đạo đối lập hoặc một
  sự đảo chiều có bằng chứng (nhượng bộ ↔ đàn áp, mở rộng ↔ thu hẹp, trực tiếp
  ↔ gián tiếp, thời bình ↔ thời chiến, tăng tốc ↔ suy giảm...), phải tách tại
  bước ngoặt đó hoặc nói rõ đây là một tiểu chuyển tiếp; không gộp thành một
  nhãn mơ hồ.
- Mở đầu bằng luận đề phân biệt điều thay đổi và điều tiếp nối. Mỗi giai đoạn
  nêu ngắn gọn: khoảng thời gian, trạng thái chủ đạo, động lực chuyển đổi,
  đặc điểm/chính sách chính và một minh họa cụ thể nếu có.
- Chọn bằng chứng vĩ mô để đặt tên giai đoạn; thiết chế, biện pháp hay sự kiện
  nhỏ chỉ dùng làm minh họa, không được thay thế đặc trưng chủ đạo của cả kỳ.
- Chỉ chọn các phương diện phù hợp với domain của đối tượng. Chính sách cai
  trị ưu tiên bộ máy, phương thức đàn áp/kiểm soát/nhượng bộ, quan hệ với bộ
  máy bản xứ và tác động chiến tranh; tiến trình kinh tế, quân sự hoặc phong
  trào phải dùng taxonomy tương ứng.
- Kết luận phải trả lời đúng subtype: yếu tố giữ nguyên/thay đổi, bước ngoặt,
  nguyên nhân, mức độ, đảo chiều, kế thừa hoặc trạng thái cuối kỳ. Nếu có bằng
  chứng về mốc kết thúc vòng đời đối tượng thì phải nêu rõ.
- Nếu khoảng hỏi vượt vòng đời đối tượng, nói rõ thời điểm đối tượng kết thúc
  và quá trình/chính sách kế tiếp; không kéo dài vòng đời để lấp khoảng.
- Chỉ khẳng định mốc, chính sách và quan hệ nhân quả có trong nguồn truy xuất.
  Nếu một chặng quan trọng thiếu bằng chứng, nêu giới hạn một lần ngắn gọn;
  không tự điền bằng kiến thức ngoài nguồn.
- Nếu backend báo không đủ bằng chứng để phân kỳ động thì không được tự đặt
  tên các “giai đoạn” như thể đã bao phủ toàn khoảng. Khi đó chỉ trình bày
  các thay đổi đã được chứng minh và nói ngắn gọn giới hạn độ phủ.
""".strip()

OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V23 = """
- ADAPTIVE_EVOLUTION_PLANNING_F9_V23:
- Với intent historical_evolution, tuân theo evolutionIntent,
  evolutionDomain và evolutionPeriods do backend cung cấp. Taxonomy là danh
  mục để chọn lọc, không phải checklist buộc dùng toàn bộ.
- Mở đầu bằng luận đề phân biệt điều thay đổi và điều tiếp nối. Phân kỳ theo
  bước ngoặt có bằng chứng, không chia đều thời gian một cách máy móc và không
  để một giai đoạn có nhiều chunk lấn át toàn bộ tiến trình.
- Mỗi giai đoạn được chọn nên nêu ngắn gọn: khoảng thời gian; bối cảnh hoặc
  động lực chuyển đổi; đặc điểm/chính sách chính; một minh họa cụ thể nếu có.
- Chỉ chọn các phương diện phù hợp với domain của đối tượng. Chính sách cai
  trị ưu tiên bộ máy, phương thức đàn áp/kiểm soát/nhượng bộ, quan hệ với bộ
  máy bản xứ và tác động chiến tranh; tiến trình kinh tế, quân sự hoặc phong
  trào phải dùng taxonomy tương ứng.
- Kết luận phải trả lời đúng subtype: yếu tố giữ nguyên/thay đổi, bước ngoặt,
  nguyên nhân, mức độ, đảo chiều, kế thừa hoặc trạng thái cuối kỳ. Không chỉ
  kể một dòng thời gian rồi bỏ phần nhận xét.
- Chỉ khẳng định mốc, chính sách và quan hệ nhân quả có trong nguồn truy xuất.
  Nếu một chặng quan trọng thiếu bằng chứng, nêu giới hạn một lần ngắn gọn;
  không tự điền bằng kiến thức ngoài nguồn và không tạo nhiều mục “chưa có dữ liệu”.
""".strip()


def effective_answer_planning_instruction(value: str) -> str:
    """Keep mandatory runtime capabilities visible in admin prompt versions."""
    current = str(value or "").strip() or ANSWER_PLANNING_INSTRUCTION
    current = current.replace(
        OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V23,
        "",
    ).strip()
    current = current.replace(
        OBSOLETE_EVOLUTION_PLANNING_INSTRUCTION_V24,
        "",
    ).strip()
    current = re.sub(
        r"\n?- EVIDENCE_VERIFIED_EVOLUTION_PLANNING_F9_V25:"
        r".*?không tự điền bằng kiến thức ngoài nguồn\.",
        "",
        current,
        flags=re.DOTALL,
    ).strip()
    current = re.sub(
        r"\n?- TIMELINE_COVERAGE_EVOLUTION_PLANNING_F9_V26:"
        r".*?chỉ nêu các thay đổi có bằng chứng và nói ngắn gọn giới hạn độ phủ\.",
        "",
        current,
        flags=re.DOTALL,
    ).strip()
    current = re.sub(
        r"\n?- RANGE_VERIFIED_EVOLUTION_PLANNING_F9_V27:"
        r".*?chỉ nêu các thay đổi có bằng chứng và nói ngắn gọn giới hạn độ phủ\.",
        "",
        current,
        flags=re.DOTALL,
    ).strip()
    additions: list[str] = []
    if "ADAPTIVE_COMPARISON_PLANNING_V22" not in current:
        additions.append(MANDATORY_COMPARISON_PLANNING_INSTRUCTION)
    if "RECOVERABLE_BOUNDARY_EVOLUTION_PLANNING_F9_V28" not in current:
        additions.append(MANDATORY_EVOLUTION_PLANNING_INSTRUCTION)
    return "\n".join((current, *additions)).strip()


COMPARISON_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "adaptive_comparison_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "comparison_intent": {"type": "string"},
                "comparison_domain": {"type": "string"},
                "object_types": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "selected_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "include_similarities": {"type": "boolean"},
                "include_differences": {"type": "boolean"},
                "include_explanation": {"type": "boolean"},
                "include_judgement": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "comparison_intent",
                "comparison_domain",
                "object_types",
                "selected_facets",
                "include_similarities",
                "include_differences",
                "include_explanation",
                "include_judgement",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
}

EVOLUTION_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "adaptive_evolution_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "evolution_intent": {"type": "string"},
                "evolution_domain": {"type": "string"},
                "subject_type": {"type": "string"},
                "selected_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "evolution_intent",
                "evolution_domain",
                "subject_type",
                "selected_facets",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
}

EVOLUTION_PERIOD_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "dynamic_evolution_period_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "periods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "label": {"type": "string"},
                            "dominant_state": {"type": "string"},
                            "boundary_cause": {"type": "string"},
                            "boundary_year": {
                                "type": ["integer", "null"],
                            },
                            "boundary_actor": {"type": "string"},
                            "boundary_action": {"type": "string"},
                            "boundary_evidence_id": {"type": "string"},
                            "boundary_excerpt": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "start",
                            "end",
                            "label",
                            "dominant_state",
                            "boundary_cause",
                            "boundary_year",
                            "boundary_actor",
                            "boundary_action",
                            "boundary_evidence_id",
                            "boundary_excerpt",
                            "evidence_ids",
                        ],
                        "additionalProperties": False,
                    },
                },
                "subject_lifetime_start": {
                    "type": ["integer", "null"],
                },
                "subject_lifetime_end": {
                    "type": ["integer", "null"],
                },
                "range_mismatch": {"type": "boolean"},
                "range_resolution": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "periods",
                "subject_lifetime_start",
                "subject_lifetime_end",
                "range_mismatch",
                "range_resolution",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}

EVOLUTION_BOUNDARY_VERIFICATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evolution_boundary_agency_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period_index": {"type": "integer"},
                            "evidence_id": {"type": "string"},
                            "evidence_sufficient": {"type": "boolean"},
                            "agency_was_correct": {"type": "boolean"},
                            "actor": {"type": "string"},
                            "action": {"type": "string"},
                            "cause": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "period_index",
                            "evidence_id",
                            "evidence_sufficient",
                            "agency_was_correct",
                            "actor",
                            "action",
                            "cause",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
                "all_boundaries_groundable": {"type": "boolean"},
            },
            "required": [
                "checks",
                "all_boundaries_groundable",
            ],
            "additionalProperties": False,
        },
    },
}


RERANK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "history_query_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "scope": {"type": "string"},
                "date_range": {"type": "string"},
                "answerable": {"type": "boolean"},
                "covered_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "facet_coverage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "facet": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["facet", "evidence_ids"],
                        "additionalProperties": False,
                    },
                },
                "missing_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "evidence_coverage": {"type": "number"},
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "normalized_question": {"type": "string"},
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "replacement": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["original", "replacement", "reason"],
                        "additionalProperties": False,
                    },
                },
                "rewrite_confidence": {"type": "number"},
                "ambiguous": {"type": "boolean"},
                "ambiguity_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "intent",
                "scope",
                "date_range",
                "answerable",
                "covered_facets",
                "facet_coverage",
                "missing_facets",
                "evidence_coverage",
                "selected_ids",
                "normalized_question",
                "corrections",
                "rewrite_confidence",
                "ambiguous",
                "ambiguity_notes",
            ],
            "additionalProperties": False,
        },
    },
}

# Chỉ xóa chỉ số nguồn ngắn; các mốc lịch sử bốn chữ số như [1954] vẫn được giữ lại.
INLINE_CITATION_PATTERN = re.compile(
    r"\[(?:nguồn\s*)?\d{1,2}(?:\s*(?:,|;|[-–—])\s*\d{1,2})*\]",
    flags=re.IGNORECASE,
)


def strip_inline_citations(answer: str) -> str:
    """Remove model-generated source markers while preserving answer Markdown."""
    cleaned = INLINE_CITATION_PATTERN.sub("", answer)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


SOURCE_META_PREFIX_PATTERN = re.compile(
    r"\b(?:theo|trong)\s+(?:các\s+)?(?:nguồn\s+tài\s+liệu|"
    r"nguồn\s+được\s+cung\s+cấp|tài\s+liệu\s+được\s+cung\s+cấp)"
    r"\s*[,;:]?\s*",
    flags=re.IGNORECASE,
)
SOURCE_META_ENUM_PATTERN = re.compile(
    r"\b(?:các\s+)?nguồn\s+tài\s+liệu\s+(?:cũng\s+)?"
    r"(?:nêu\s+rõ|liệt\s+kê|ghi\s+nhận)\s+"
    r"(?=(?:nhiều|một\s+số|các)\b)",
    flags=re.IGNORECASE,
)
SOURCE_META_CLAIM_PATTERN = re.compile(
    r"\b(?:(?:các\s+)?nguồn\s+tài\s+liệu|tài\s+liệu)\s+"
    r"(?:cũng\s+)?(?:cho\s+thấy(?:\s+rằng)?|chỉ\s+ra(?:\s+rằng)?|"
    r"nêu\s+rằng|ghi\s+nhận\s+rằng|\u0111ề\s+cập\s+rằng)\s+",
    flags=re.IGNORECASE,
)


def naturalize_source_meta_language(answer: str) -> str:
    """Remove internal RAG attribution while preserving historical content."""
    cleaned = SOURCE_META_PREFIX_PATTERN.sub("", answer)

    def replace_enumeration(match: re.Match[str]) -> str:
        previous = cleaned[:match.start()].rstrip()
        sentence_start = not previous or previous[-1:] in ".!?\n"
        return "Có " if sentence_start else "có "

    cleaned = SOURCE_META_ENUM_PATTERN.sub(replace_enumeration, cleaned)
    cleaned = SOURCE_META_CLAIM_PATTERN.sub("", cleaned)

    def capitalize_sentence(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2).upper()}"

    cleaned = re.sub(
        r"(^|[.!?]\s+|\n)([a-zà-ỹđ])",
        capitalize_sentence,
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

DEFAULT_SYSTEM_PROMPT = """
Bạn là trợ lý học Lịch sử Việt Nam dựa trên kho tài liệu đã được quản trị viên kiểm duyệt.
Mục tiêu của bạn là trả lời chính xác, có chọn lọc, dễ đọc và luôn truy vết được nguồn.

QUY TẮC NỘI DUNG:
- Chỉ khẳng định thông tin có trong nguồn được cung cấp; không tự bổ sung dữ kiện.
- Phân biệt nhân vật chính với người chỉ được nhắc đến trong bối cảnh.
- Với câu hỏi rộng, hãy chia theo thời kỳ, khu vực, tổ chức hoặc vai trò phù hợp.
- Nếu câu hỏi mơ hồ, nêu cách hiểu đang sử dụng trước khi trả lời.
- Khi nguồn không đủ hoặc mâu thuẫn, nói rõ giới hạn đó.
""".strip()


QUERY_STOP_WORDS = {
    "ai", "bao", "co", "cua", "da", "duoc", "giai", "gi", "gom",
    "hay", "la", "mot", "nao", "nhu", "nguoi", "nhung", "ra", "sao", "the",
    "thi", "trong", "va", "ve",
}

# Nhận diện các vế phổ biến trong câu hỏi lịch sử nhiều phần.
# Trigger được so khớp sau khi bỏ dấu nên vẫn nhận ra "lạnh đão".
QUERY_FACETS = (
    (("dien ra", "nhu the nao"), "progress"),
    (("luc luong", "tham chien", "quan doi"), "forces"),
    (("lanh dao", "chi huy", "tuong linh"), "leadership"),
    (("ket qua", "thang loi"), "result"),
    (("y nghia",), "significance"),
    (("nguyen nhan", "boi canh"), "cause"),
)


class RagService:
    def __init__(self, repository: Neo4jKnowledgeRepository) -> None:
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.repository = repository

    def _refine_comparison_plan(
        self,
        question: str,
        plan: QueryPlan,
    ) -> tuple[QueryPlan, bool]:
        """Refine intent, domain and facets without letting the model answer."""
        if plan.intent != "comparison":
            return plan, False

        allowed_intents = {
            "similarities_differences",
            "main_similarity",
            "main_difference",
            "extent_comparison",
            "cause_comparison",
            "consequence_comparison",
            "effectiveness_comparison",
            "significance_comparison",
            "continuity_change",
            "role_comparison",
            "interpretation_comparison",
            "judgement",
            "focused_facets",
        }
        domain_catalog = {
            domain: [
                {
                    "id": facet_id,
                    "label": facet_label(facet_id),
                    "meaning": HISTORY_FACETS[facet_id].search_text,
                }
                for facet_id in facets
                if facet_id in HISTORY_FACETS
            ]
            for domain, facets in COMPARISON_DOMAIN_FACETS.items()
        }
        prompt = f"""
Lập kế hoạch ba tầng cho câu hỏi so sánh lịch sử trước khi truy xuất dữ liệu:
1) nhận diện comparison_intent; 2) nhận diện domain của đối tượng;
3) chọn tiêu chí truy xuất phù hợp.

NGUYÊN TẮC:
- explicit_facets do bộ phân tích chắc chắn phát hiện có ưu tiên tuyệt đối.
- Nếu explicit_facets không rỗng, không tự mở rộng ngoài chúng.
- Câu hỏi toàn diện chọn 4–7 tiêu chí; main_similarity/main_difference chọn
  2–4; câu hỏi nguyên nhân, kết quả, ý nghĩa, vai trò hoặc nhận định chọn 1–4.
- Chọn theo bản chất đối tượng, không dùng checklist chung.
- Không chọn một tiêu chí chỉ để sau đó nói rằng không có dữ liệu.
- Đây chỉ là lựa chọn cấu trúc truy xuất, không được thêm hoặc khẳng định dữ kiện
  lịch sử trong rationale.

Câu hỏi: {question}
comparison_intent dự phòng: {plan.comparison_intent or "similarities_differences"}
domain dự phòng: {plan.comparison_domain or "historical_event"}
Hai đối tượng: {json.dumps(plan.comparison_subjects, ensure_ascii=False)}
explicit_facets: {json.dumps(plan.explicit_facets, ensure_ascii=False)}
Taxonomy theo domain: {json.dumps(domain_catalog, ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0,
                response_format=COMPARISON_PLAN_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là bộ lập kế hoạch retrieval RAG lịch sử. "
                            "Bạn chỉ chọn tiêu chí, không trả lời câu hỏi."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            proposed_intent = str(
                parsed.get("comparison_intent") or ""
            ).strip()
            comparison_intent = (
                proposed_intent
                if proposed_intent in allowed_intents
                else plan.comparison_intent or "similarities_differences"
            )
            # Intent đã được nhận diện rõ bằng quy tắc/explicit facets không bị
            # model đổi sang một yêu cầu rộng hơn.
            if (
                plan.comparison_intent
                and plan.comparison_intent != "similarities_differences"
            ):
                comparison_intent = plan.comparison_intent

            proposed_domain = str(
                parsed.get("comparison_domain") or ""
            ).strip()
            comparison_domain = (
                proposed_domain
                if proposed_domain in COMPARISON_DOMAIN_FACETS
                else plan.comparison_domain or "historical_event"
            )
            candidate_facets = comparison_facets_for_domain(comparison_domain)
            explicit_facets = tuple(dict.fromkeys(plan.explicit_facets))
            allowed_facets = tuple(dict.fromkeys((
                *explicit_facets,
                *candidate_facets,
            )))
            proposed = parsed.get("selected_facets")
            proposed = proposed if isinstance(proposed, list) else []
            proposed_set = {
                str(facet_id).strip()
                for facet_id in proposed
                if str(facet_id).strip() in allowed_facets
            }

            intent_anchors = {
                "cause_comparison": ("cause",),
                "consequence_comparison": ("result", "significance"),
                "effectiveness_comparison": (
                    "objectives_consequences", "result", "limitations",
                ),
                "significance_comparison": ("significance",),
                "continuity_change": ("continuity_change",),
                "role_comparison": ("role_contribution",),
                "interpretation_comparison": ("source_perspective",),
                "judgement": ("result", "significance"),
            }
            if explicit_facets:
                # Không được thêm facet ngoài yêu cầu người dùng đã nêu rõ.
                proposed_set = set(explicit_facets)
            else:
                proposed_set.update(intent_anchors.get(comparison_intent, ()))
                if comparison_intent == "similarities_differences":
                    proposed_set.add("comparison_context")
                    if comparison_domain == "military_strategy":
                        proposed_set.update(("representative_events", "result"))

            max_facets = (
                4
                if comparison_intent in {
                    "main_similarity", "main_difference",
                    "extent_comparison", "judgement",
                }
                else 4
                if comparison_intent not in {
                    "similarities_differences", "focused_facets",
                }
                else 7
            )
            selected = tuple(
                facet_id for facet_id in allowed_facets
                if facet_id in proposed_set
            )[:max_facets]
            if not selected:
                return plan, False

            parsed_object_types = parsed.get("object_types")
            parsed_object_types = (
                parsed_object_types
                if isinstance(parsed_object_types, list)
                else []
            )
            object_types = tuple(
                str(value).strip()
                for value in parsed_object_types
                if str(value).strip() in COMPARISON_DOMAIN_FACETS
            )
            if len(object_types) != len(plan.comparison_subjects):
                object_types = tuple(
                    comparison_domain for _ in plan.comparison_subjects
                )

            answer_structure = {
                "main_similarity": "comparison_main_similarity",
                "main_difference": "comparison_main_difference",
                "extent_comparison": "comparison_extent_with_judgement",
                "cause_comparison": "comparison_focused_causes",
                "consequence_comparison": "comparison_focused_outcomes",
                "effectiveness_comparison": "comparison_effectiveness_with_criteria",
                "significance_comparison": "comparison_focused_significance",
                "continuity_change": "comparison_continuity_change",
                "role_comparison": "comparison_focused_roles",
                "interpretation_comparison": "comparison_source_perspectives",
                "judgement": "comparison_claim_evaluation",
                "focused_facets": "comparison_selected_facets",
            }.get(
                comparison_intent,
                "comparison_matrix_then_similarities_differences",
            )
            include_similarities = comparison_intent in {
                "similarities_differences", "main_similarity",
                "extent_comparison", "continuity_change",
            }
            include_differences = comparison_intent in {
                "similarities_differences", "main_difference",
                "extent_comparison", "continuity_change", "judgement",
            }
            include_explanation = comparison_intent not in {
                "main_similarity", "main_difference",
            }
            include_judgement = comparison_intent in {
                "extent_comparison", "effectiveness_comparison", "judgement",
            }
            return replace(
                plan,
                required_facets=selected,
                answer_structure=answer_structure,
                comparison_domain=comparison_domain,
                comparison_object_types=object_types,
                comparison_intent=comparison_intent,
                include_similarities=include_similarities,
                include_differences=include_differences,
                include_explanation=include_explanation,
                include_judgement=include_judgement,
            ), True
        except Exception:
            logger.exception("Comparison planner lỗi; dùng kế hoạch theo miền dự phòng")
            return plan, False

    def _refine_evolution_plan(
        self,
        question: str,
        plan: QueryPlan,
    ) -> tuple[QueryPlan, bool]:
        """Select only F9 subtype/domain/facets before evidence discovery."""
        if plan.intent != "historical_evolution":
            return plan, False

        allowed_intents = {
            "evolution_over_time",
            "continuity_and_change",
            "turning_points",
            "extent_of_change",
            "cause_of_change",
            "before_after",
            "short_long_term",
            "acceleration_slowdown",
            "reversal",
            "inheritance_development",
            "periodization",
            "decisive_change",
        }
        allowed_domains = set(EVOLUTION_DOMAIN_FACETS)
        catalog = {
            domain: [
                {
                    "id": facet_id,
                    "label": facet_label(facet_id),
                    "meaning": HISTORY_FACETS[facet_id].search_text,
                }
                for facet_id in evolution_facets_for_domain(domain)
                if facet_id in HISTORY_FACETS
            ]
            for domain in EVOLUTION_DOMAIN_FACETS
        }
        prompt = f"""
Lập kế hoạch truy xuất cho câu hỏi F9 về tiếp nối và thay đổi lịch sử:
1) nhận diện evolution_intent; 2) nhận diện evolution_domain;
3) nhận diện subject_type; 4) chọn 3–6 facet thực sự cần để trả lời.

NGUYÊN TẮC:
- Không trả lời câu hỏi và không thêm dữ kiện lịch sử trong rationale.
- Luôn giữ facet cấu trúc phù hợp với subtype: tiến trình, tiếp nối/thay đổi,
  nguyên nhân, kết quả hoặc ý nghĩa; sau đó chọn facet theo domain.
- Không coi taxonomy là checklist. Chỉ chọn phương diện giúp nhận ra bước
  chuyển, yếu tố giữ nguyên hoặc hệ quả của tiến trình đang hỏi.
- Không loại bỏ subtype đã được bộ phân tích nhận diện rõ chỉ vì một vài từ
  trong câu hỏi có thể thuộc subtype rộng hơn.
- Đây mới là lớp nhận diện subject/domain. Không tạo period tại bước này.
- Nếu câu hỏi theo dõi tình hình một quốc gia nói chung hoặc không giới hạn
  một miền, dùng domain=multi_domain và subject_type=multi_domain_historical_process.

Câu hỏi: {question}
evolution_intent dự phòng: {plan.evolution_intent}
evolution_domain dự phòng: {plan.evolution_domain}
Khoảng thời gian: {plan.explicit_date_range or "không nêu"}
Taxonomy theo domain: {json.dumps(catalog, ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0,
                response_format=EVOLUTION_PLAN_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là bộ lập kế hoạch retrieval RAG lịch sử F9. "
                            "Bạn chỉ phân loại và chọn facet, không trả lời."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            proposed_intent = str(
                parsed.get("evolution_intent") or ""
            ).strip()
            evolution_intent = (
                proposed_intent
                if proposed_intent in allowed_intents
                else plan.evolution_intent or "evolution_over_time"
            )
            if plan.evolution_intent != "evolution_over_time":
                evolution_intent = plan.evolution_intent

            proposed_domain = str(
                parsed.get("evolution_domain") or ""
            ).strip()
            evolution_domain = (
                proposed_domain
                if proposed_domain in allowed_domains
                else plan.evolution_domain or "historical_event"
            )
            subject_type = str(
                parsed.get("subject_type") or ""
            ).strip() or evolution_domain
            if (
                subject_type in HISTORY_FACETS
                or subject_type in allowed_domains
                or subject_type in {
                    "historical_evolution",
                    "historical_process",
                }
            ):
                subject_key = self._normalize_search_text(plan.subject)
                if evolution_domain == "political_administration":
                    subject_type = (
                        "governance_policy"
                        if "chinh sach" in subject_key
                        else "political_administrative_process"
                    )
                else:
                    subject_type = f"{evolution_domain}_process"
            allowed_facets = evolution_facets_for_domain(evolution_domain)
            proposed = parsed.get("selected_facets")
            proposed = proposed if isinstance(proposed, list) else []
            proposed_set = {
                str(facet_id).strip()
                for facet_id in proposed
                if str(facet_id).strip() in allowed_facets
            }
            anchors = {
                "evolution_over_time": (
                    "historical_evolution", "continuity_change", "cause",
                ),
                "continuity_and_change": (
                    "continuity_change", "historical_evolution",
                ),
                "turning_points": (
                    "historical_evolution", "cause", "result",
                ),
                "extent_of_change": (
                    "continuity_change", "historical_evolution", "result",
                ),
                "cause_of_change": ("cause", "historical_evolution"),
                "before_after": (
                    "historical_evolution", "continuity_change",
                ),
                "short_long_term": ("result", "significance"),
                "acceleration_slowdown": (
                    "historical_evolution", "cause",
                ),
                "reversal": (
                    "historical_evolution", "cause", "continuity_change",
                ),
                "inheritance_development": (
                    "continuity_change", "historical_evolution", "result",
                ),
                "periodization": ("historical_evolution", "cause"),
                "decisive_change": (
                    "historical_evolution", "result", "significance",
                ),
            }
            proposed_set.update(anchors.get(evolution_intent, ()))
            selected = tuple(
                facet_id for facet_id in allowed_facets
                if facet_id in proposed_set
            )[:6]
            if not selected:
                return plan, False
            return replace(
                plan,
                required_facets=selected,
                answer_structure=f"evolution_{evolution_intent}",
                evolution_intent=evolution_intent,
                evolution_domain=evolution_domain,
                evolution_subject_type=subject_type,
            ), True
        except Exception:
            logger.exception("Evolution planner lỗi; dùng kế hoạch F9 dự phòng")
            return plan, False

    @staticmethod
    def _explicit_evolution_range(
        plan: QueryPlan,
    ) -> tuple[int, int] | None:
        years = [
            int(value)
            for value in re.findall(r"\b\d{4}\b", plan.explicit_date_range)
        ]
        if len(years) < 2:
            return None
        return min(years), max(years)

    @staticmethod
    def _evolution_timeline_windows(
        requested_range: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Build recall windows, not historical periods, across an F9 range."""
        start, end = requested_range
        span = end - start + 1
        window_count = min(9, max(4, (span + 6) // 7))
        width = max(1, (span + window_count - 1) // window_count)
        windows: list[tuple[int, int]] = []
        cursor = start
        while cursor <= end:
            window_end = min(end, cursor + width - 1)
            windows.append((cursor, window_end))
            cursor = window_end + 1
        return windows

    def _retrieve_evolution_timeline_coverage(
        self,
        question: str,
        plan: QueryPlan,
    ) -> list[dict]:
        """Recall a small quota from every time window before periodization."""
        requested_range = self._explicit_evolution_range(plan)
        retrieve = getattr(
            self.repository,
            "retrieve_timeline_windows",
            None,
        )
        if (
            plan.intent != "historical_evolution"
            or requested_range is None
            or not callable(retrieve)
        ):
            return []
        try:
            focus_text = " ".join(
                facet_detail_search_text(facet)
                for facet in plan.required_facets[:3]
            )
            return list(retrieve(
                " ".join((
                    plan.subject,
                    focus_text,
                    question,
                )).strip(),
                self._evolution_timeline_windows(requested_range),
                per_window=4,
            ))
        except Exception:
            logger.exception(
                "Không thể quét phủ timeline F9; tiếp tục với hybrid retrieval"
            )
            return []

    @classmethod
    def _ground_evolution_boundary_excerpt(
        cls,
        raw_excerpt: str,
        card: dict,
        boundary_year: int | None,
    ) -> str:
        """Return an actual source passage instead of trusting model wording."""
        source = re.sub(
            r"\s+",
            " ",
            " ".join((
                str(card.get("heading") or ""),
                str(card.get("text") or ""),
            )),
        ).strip()
        if not source:
            return ""
        raw_excerpt = re.sub(r"\s+", " ", raw_excerpt or "").strip()
        if raw_excerpt and raw_excerpt in source:
            return raw_excerpt[:360]

        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n+", source)
            if segment.strip()
        ]
        year_text = str(boundary_year) if boundary_year is not None else ""
        candidates = [
            segment for segment in segments
            if year_text and year_text in segment
        ]
        if not candidates and raw_excerpt:
            raw_terms = cls._salient_terms(raw_excerpt)
            candidates = sorted(
                segments,
                key=lambda segment: len(
                    raw_terms & cls._salient_terms(segment)
                ),
                reverse=True,
            )[:1]
        if not candidates:
            candidates = segments[:1]
        if not candidates:
            return source[:360]

        raw_terms = cls._salient_terms(raw_excerpt)
        best = max(
            candidates,
            key=lambda segment: (
                len(raw_terms & cls._salient_terms(segment)),
                min(len(segment), 360),
            ),
        )
        return best[:360]

    @classmethod
    def _evolution_evidence_cards(
        cls,
        candidates: list[dict],
        requested_range: tuple[int, int] | None = None,
        subject: str = "",
        limit: int = 36,
    ) -> list[dict]:
        """Build compact, evidence-ID-bound cards for the F9 period planner."""
        turning_terms = cls._salient_terms(
            "bước ngoặt chuyển sang thay thế bắt đầu kết thúc thất bại "
            "thắng lợi hiệp định cải cách khủng hoảng thành lập giải phóng "
            "thống nhất đảo chính thay đổi chính quyền thay đổi chiến lược "
            "xóa bỏ ban hành bãi bỏ chấm dứt tăng cường suy yếu sụp đổ"
        )
        subject_terms = {
            term
            for term in cls._salient_terms(subject)
            if not term.isdigit()
        }
        subject_terms -= cls._salient_terms(
            "thay đổi tiếp nối biến đổi như thế nào qua các giai đoạn "
            "từ đến thời kỳ bước ngoặt tiến trình"
        )
        ranked: list[tuple[float, dict]] = []
        seen: set[str] = set()
        for item in candidates:
            item_id = str(item.get("id") or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            heading_raw = str(item.get("heading") or "")
            text_raw = str(item.get("text") or "")
            content_raw = f"{heading_raw} {text_raw}"
            heading_years = sorted({
                int(value)
                for value in re.findall(
                    r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                    heading_raw,
                )
            })
            text_years = sorted({
                int(value)
                for value in re.findall(
                    r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                    text_raw,
                )
            })
            explicit_content_years = sorted(set(heading_years) | set(text_years))
            content_terms = cls._salient_terms(content_raw)
            turning_overlap = len(turning_terms & content_terms)
            subject_overlap = len(subject_terms & content_terms)
            subject_relevance = (
                min(1.0, subject_overlap / max(2, min(6, len(subject_terms))))
                if subject_terms
                else 0.5
            )
            boundary_strength = min(
                1.0,
                0.55 * min(1.0, turning_overlap / 3)
                + 0.30 * (1.0 if explicit_content_years else 0.0)
                + 0.15 * subject_relevance,
            )
            rank = (
                0.30 * boundary_strength
                + 0.25 * subject_relevance
                + 0.20 * cls._concrete_evidence_score(item)
                + 0.15 * float(item.get("score") or 0)
                + 0.10 * (1.0 if explicit_content_years else 0.0)
            )
            ranked.append((rank, {
                "id": item_id,
                "source_title": str(item.get("sourceTitle") or "")[:180],
                "heading": str(item.get("heading") or "")[:220],
                # Metadata chỉ dùng để lấy mẫu theo vùng thời gian. Không được
                # coi khoảng của sách/PDF là bằng chứng cho một bước ngoặt.
                "metadata_year_start": item.get("yearStart"),
                "metadata_year_end": item.get("yearEnd"),
                "heading_years": heading_years[:20],
                "text_years": text_years[:20],
                "explicit_content_years": explicit_content_years[:20],
                "temporal_evidence_levels": {
                    "direct_text_year": bool(text_years),
                    "section_heading_year": bool(heading_years),
                    "metadata_range_only": bool(
                        not explicit_content_years
                        and (
                            item.get("yearStart") is not None
                            or item.get("yearEnd") is not None
                        )
                    ),
                },
                "subject_relevance": round(subject_relevance, 3),
                "boundary_strength": round(boundary_strength, 3),
                "boundary_eligible": bool(
                    explicit_content_years
                    and turning_overlap >= 1
                    and subject_relevance >= 0.25
                ),
                "text": text_raw[:900],
                "related_entities": [
                    str(value)[:160]
                    for value in (item.get("relatedEntities") or [])[:16]
                ],
            }))
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        if requested_range is None:
            return [card for _, card in ranked[:limit]]

        # Lấy mẫu đều trên toàn khoảng chỉ để tránh mất evidence ở đầu/cuối.
        # Đây không phải phân kỳ: ranh giới cuối cùng vẫn do bước ngoặt quyết định.
        start, end = requested_range
        bucket_count = min(8, max(2, (end - start) // 10 + 1))
        bucket_width = max(1, (end - start + 1) / bucket_count)
        selected: list[dict] = []
        selected_ids: set[str] = set()
        for bucket_index in range(bucket_count):
            bucket_start = start + bucket_index * bucket_width
            bucket_end = (
                end
                if bucket_index == bucket_count - 1
                else start + (bucket_index + 1) * bucket_width
            )
            bucket_candidates: list[dict] = []
            for _, card in ranked:
                card_years = [
                    int(value)
                    for value in card.get("explicit_content_years") or []
                ]
                year_start = card.get("metadata_year_start")
                year_end = card.get("metadata_year_end")
                # Metadata chỉ là fallback lấy mẫu cho state evidence.
                if (
                    not card_years
                    and isinstance(year_start, (int, float))
                    and isinstance(year_end, (int, float))
                ):
                    card_years.extend((int(year_start), int(year_end)))
                if card_years and (
                    min(card_years) <= bucket_end
                    and max(card_years) >= bucket_start
                ):
                    bucket_candidates.append(card)
            for card in bucket_candidates[:3]:
                if card["id"] not in selected_ids:
                    selected.append(card)
                    selected_ids.add(card["id"])

        for _, card in ranked:
            if len(selected) >= limit:
                break
            if card["id"] not in selected_ids:
                selected.append(card)
                selected_ids.add(card["id"])
        return selected[:limit]

    def _verify_evolution_boundary_agency(
        self,
        subject: str,
        periods: list[tuple],
    ) -> list[tuple] | None:
        """Verify actor/action semantic roles against locked excerpts for F9."""
        verification_model = str(
            getattr(self.settings, "openai_entity_model", "") or ""
        ).strip()
        # Unit-test/minimal configurations may intentionally omit the verifier.
        if not verification_model:
            return periods

        claims = [
            {
                "period_index": index,
                "range": [period[0], period[1]],
                "boundary_year": period[5],
                "claimed_actor": period[6],
                "claimed_action": period[7],
                "claimed_cause": period[4],
                "evidence_id": period[8],
                "evidence_excerpt": period[9],
            }
            for index, period in enumerate(periods)
            if period[9]
        ]
        if not claims:
            return periods

        prompt = f"""
Kiểm tra vai nghĩa cho các bước ngoặt F9 của subject: {subject}

Chỉ dùng evidence_excerpt trong từng claim, không dùng kiến thức ngoài.
Với mỗi claim:
- xác định đúng chủ thể thực hiện hành động, hành động và đối tượng/tác động;
- không nhầm chủ thể với tổ chức/người được nhắc trong bổ ngữ;
- ví dụ cấu trúc “A xóa bỏ cải cách mà B tiến hành” thì actor của “xóa bỏ”
  là A, không phải B;
- nếu claimed_actor/action/cause sai nhưng excerpt đủ rõ, trả actor, action,
  cause đã sửa theo đúng ngữ pháp của excerpt;
- evidence_sufficient=false nếu excerpt không đủ để xác định quan hệ đó.

CLAIMS:
{json.dumps(claims, ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=verification_model,
                temperature=0,
                response_format=EVOLUTION_BOUNDARY_VERIFICATION_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là bộ kiểm tra semantic role cho bước ngoặt "
                            "lịch sử F9. Không bổ sung kiến thức ngoài excerpt."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            raw_checks = parsed.get("checks")
            raw_checks = raw_checks if isinstance(raw_checks, list) else []
            checks = {
                int(check["period_index"]): check
                for check in raw_checks
                if (
                    isinstance(check, dict)
                    and isinstance(check.get("period_index"), int)
                )
            }
            corrected = list(periods)
            downgraded_indices: list[int] = []
            for claim in claims:
                index = int(claim["period_index"])
                check = checks.get(index)
                if (
                    check is None
                    or str(check.get("evidence_id") or "")
                    != str(claim["evidence_id"])
                ):
                    return None
                actor = str(check.get("actor") or "").strip()
                action = str(check.get("action") or "").strip()
                cause = str(check.get("cause") or "").strip()
                period = list(corrected[index])
                if (
                    bool(check.get("evidence_sufficient"))
                    and actor
                    and action
                    and cause
                ):
                    period[4] = cause[:240]
                    period[6] = actor[:160]
                    period[7] = action[:220]
                else:
                    # The year and passage were already locked to a relevant
                    # evidence card.  If the excerpt cannot safely establish
                    # agency, keep the period boundary but remove the
                    # unsupported actor/action/cause claim.  One uncertain
                    # semantic role must not discard the other verified
                    # periods and force generation back to a coarse timeline.
                    period[4] = ""
                    period[6] = ""
                    period[7] = ""
                    downgraded_indices.append(index)
                corrected[index] = tuple(period)
            if downgraded_indices:
                logger.warning(
                    "F9 boundary semantic claims downgraded for periods %s; "
                    "kept evidence-locked years/excerpts",
                    downgraded_indices,
                )
            return corrected
        except Exception:
            logger.exception(
                "Không thể kiểm tra semantic role của bước ngoặt F9"
            )
            return None

    def _derive_evolution_period_plan(
        self,
        question: str,
        plan: QueryPlan,
        candidates: list[dict],
        signals: QuerySignals,
        *,
        coverage_revision: bool = False,
    ) -> tuple[QueryPlan, bool]:
        """Create an F9-only periodization from retrieved turning-point evidence."""
        def reject(reason: str) -> tuple[QueryPlan, bool]:
            logger.warning(
                "F9 period plan rejected (%s): %s",
                "coverage_revision" if coverage_revision else "discovery",
                reason,
            )
            return plan, False

        requested_range = self._explicit_evolution_range(plan)
        if (
            plan.intent != "historical_evolution"
            or requested_range is None
            or not candidates
        ):
            return reject("missing intent, explicit range, or candidates")

        evidence_cards = self._evolution_evidence_cards(
            candidates,
            requested_range=requested_range,
            subject=plan.subject,
        )
        if len(evidence_cards) < 2:
            return reject("fewer than two evidence cards")

        known_lifetime = known_entity_year_range(signals.primary_entity)
        previous_periods = [
            {
                "range": [start, end],
                "label": (
                    plan.evolution_period_labels[index]
                    if index < len(plan.evolution_period_labels)
                    else ""
                ),
                "dominant_state": (
                    plan.evolution_period_states[index]
                    if index < len(plan.evolution_period_states)
                    else ""
                ),
            }
            for index, (start, end) in enumerate(plan.evolution_periods)
        ]
        prompt = f"""
Tạo period plan ĐỘNG chỉ cho câu hỏi F9 tiếp nối và thay đổi.
Chế độ: {"KIỂM TRA LẠI ĐỘ PHỦ" if coverage_revision else "KHÁM PHÁ TIMELINE"}.

Câu hỏi: {question}
Subject: {plan.subject}
Subject type: {plan.evolution_subject_type or plan.evolution_domain}
Domain: {plan.evolution_domain}
Subtype F9: {plan.evolution_intent}
Khoảng người dùng hỏi: {requested_range[0]}–{requested_range[1]}
Vòng đời đã biết từ ontology (có thể rỗng): {known_lifetime or "không có"}
Period plan sơ bộ (có thể rỗng, không mặc nhiên đúng):
{json.dumps(previous_periods, ensure_ascii=False)}

QUY TẮC BẮT BUỘC:
- Không dùng phân kỳ học thuộc hoặc một bộ giai đoạn cố định theo khoảng năm.
- Chỉ chọn ranh giới từ card có boundary_eligible=true và nội dung cho thấy
  thay đổi trạng thái trực tiếp của subject. metadata_year_start/end là khoảng
  của nguồn, TUYỆT ĐỐI không dùng một mình để đặt ranh giới.
- Với từng chặng, dominant_state phải mô tả đặc trưng vĩ mô của subject;
  không lấy một hội đồng, nghị định, thiết chế hoặc sự kiện nhỏ làm nhãn cả kỳ.
- Với mỗi ranh giới tự nhiên, phải ghi đủ boundary_year, boundary_actor,
  boundary_action, boundary_evidence_id và boundary_excerpt. Excerpt phải là
  một đoạn NGUYÊN VĂN ngắn có trong chính card đó, giữ nguyên trật tự chủ thể
  – hành động – đối tượng để tránh đảo nghĩa.
- boundary_year có thể là bước ngoặt mở chặng (sát start) hoặc kết thúc chặng
  (sát end), nhưng phải nằm sát đúng một đầu của chính chặng đó; không đặt một
  sự kiện giữa kỳ vào trường boundary_year.
- Chặng đầu có thể dùng boundary_year=null nếu năm đầu chỉ là biên cắt do
  người dùng chọn, không phải bước ngoặt tự nhiên. Các chặng sau bắt buộc có
  boundary_year xuất hiện trực tiếp trong text_years hoặc heading_years. Mốc
  trong heading là bằng chứng cấp mục/chương hợp lệ; metadata_year_start/end
  chỉ dùng recall và không đủ xác lập ranh giới.
- Nếu evidence cho thấy trong một chặng có hai trạng thái đối lập hay đảo
  chiều đáng kể (mở/nới lỏng ↔ siết/đàn áp; trực tiếp ↔ gián tiếp; mở rộng ↔
  thu hẹp; thời bình ↔ thời chiến; tăng trưởng ↔ khủng hoảng...), phải tách
  thành hai chặng tại bằng chứng chuyển đổi. Quy tắc này áp dụng theo ngữ
  nghĩa của domain, không dựa trên một danh sách mốc năm cố định.
- Chấm ngầm theo relevanceToSubject, magnitudeOfChange, causalImportance,
  boundaryStrength, evidenceCoverage, historicalSignificance và redundancy.
- Tạo 2–7 giai đoạn, sắp thời gian, bao phủ đúng khoảng người dùng hỏi. Có thể
  dùng mốc đầu/cuối làm biên cắt dù không có sự kiện đúng vào hai năm đó.
- Mỗi giai đoạn phải có ít nhất một evidence_id trực tiếp hỗ trợ dominant_state.
  Không đưa một sự kiện nhỏ thành ranh giới chỉ vì nó có năm.
- Một mốc chỉ đủ làm ranh giới khi đồng thời: thay đổi trạng thái cốt lõi của
  subject, có trạng thái trước/sau phân biệt, ảnh hưởng kéo dài chứ không chỉ
  là một biện pháp đơn lẻ, và phù hợp domain đang hỏi.
- Mô tả trạng thái bằng vector phương diện phù hợp domain. Với chính sách cai
  trị có thể xét mức tập quyền, phương thức kiểm soát/đàn áp/nhượng bộ, vai
  trò bộ máy bản xứ và chế độ thời bình/thời chiến. Với domain khác phải tự
  chọn vector tương ứng; đây không phải checklist dùng cho mọi câu hỏi.
- Kiểm tra toàn khoảng hỏi: một chặng dài không được che lấp một lần đảo chiều
  đã có evidence. Ngược lại, gộp các mốc nếu state vector trước/sau về bản
  chất không đổi. Các cửa sổ timeline chỉ bảo đảm recall, không phải period.
- Nếu khoảng hỏi cắt nhiều PDF, ghép theo lịch sử; tuyệt đối không dùng tên
  hoặc khoảng năm của PDF làm periodization.
- Nếu khoảng hỏi vượt vòng đời subject, đánh dấu range_mismatch và dùng phần
  cuối để giải thích quá trình kế tiếp; không kéo dài subject hoặc bịa trạng thái.
- label mô tả trạng thái chủ đạo; boundary_cause nói ngắn gọn điều gì tạo
  chuyển đổi dựa đúng boundary_excerpt. Không viết câu trả lời hoàn chỉnh.

EVIDENCE CARDS:
{json.dumps(evidence_cards, ensure_ascii=False)}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0,
                response_format=EVOLUTION_PERIOD_PLAN_RESPONSE_FORMAT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là bộ phân kỳ động cho riêng câu hỏi lịch sử "
                            "F9. Mọi period phải được khóa bằng evidence ID."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            try:
                confidence = float(parsed.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0
            if confidence < 0.55:
                return reject(f"confidence={confidence:.3f}")

            cards_by_id = {
                str(card["id"]): card for card in evidence_cards
            }
            valid_ids = set(cards_by_id)
            proposed_periods = parsed.get("periods")
            proposed_periods = (
                proposed_periods if isinstance(proposed_periods, list) else []
            )
            normalized_periods: list[tuple[
                int,
                int,
                str,
                str,
                str,
                int | None,
                str,
                str,
                str,
                str,
                tuple[str, ...],
            ]] = []
            for raw_period in proposed_periods[:7]:
                if not isinstance(raw_period, dict):
                    continue
                try:
                    start = int(raw_period.get("start"))
                    end = int(raw_period.get("end"))
                except (TypeError, ValueError):
                    continue
                if start > end:
                    continue
                start = max(requested_range[0], start)
                end = min(requested_range[1], end)
                if start > end:
                    continue
                evidence_ids = tuple(dict.fromkeys(
                    str(value).strip()
                    for value in (raw_period.get("evidence_ids") or [])
                    if str(value).strip() in valid_ids
                ))
                label = str(raw_period.get("label") or "").strip()
                dominant_state = str(
                    raw_period.get("dominant_state") or label
                ).strip()
                boundary_cause = str(
                    raw_period.get("boundary_cause") or ""
                ).strip()
                boundary_actor = str(
                    raw_period.get("boundary_actor") or ""
                ).strip()
                boundary_action = str(
                    raw_period.get("boundary_action") or ""
                ).strip()
                boundary_evidence_id = str(
                    raw_period.get("boundary_evidence_id")
                    or (evidence_ids[0] if evidence_ids else "")
                ).strip()
                boundary_excerpt = re.sub(
                    r"\s+",
                    " ",
                    str(raw_period.get("boundary_excerpt") or "").strip(),
                )
                raw_boundary_year = raw_period.get("boundary_year")
                try:
                    boundary_year = (
                        int(raw_boundary_year)
                        if raw_boundary_year is not None
                        else None
                    )
                except (TypeError, ValueError):
                    boundary_year = None
                boundary_card = cards_by_id.get(boundary_evidence_id)
                if (
                    not evidence_ids
                    or not label
                    or not dominant_state
                    or boundary_card is None
                ):
                    continue
                if boundary_evidence_id not in evidence_ids:
                    evidence_ids = (
                        boundary_evidence_id,
                        *evidence_ids,
                    )
                explicit_years = {
                    int(value)
                    for value in (
                        boundary_card.get("explicit_content_years") or []
                    )
                }
                if boundary_year is not None and boundary_year not in explicit_years:
                    continue
                boundary_excerpt = self._ground_evolution_boundary_excerpt(
                    boundary_excerpt,
                    boundary_card,
                    boundary_year,
                )
                if boundary_year is not None and len(boundary_excerpt) < 18:
                    continue
                normalized_periods.append((
                    start,
                    end,
                    label[:180],
                    dominant_state[:220],
                    boundary_cause[:240],
                    boundary_year,
                    boundary_actor[:160],
                    boundary_action[:220],
                    boundary_evidence_id,
                    boundary_excerpt[:360],
                    evidence_ids,
                ))

            normalized_periods.sort(key=lambda item: (item[0], item[1]))
            deduplicated: list[
                tuple[
                    int,
                    int,
                    str,
                    str,
                    str,
                    int | None,
                    str,
                    str,
                    str,
                    str,
                    tuple[str, ...],
                ]
            ] = []
            for period in normalized_periods:
                if deduplicated and period[:2] == deduplicated[-1][:2]:
                    continue
                deduplicated.append(period)
            normalized_periods = deduplicated
            if not 2 <= len(normalized_periods) <= 7:
                return reject(
                    f"normalized period count={len(normalized_periods)}"
                )

            def evidence_supports_range_cut(
                period: tuple,
                cut_year: int,
            ) -> bool:
                for evidence_id in period[10]:
                    card = cards_by_id.get(str(evidence_id)) or {}
                    content_years = [
                        int(value)
                        for value in (
                            card.get("explicit_content_years") or []
                        )
                    ]
                    if (
                        content_years
                        and min(content_years) <= cut_year <= max(content_years)
                    ):
                        return True
                    if not content_years:
                        metadata_start = card.get("metadata_year_start")
                        metadata_end = card.get("metadata_year_end")
                        if (
                            isinstance(metadata_start, (int, float))
                            and isinstance(metadata_end, (int, float))
                            and int(metadata_start)
                            <= cut_year
                            <= int(metadata_end)
                        ):
                            return True
                return False

            # The first/last years are user-selected cuts, not natural
            # historical boundaries. Extend the edge period when its own
            # evidence covers that cut instead of rejecting an otherwise
            # valid plan merely because the model returned 1898–1944.
            first_period = list(normalized_periods[0])
            if first_period[0] != requested_range[0]:
                if not evidence_supports_range_cut(
                    normalized_periods[0],
                    requested_range[0],
                ):
                    return reject("first period lacks evidence for range start")
                first_period[0] = requested_range[0]
                if (
                    first_period[5] is not None
                    and abs(int(first_period[5]) - requested_range[0]) > 1
                ):
                    first_period[4] = ""
                    first_period[5] = None
                    first_period[6] = ""
                    first_period[7] = ""
                    first_period[9] = ""
                normalized_periods[0] = tuple(first_period)

            # The first period always begins at the user's left-hand range
            # cut.  A planner sometimes attaches a later event inside that
            # first period (for example 1911 inside 1897–1918) as if it were
            # the opening boundary.  That is internally inconsistent.  Keep
            # the period/state evidence, but remove this misplaced boundary
            # claim so it cannot invalidate the complete timeline.
            first_period = list(normalized_periods[0])
            if (
                first_period[5] is not None
                and abs(int(first_period[5]) - requested_range[0]) > 1
            ):
                first_period[4] = ""
                first_period[5] = None
                first_period[6] = ""
                first_period[7] = ""
                first_period[9] = ""
                normalized_periods[0] = tuple(first_period)

            last_period = list(normalized_periods[-1])
            if last_period[1] != requested_range[1]:
                if not evidence_supports_range_cut(
                    normalized_periods[-1],
                    requested_range[1],
                ):
                    return reject("last period lacks evidence for range end")
                last_period[1] = requested_range[1]
                normalized_periods[-1] = tuple(last_period)

            if (
                normalized_periods[0][0] != requested_range[0]
                or normalized_periods[-1][1] != requested_range[1]
            ):
                return reject("periods do not cover requested boundaries")
            if any(
                current[0] > previous[1] + 1
                or current[0] < previous[1] - 1
                for previous, current in zip(
                    normalized_periods,
                    normalized_periods[1:],
                    strict=False,
                )
            ):
                return reject("periods contain a gap or excessive overlap")
            for period_index, period in enumerate(normalized_periods):
                boundary_year = period[5]
                boundary_card = cards_by_id.get(period[8]) or {}
                if period_index == 0 and boundary_year is None:
                    continue
                # Actor/action/cause are model proposals, not evidence.  Do
                # not discard an otherwise grounded timeline merely because
                # one of those proposal fields is empty: the semantic-role
                # verifier below derives and corrects them from the locked
                # source excerpt.  The hard preconditions here are only the
                # year, source passage and relevance of the evidence card.
                if (
                    boundary_year is None
                    or min(
                        abs(boundary_year - period[0]),
                        abs(boundary_year - period[1]),
                    ) > 1
                    or not period[9]
                    or not bool(boundary_card.get("boundary_eligible"))
                    or float(
                        boundary_card.get("subject_relevance") or 0
                    ) < 0.25
                ):
                    return reject(
                        "period "
                        f"{period_index} has an ungrounded boundary "
                        f"(start={period[0]}, year={boundary_year}, "
                        f"excerpt={bool(period[9])}, "
                        "eligible="
                        f"{bool(boundary_card.get('boundary_eligible'))}, "
                        "relevance="
                        f"{float(boundary_card.get('subject_relevance') or 0):.3f})"
                    )

            verified_periods = self._verify_evolution_boundary_agency(
                plan.subject,
                normalized_periods,
            )
            if verified_periods is None:
                return reject("semantic-role verifier rejected a boundary")
            normalized_periods = verified_periods

            lifetime_start = parsed.get("subject_lifetime_start")
            lifetime_end = parsed.get("subject_lifetime_end")
            try:
                lifetime = (
                    int(lifetime_start),
                    int(lifetime_end),
                ) if lifetime_start is not None and lifetime_end is not None else ()
            except (TypeError, ValueError):
                lifetime = ()
            if lifetime and lifetime[0] > lifetime[1]:
                lifetime = ()
            if known_lifetime:
                lifetime = known_lifetime

            mismatch = bool(parsed.get("range_mismatch"))
            if lifetime:
                mismatch = (
                    requested_range[0] < lifetime[0]
                    or requested_range[1] > lifetime[1]
                )
            resolution = str(
                parsed.get("range_resolution") or ""
            ).strip()
            if mismatch and not resolution:
                resolution = "explain_transition_to_successor"

            return replace(
                plan,
                evolution_periods=tuple(
                    (start, end)
                    for start, end, *_ in normalized_periods
                ),
                evolution_period_labels=tuple(
                    period[2] for period in normalized_periods
                ),
                evolution_period_states=tuple(
                    period[3] for period in normalized_periods
                ),
                evolution_boundary_causes=tuple(
                    period[4] for period in normalized_periods
                ),
                evolution_boundary_years=tuple(
                    period[5] for period in normalized_periods
                ),
                evolution_boundary_actors=tuple(
                    period[6] for period in normalized_periods
                ),
                evolution_boundary_actions=tuple(
                    period[7] for period in normalized_periods
                ),
                evolution_boundary_evidence_ids=tuple(
                    period[8] for period in normalized_periods
                ),
                evolution_boundary_excerpts=tuple(
                    period[9] for period in normalized_periods
                ),
                evolution_period_evidence_ids=tuple(
                    period[10] for period in normalized_periods
                ),
                evolution_subject_lifetime=lifetime,
                evolution_range_mismatch=mismatch,
                evolution_range_resolution=resolution[:120],
            ), True
        except Exception:
            logger.exception(
                "Không thể tạo period plan F9 động; dùng retrieval không phân kỳ"
            )
            return reject("planner exception")

    def _embed_with_usage(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], TokenUsage]:
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
        )
        usage = TokenUsage()
        usage.add_response(response)
        return [item.embedding for item in response.data], usage

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings, _ = self._embed_with_usage(texts)
        return embeddings

    def embed_chunks(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings, _ = self.embed_chunks_with_usage(texts, batch_size)
        return embeddings

    def embed_chunks_with_usage(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> tuple[list[list[float]], TokenUsage]:
        embeddings: list[list[float]] = []
        usage = TokenUsage()
        for offset in range(0, len(texts), batch_size):
            batch_embeddings, batch_usage = self._embed_with_usage(
                texts[offset:offset + batch_size]
            )
            embeddings.extend(batch_embeddings)
            usage.add(batch_usage)
        return embeddings, usage

    def _log_query(self, payload: dict) -> None:
        """Keep observability failures from discarding a completed chat answer."""
        try:
            self.repository.log_query(payload)
        except Exception:
            logger.exception("Không thể ghi AIQueryLog vào Neo4j")

    def _active_prompt_layers(self) -> dict[str, str]:
        active = self.repository.get_active_prompt() or {}
        return {
            "system": str(active.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT),
            "query_normalization": str(
                active.get("queryNormalizationInstruction")
                or QUERY_NORMALIZATION_INSTRUCTION
            ),
            "answer_planning": str(
                effective_answer_planning_instruction(
                    active.get("answerPlanningInstruction")
                    or ANSWER_PLANNING_INSTRUCTION
                )
            ),
            "presentation": str(
                active.get("presentationInstruction")
                or PRESENTATION_INSTRUCTION
            ),
            "output_contract": str(
                active.get("outputContract") or OUTPUT_CONTRACT
            ),
        }

    def _answer_event_phases_best_effort(self, question: str) -> str:
        """Keep a concise overview when indexed evidence lacks phase details."""
        layers = self._active_prompt_layers()
        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.15,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{layers['system']}\n\n{layers['answer_planning']}"
                        f"\n\n{layers['presentation']}"
                        f"\n\n{layers['output_contract']}\n\n"
                        "CHẾ ĐỘ BEST-EFFORT KHÔNG NGUỒN:\n"
                        "- Kho truy xuất chưa có bằng chứng đủ chi tiết, nhưng "
                        "quản trị viên vẫn cho phép trả lời một bản tổng quan "
                        "ngắn bằng kiến thức lịch sử phổ quát của model.\n"
                        "- Trình bày các đợt/giai đoạn và diễn biến khái quát "
                        "như một câu trả lời học tập hữu ích.\n"
                        "- Đây là ngoại lệ rõ ràng đối với quy tắc chỉ dùng "
                        "nguồn: vẫn phải nêu các địa danh, cứ điểm hoặc khu vực "
                        "tiêu biểu gắn với từng đợt nếu đó là kiến thức lịch sử "
                        "phổ quát mà model biết chắc. Không thay tên riêng bằng "
                        "cách viết chung chung như 'phía bắc' hoặc 'khu trung "
                        "tâm'; không tự tạo địa danh khi không chắc chắn.\n"
                        "- Riêng khi câu hỏi là các đợt của Chiến dịch Điện "
                        "Biên Phủ, phải giữ các cứ điểm tiêu biểu của đợt đầu "
                        "là Him Lam, Độc Lập và Bản Kéo; có thể bổ sung các "
                        "cứ điểm phía đông ở đợt sau nếu chắc chắn.\n"
                        "- Không bổ sung ngày tháng chính xác, tên nhân vật "
                        "hoặc số liệu trong chế độ này; các chi tiết đó chỉ "
                        "được thêm khi RAG có bằng chứng.\n"
                        "- Không nói về kho dữ liệu, nguồn, citation hay chế "
                        "độ best-effort trong câu trả lời."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        return naturalize_source_meta_language(strip_inline_citations(
            response.choices[0].message.content or ""
        )).strip()

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
        without_accents = "".join(
            character for character in decomposed
            if unicodedata.category(character) != "Mn"
        )
        return re.sub(r"[^0-9a-z]+", " ", without_accents).strip()

    @classmethod
    def _salient_terms(cls, value: str) -> set[str]:
        return {
            term for term in cls._normalize_search_text(value).split()
            if len(term) >= 2 and term not in QUERY_STOP_WORDS
        }

    @staticmethod
    def _expanded_exact_phrases(signals: QuerySignals) -> list[str]:
        """Search every curated spelling of a recognized named subject."""
        phrases: list[str] = []
        for phrase in signals.exact_phrases:
            for candidate in (
                phrase,
                *expand_known_entity_aliases(phrase),
            ):
                cleaned = str(candidate or "").strip()
                if cleaned and cleaned.casefold() not in {
                    item.casefold() for item in phrases
                }:
                    phrases.append(cleaned)
        return phrases

    @staticmethod
    def _candidate_year_span(item: dict) -> tuple[int, int] | None:
        """Prefer an explicit source-volume range over inferred chunk years."""
        source_years = [
            int(value)
            for value in re.findall(
                r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                str(item.get("sourceTitle") or ""),
            )
        ]
        if source_years:
            return min(source_years), max(source_years)

        item_year_start = item.get("yearStart")
        item_year_end = item.get("yearEnd")
        if (
            isinstance(item_year_start, (int, float))
            and isinstance(item_year_end, (int, float))
        ):
            return int(item_year_start), int(item_year_end)
        return None

    @staticmethod
    def _candidate_content_year_span(item: dict) -> tuple[int, int] | None:
        """Prefer chunk-level years when balancing an evolution timeline."""
        item_year_start = item.get("yearStart")
        item_year_end = item.get("yearEnd")
        if (
            isinstance(item_year_start, (int, float))
            and isinstance(item_year_end, (int, float))
        ):
            return int(item_year_start), int(item_year_end)
        years = [
            int(value)
            for value in re.findall(
                r"\b(?:1[0-9]{3}|20[0-9]{2})\b",
                " ".join((
                    str(item.get("heading") or ""),
                    str(item.get("text") or ""),
                )),
            )
        ]
        if years:
            return min(years), max(years)
        return RagService._candidate_year_span(item)

    @classmethod
    def _ground_candidates_by_known_subject(
        cls,
        candidates: list[dict],
        signals: QuerySignals,
    ) -> tuple[list[dict], bool]:
        """Reject semantically similar chunks about a different named subject."""
        aliases = expand_known_entity_aliases(signals.primary_entity)
        alias_keys = {
            cls._normalize_search_text(alias)
            for alias in aliases
            if cls._normalize_search_text(alias)
        }
        if not alias_keys:
            return candidates, False

        year_range = known_entity_year_range(signals.primary_entity)
        anchors: list[dict] = []
        for item in candidates:
            searchable = " ".join((
                str(item.get("heading") or ""),
                str(item.get("text") or ""),
                " ".join(str(value) for value in item.get("relatedEntities") or []),
            ))
            normalized = cls._normalize_search_text(searchable)
            item_year_span = cls._candidate_year_span(item)
            overlaps_time_anchor = True
            if year_range and item_year_span:
                overlaps_time_anchor = (
                    item_year_span[0] <= year_range[1]
                    and item_year_span[1] >= year_range[0]
                )
            if (
                overlaps_time_anchor
                and any(alias in normalized for alias in alias_keys)
            ):
                anchors.append(item)

        if not anchors:
            return candidates, False

        anchor_pages = {
            (str(item.get("sourceId") or ""), int(item.get("pageStart") or 0))
            for item in anchors
        }
        grounded: list[dict] = []
        seen: set[str] = set()
        for item in candidates:
            item_id = str(item.get("id") or "")
            source_id = str(item.get("sourceId") or "")
            page = int(item.get("pageStart") or 0)
            is_anchor = any(
                str(anchor.get("id") or "") == item_id
                for anchor in anchors
            )
            is_adjacent = any(
                source_id == anchor_source and abs(page - anchor_page) <= 1
                for anchor_source, anchor_page in anchor_pages
            )
            if (is_anchor or is_adjacent) and item_id not in seen:
                grounded.append(item)
                seen.add(item_id)
        return grounded, True

    @classmethod
    def _filter_results_by_known_period(
        cls,
        results: list[dict],
        signals: QuerySignals,
    ) -> tuple[list[dict], bool]:
        """Reject retrospective volumes outside a curated event period.

        A later facet/recovery path can reintroduce a chunk which only recalls
        a well-known event.  Explicit source ranges such as ``1965-1975`` must
        not become primary evidence for the 1954 Điện Biên Phủ event.
        """
        year_range = known_entity_year_range(signals.primary_entity)
        if not year_range:
            return results, False

        kept: list[dict] = []
        removed = False
        for item in results:
            item_year_span = cls._candidate_year_span(item)

            outside_period = (
                item_year_span is not None
                and (
                    item_year_span[0] > year_range[1]
                    or item_year_span[1] < year_range[0]
                )
            )
            if outside_period:
                removed = True
                continue
            kept.append(item)

        # Legacy titles can be malformed. Do not suppress every available
        # piece of evidence solely because of source-title metadata.
        return (kept, removed) if kept else (results, False)

    @classmethod
    def _filter_comparison_period_candidates(
        cls,
        candidates: list[dict],
        plan: QueryPlan,
    ) -> tuple[list[dict], bool]:
        """Remove sources explicitly outside both sides of a dated comparison."""
        if plan.intent != "comparison" or not plan.comparison_date_ranges:
            return candidates, False

        kept: list[dict] = []
        removed = False
        for item in candidates:
            span = cls._candidate_year_span(item)
            if span is None:
                kept.append(item)
                continue
            overlaps_a_side = any(
                span[0] <= end and span[1] >= start
                for start, end in plan.comparison_date_ranges
            )
            if overlaps_a_side:
                kept.append(item)
            else:
                removed = True
        return kept, removed

    @classmethod
    def _filter_evolution_period_candidates(
        cls,
        candidates: list[dict],
        plan: QueryPlan,
    ) -> tuple[list[dict], bool]:
        """Reject chunks wholly outside an explicit F9 time range."""
        if (
            plan.intent != "historical_evolution"
            or not plan.explicit_date_range
        ):
            return candidates, False
        years = [
            int(value)
            for value in re.findall(r"\b\d{4}\b", plan.explicit_date_range)
        ]
        if len(years) < 2:
            return candidates, False
        start, end = min(years), max(years)
        kept: list[dict] = []
        removed = False
        for item in candidates:
            span = cls._candidate_content_year_span(item)
            if span is None or (span[0] <= end and span[1] >= start):
                kept.append(item)
            else:
                removed = True
        return (kept, removed) if kept else (candidates, False)

    @classmethod
    def _requested_query_facets(
        cls,
        question: str,
        plan: QueryPlan | None = None,
    ) -> list[str]:
        if plan and plan.required_facets:
            return list(plan.required_facets)
        normalized_question = cls._normalize_search_text(question)
        return [
            facet for triggers, facet in QUERY_FACETS
            if any(trigger in normalized_question for trigger in triggers)
        ]

    @classmethod
    def _facet_evidence_map(
        cls,
        facets: list[str] | tuple[str, ...],
        results: list[dict],
    ) -> dict[str, list[str]]:
        evidence: dict[str, list[str]] = {facet: [] for facet in facets}
        for result in results:
            result_id = str(result.get("id") or "").strip()
            if not result_id:
                continue
            searchable = " ".join((
                str(result.get("sourceTitle") or ""),
                str(result.get("heading") or ""),
                str(result.get("text") or ""),
            ))
            indexed_facets = list(result.get("facets") or [])
            for facet in facets:
                if text_supports_facet(facet, searchable, indexed_facets):
                    evidence[facet].append(result_id)
        return evidence

    @classmethod
    def _concrete_evidence_score(cls, result: dict) -> float:
        """Estimate whether a chunk can teach through facts, not only abstractions."""
        text = " ".join((
            str(result.get("heading") or ""),
            str(result.get("text") or ""),
        ))
        normalized = cls._normalize_search_text(text)
        score = 0.0

        entities = {
            cls._normalize_search_text(str(entity))
            for entity in (result.get("relatedEntities") or [])
            if cls._normalize_search_text(str(entity))
        }
        if entities:
            score += min(0.24, 0.12 + 0.04 * len(entities))
        if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text):
            score += 0.14
        if re.search(r"\b\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?\b", text):
            score += 0.08
        if re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:%|ha|km|tan|nguoi|dong|trieu|ty|van|mau|"
            r"su doan|dai doan|trung doan|don dien|nha may)\b",
            normalized,
        ):
            score += 0.24
        if (
            ":" in text
            or ";" in text
            or any(marker in normalized for marker in (
                "vi du", "chang han", "cu the", "bao gom", "gom co",
                "chia thanh", "lan thu nhat", "thu nhat", "thu hai",
            ))
        ):
            score += 0.14
        action_count = sum(
            marker in normalized
            for marker in (
                "thanh lap", "ban hanh", "dat ra", "xay dung", "to chuc",
                "ky ket", "chia thanh", "chi huy", "giai phong", "dan ap",
            )
        )
        if action_count >= 2:
            score += 0.12
        return round(min(1.0, score), 3)

    @classmethod
    def _facet_detail_evidence_map(
        cls,
        facets: list[str] | tuple[str, ...],
        results: list[dict],
    ) -> dict[str, list[str]]:
        """Map facets to chunks that contain usable concrete illustrations."""
        evidence: dict[str, list[str]] = {facet: [] for facet in facets}
        for result in results:
            result_id = str(result.get("id") or "").strip()
            if not result_id or cls._concrete_evidence_score(result) < 0.32:
                continue
            searchable = " ".join((
                str(result.get("sourceTitle") or ""),
                str(result.get("heading") or ""),
                str(result.get("text") or ""),
            ))
            indexed_facets = list(result.get("facets") or [])
            for facet in facets:
                detail_terms = cls._salient_terms(facet_detail_search_text(facet))
                detail_overlap = len(
                    detail_terms & cls._salient_terms(searchable)
                )
                if (
                    detail_overlap >= 2
                    and text_supports_facet(facet, searchable, indexed_facets)
                ):
                    evidence[facet].append(result_id)
        return evidence

    @classmethod
    def _detail_analysis(
        cls,
        facets: list[str] | tuple[str, ...],
        results: list[dict],
    ) -> dict:
        concrete_count = sum(
            cls._concrete_evidence_score(result) >= 0.32
            for result in results
        )
        if facets:
            detail_map = cls._facet_detail_evidence_map(facets, results)
            concrete_facets = [facet for facet, ids in detail_map.items() if ids]
            missing = [facet for facet, ids in detail_map.items() if not ids]
            coverage = len(concrete_facets) / len(facets)
        else:
            concrete_facets = []
            missing = []
            coverage = min(1.0, concrete_count / min(2, max(1, len(results))))
        return {
            "concrete_evidence_count": concrete_count,
            "concrete_facets": concrete_facets,
            "missing_concrete_facets": missing,
            "detail_coverage": round(coverage, 3),
        }

    @classmethod
    def _ground_event_phase_results(
        cls,
        results: list[dict],
        candidates: list[dict],
        signals: QuerySignals,
        top_k: int,
    ) -> list[dict]:
        """Reject cross-event phase matches unless the requested event has dates."""
        subject = cls._normalize_search_text(signals.primary_entity)
        if not subject:
            return []

        pool: list[dict] = []
        seen: set[str] = set()
        for item in [*results, *candidates]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            pool.append(item)

        phase_pattern = re.compile(
            r"\b(?:dot|giai doan)\s+(?:thu\s+)?"
            r"(?:[1-9]|mot|hai|ba|bon|nam|sau|bay|tam|chin)\b"
        )
        date_pattern = re.compile(
            r"\b\d{1,2}\s*[-/.]\s*\d{1,2}"
            r"(?:\s*[-/.]\s*(?:\d{2}|\d{4}))?\b"
        )
        anchors: list[dict] = []
        for item in pool:
            raw_text = " ".join((
                str(item.get("heading") or ""),
                str(item.get("text") or ""),
            ))
            normalized = cls._normalize_search_text(raw_text)
            if (
                subject in normalized
                and phase_pattern.search(normalized)
                and date_pattern.search(raw_text)
            ):
                anchors.append(item)

        if not anchors:
            return []

        anchor_pages = {
            (
                str(item.get("sourceId") or ""),
                int(item.get("pageStart") or 0),
            )
            for item in anchors
        }
        grounded: list[dict] = []
        for item in pool:
            normalized = cls._normalize_search_text(" ".join((
                str(item.get("heading") or ""),
                str(item.get("text") or ""),
            )))
            source_id = str(item.get("sourceId") or "")
            page = int(item.get("pageStart") or 0)
            subject_match = subject in normalized
            adjacent_to_anchor = any(
                source_id == anchor_source and abs(page - anchor_page) <= 2
                for anchor_source, anchor_page in anchor_pages
            )
            if subject_match or adjacent_to_anchor:
                grounded.append(item)
            if len(grounded) >= top_k:
                break
        return grounded

    @classmethod
    def _estimate_evidence_coverage(
        cls,
        question: str,
        results: list[dict],
        requested_facets: list[str] | tuple[str, ...] | None = None,
    ) -> float:
        """Estimate how many explicitly requested facets have direct evidence."""
        requested_facets = list(requested_facets or cls._requested_query_facets(question))
        if not results:
            return 0.0
        if not requested_facets:
            return 1.0
        evidence = cls._facet_evidence_map(requested_facets, results)
        covered = sum(1 for facet in requested_facets if evidence.get(facet))
        return covered / len(requested_facets)

    def _calibrate_confidence(
        self,
        signals: QuerySignals,
        results: list[dict],
        analysis: dict,
        question: str = "",
        requested_facets: list[str] | tuple[str, ...] = (),
    ) -> tuple[float, dict[str, float]]:
        """Score evidence quality instead of trusting retrieval rank alone."""
        if not results:
            factors = {
                "retrieval_quality": 0.0,
                "evidence_coverage": 0.0,
                "direct_evidence": 0.0,
                "entity_match": 0.0,
                "trusted_source_ratio": 0.0,
                "page_diversity": 0.0,
                "detail_coverage": 0.0,
            }
            return 0.0, factors

        top_results = results[:4]
        retrieval_quality = min(
            1.0,
            sum(float(item.get("score") or 0) for item in top_results)
            / len(top_results),
        )
        deterministic_coverage = self._estimate_evidence_coverage(
            question,
            results,
            requested_facets,
        )
        reported_coverage = analysis.get("evidence_coverage")
        if requested_facets or not isinstance(reported_coverage, (int, float)):
            reported_coverage = deterministic_coverage
        evidence_coverage = max(0.0, min(1.0, float(reported_coverage)))
        direct_evidence = 1.0 if any(
            item.get("directEvidence") for item in results
        ) else 0.0
        entity_match = 1.0 if (
            not signals.primary_entity
            or any(item.get("entityMatch") for item in results)
        ) else 0.0
        trusted_source_ratio = sum(
            1 for item in results
            if str(item.get("trustLevel") or "").casefold() == "official"
        ) / len(results)
        unique_pages = {
            (str(item.get("sourceId") or ""), int(item.get("pageStart") or 0))
            for item in results
        }
        page_diversity = min(1.0, len(unique_pages) / min(2, len(results)))
        detail_coverage = float(
            self._detail_analysis(requested_facets, results)["detail_coverage"]
        )

        factors = {
            "retrieval_quality": retrieval_quality,
            "evidence_coverage": evidence_coverage,
            "direct_evidence": direct_evidence,
            "entity_match": entity_match,
            "trusted_source_ratio": trusted_source_ratio,
            "page_diversity": page_diversity,
            "detail_coverage": detail_coverage,
        }
        confidence = (
            0.25 * retrieval_quality
            + 0.25 * evidence_coverage
            + 0.15 * detail_coverage
            + 0.15 * direct_evidence
            + 0.10 * entity_match
            + 0.05 * trusted_source_ratio
            + 0.05 * page_diversity
        )
        if signals.is_identity_query and not (direct_evidence and entity_match):
            confidence = min(confidence, 0.45)
        if requested_facets and evidence_coverage < 1.0:
            # Không để điểm vector cao che lấp việc câu trả lời rộng còn thiếu ý.
            confidence = min(confidence, 0.45 + 0.40 * evidence_coverage)
        return max(0.0, min(1.0, confidence)), factors

    @classmethod
    def _evidence_fallback(
        cls,
        question: str,
        candidates: list[dict],
        limit: int,
    ) -> list[dict]:
        """Keep lexically strong evidence when the model reranker rejects every ID."""
        question_terms = cls._salient_terms(question)
        if len(question_terms) < 2:
            return []

        ranked: list[tuple[int, float, float, dict]] = []
        for candidate in candidates:
            searchable = " ".join((
                str(candidate.get("sourceTitle") or ""),
                str(candidate.get("heading") or ""),
                str(candidate.get("text") or ""),
            ))
            overlap = len(question_terms & cls._salient_terms(searchable))
            coverage = overlap / len(question_terms)
            # Hai khái niệm trọng tâm và 22% câu hỏi là ngưỡng đủ chặt
            # để không hồi sinh các đoạn chỉ trùng một từ phổ biến.
            if overlap >= 2 and coverage >= 0.22:
                ranked.append((
                    overlap,
                    coverage,
                    float(candidate.get("score") or 0),
                    candidate,
                ))

        ranked.sort(key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ranked[:limit]]

    @staticmethod
    def _recovery_query(
        question: str,
        analysis: dict,
        plan: QueryPlan | None = None,
    ) -> str:
        """Put inferred canonical context first so it survives full-text term limits."""
        additions = [str(analysis.get("scope") or "").strip()]
        if plan is None or plan.allow_inferred_date_recovery or plan.explicit_date_range:
            additions.append(str(analysis.get("date_range") or "").strip())
        parts: list[str] = []
        for part in [*additions, question.strip()]:
            if part and part.casefold() not in {item.casefold() for item in parts}:
                parts.append(part)
        return ". ".join(parts)

    @classmethod
    def _expanded_recovery_queries(
        cls,
        question: str,
        analysis: dict,
        plan: QueryPlan | None = None,
    ) -> list[str]:
        """Create canonical, facet and concrete-evidence retrieval queries."""
        base_query = cls._recovery_query(question, analysis, plan)
        canonical_context = ". ".join(
            part for part in (
                str(analysis.get("scope") or "").strip(),
                str(analysis.get("date_range") or "").strip()
                if plan is None or plan.allow_inferred_date_recovery or plan.explicit_date_range
                else "",
            )
            if part
        )
        queries = [base_query]
        existing = {base_query.casefold()}

        planned_facets = cls._requested_query_facets(question, plan)
        if plan:
            planned_facets = list(dict.fromkeys([
                *planned_facets,
                *plan.optional_facets,
            ]))
        if plan and plan.intent == "comparison" and plan.comparison_subjects:
            for facet in planned_facets:
                for index, subject in enumerate(plan.comparison_subjects):
                    date_context = ""
                    if index < len(plan.comparison_date_ranges):
                        start, end = plan.comparison_date_ranges[index]
                        date_context = f"{start}-{end}"
                    subject_aliases = expand_known_entity_aliases(subject)
                    subject_context = " ; ".join(
                        subject_aliases[:6]
                    ) or subject
                    detail_query = ". ".join(
                        part for part in (
                            facet_detail_search_text(facet),
                            date_context,
                            subject_context,
                        )
                        if part
                    )
                    if detail_query.casefold() not in existing:
                        queries.append(detail_query)
                        existing.add(detail_query.casefold())
        else:
            for facet in planned_facets:
                detail_query = ". ".join(
                    part for part in (
                        facet_detail_search_text(facet),
                        canonical_context,
                        plan.subject if plan else question,
                    )
                    if part
                )
                if detail_query.casefold() not in existing:
                    queries.append(detail_query)
                    existing.add(detail_query.casefold())
            if plan and plan.intent == "historical_evolution":
                # Phase 1 của F9: tìm ứng viên bước ngoặt theo subject/domain.
                # Không định sẵn giai đoạn từ riêng khoảng năm.
                for probe_text in (
                    "bắt đầu kết thúc chuyển sang thay thế bước ngoặt "
                    "thay đổi chính sách chiến lược cơ cấu trạng thái",
                    "hiệp định cải cách khủng hoảng thay đổi chính quyền "
                    "thành lập giải phóng thống nhất tác động quốc tế",
                    "trạng thái đầu kỳ trạng thái cuối kỳ yếu tố tiếp nối "
                    "vòng đời đối tượng quá trình kế tiếp",
                    "đảo chiều nhượng bộ nới lỏng mở rộng tăng tốc "
                    "siết chặt đàn áp thu hẹp suy giảm",
                    "chủ thể ban hành xóa bỏ bãi bỏ thay thế chấm dứt "
                    "thời điểm hành động đối tượng bị tác động",
                    "trực tiếp gián tiếp tập trung phân quyền thời bình "
                    "thời chiến huy động khủng hoảng phục hồi",
                ):
                    probe_query = ". ".join(
                        part for part in (
                            probe_text,
                            plan.evolution_domain,
                            canonical_context,
                            plan.subject,
                        )
                        if part
                    )
                    if probe_query.casefold() not in existing:
                        queries.append(probe_query)
                        existing.add(probe_query.casefold())
                # Phase 2 chỉ xuất hiện sau khi period planner đã tạo các
                # chặng động có evidence ID.
                for period_index, (period_start, period_end) in enumerate(
                    plan.evolution_periods
                ):
                    period_state = (
                        plan.evolution_period_states[period_index]
                        if period_index < len(plan.evolution_period_states)
                        else ""
                    )
                    boundary_context = " ".join((
                        plan.evolution_boundary_actors[period_index]
                        if period_index < len(plan.evolution_boundary_actors)
                        else "",
                        plan.evolution_boundary_actions[period_index]
                        if period_index < len(plan.evolution_boundary_actions)
                        else "",
                    )).strip()
                    period_query = ". ".join(
                        part for part in (
                            f"{period_start}-{period_end}",
                            "giai đoạn bước ngoặt bối cảnh mục tiêu biện pháp "
                            "yếu tố tiếp nối thay đổi nguyên nhân kết quả",
                            period_state,
                            boundary_context,
                            plan.subject,
                        )
                        if part
                    )
                    if period_query.casefold() not in existing:
                        queries.append(period_query)
                        existing.add(period_query.casefold())

        normalized_question = cls._normalize_search_text(question)
        asks_for_explanation = any(marker in normalized_question for marker in (
            "noi cho", "cho biet", "trinh bay", "phan tich", "giai thich",
            "nhu the nao", "tai sao", "ke ve", "tom tat", "tong quan",
        ))
        asks_for_short_answer = any(marker in normalized_question for marker in (
            "ngan gon", "tom gon", "mot cau", "mot dong",
        ))
        if not planned_facets and asks_for_explanation and not asks_for_short_answer:
            detail_query = ". ".join(
                part for part in (
                    "dẫn chứng cụ thể tên nhân vật sự kiện địa điểm "
                    "mốc thời gian biện pháp số liệu kết quả",
                    canonical_context,
                    plan.subject if plan else question,
                )
                if part
            )
            if detail_query.casefold() not in existing:
                queries.append(detail_query)
        return queries

    @classmethod
    def _needs_query_recovery(
        cls,
        question: str,
        analysis: dict,
        has_results: bool,
        plan: QueryPlan | None = None,
    ) -> bool:
        if not has_results:
            return True
        # Câu hỏi nhiều vế cần nhiều truy vấn chuyên biệt, kể cả khi
        # người dùng đã ghi rõ năm.
        requested_facets = cls._requested_query_facets(question, plan)
        if len(requested_facets) >= 2:
            return not (plan and plan.required_facets)
        if plan and not plan.allow_inferred_date_recovery:
            return False
        inferred_context = " ".join((
            str(analysis.get("scope") or ""),
            str(analysis.get("date_range") or ""),
        ))
        inferred_years = set(re.findall(r"\b\d{3,4}\b", inferred_context))
        explicit_years = set(re.findall(r"\b\d{3,4}\b", question))
        # Khi reranker nhận ra một mốc năm mà người dùng không ghi, chạy lại retrieval
        # với mốc chuẩn đó thay vì chấp nhận các đoạn chỉ nhắc thoáng qua sự kiện.
        return bool(inferred_years - explicit_years)

    @staticmethod
    def _merge_retrievals(retrievals: list[dict], candidate_k: int) -> dict:
        """Fuse original and expanded-query ranks without comparing raw channel scores."""
        fused: dict[str, dict] = {}
        per_query_rankings: list[list[str]] = []
        query_terms: list[str] = []
        channels: list[str] = []
        for retrieval in retrievals:
            current_ranking: list[str] = []
            for term in retrieval.get("queryTerms") or []:
                if term not in query_terms:
                    query_terms.append(term)
            for channel in retrieval.get("channels") or []:
                if channel not in channels:
                    channels.append(channel)
            for rank, item in enumerate(retrieval.get("items") or [], start=1):
                item_id = str(item.get("id") or "").strip()
                if not item_id:
                    continue
                current_ranking.append(item_id)
                entry = fused.setdefault(item_id, {
                    **item,
                    "id": item_id,
                    "multiQueryRrf": 0.0,
                    "queryHits": 0,
                    "bestQueryRank": rank,
                })
                merged_channels = list(entry.get("channels") or [])
                for channel in item.get("channels") or []:
                    if channel not in merged_channels:
                        merged_channels.append(channel)
                merged_phrases = list(entry.get("matchedPhrases") or [])
                for phrase in item.get("matchedPhrases") or []:
                    if phrase not in merged_phrases:
                        merged_phrases.append(phrase)
                direct_evidence = bool(
                    entry.get("directEvidence") or item.get("directEvidence")
                )
                entity_match = bool(entry.get("entityMatch") or item.get("entityMatch"))
                entry["multiQueryRrf"] += 1.0 / (60 + rank)
                entry["queryHits"] += 1
                entry["bestQueryRank"] = min(
                    int(entry.get("bestQueryRank") or rank),
                    rank,
                )
                if float(item.get("score") or 0) > float(entry.get("score") or 0):
                    multi_query_rrf = entry["multiQueryRrf"]
                    query_hits = entry["queryHits"]
                    entry.update(item)
                    entry["id"] = item_id
                    entry["multiQueryRrf"] = multi_query_rrf
                    entry["queryHits"] = query_hits
                entry["channels"] = merged_channels
                entry["matchedPhrases"] = merged_phrases
                entry["directEvidence"] = direct_evidence
                entry["entityMatch"] = entity_match
            per_query_rankings.append(current_ranking)

        ordered = sorted(
            fused.values(),
            key=lambda item: (
                float(item["multiQueryRrf"]),
                int(item["queryHits"]),
                float(item.get("score") or 0),
            ),
            reverse=True,
        )
        max_rrf = max((float(item["multiQueryRrf"]) for item in ordered), default=1.0)
        for item in ordered:
            rrf_score = float(item["multiQueryRrf"]) / max_rrf
            best_rank = int(item.get("bestQueryRank") or 999)
            focused_floor = 0.0
            if best_rank == 1:
                focused_floor = 0.62
            elif best_rank == 2:
                focused_floor = 0.50
            item["score"] = round(max(rrf_score, focused_floor), 6)

        # A broad chunk can rank moderately for every query and otherwise crowd out
        # the best concrete chunk of a focused query. Reserve a small quota from
        # each query before filling the rest by reciprocal-rank fusion.
        reserved_ids: list[str] = []
        reserved_seen: set[str] = set()
        quota = 2 if len(per_query_rankings) * 2 <= candidate_k else 1
        for ranking in per_query_rankings:
            for item_id in ranking[:quota]:
                if item_id not in reserved_seen:
                    reserved_ids.append(item_id)
                    reserved_seen.add(item_id)
        retained_ids = reserved_ids[:candidate_k]
        for item in ordered:
            item_id = str(item.get("id") or "")
            if item_id not in reserved_seen:
                retained_ids.append(item_id)
                reserved_seen.add(item_id)
            if len(retained_ids) >= candidate_k:
                break
        retained = [fused[item_id] for item_id in retained_ids if item_id in fused]
        return {
            "items": retained[:candidate_k],
            "candidateCount": len(ordered),
            "queryTerms": query_terms,
            "exactMatchCount": sum(
                1 for item in ordered if item.get("directEvidence")
            ),
            "channels": channels,
        }

    @staticmethod
    def _base_query_analysis(
        question: str,
        normalization: QueryNormalization | None,
    ) -> dict:
        corrections = [
            {
                "original": correction.original,
                "replacement": correction.replacement,
                "reason": correction.reason,
            }
            for correction in (normalization.corrections if normalization else ())
        ]
        return {
            "normalized_question": question,
            "query_corrections": corrections,
            "rewrite_confidence": 1.0,
            "query_ambiguous": bool(normalization and normalization.ambiguous_terms),
            "ambiguity_notes": list(
                normalization.ambiguous_terms if normalization else ()
            ),
        }

    @classmethod
    def _model_query_analysis(
        cls,
        question: str,
        parsed: dict,
        normalization: QueryNormalization | None,
    ) -> dict:
        """Accept a contextual rewrite only when it cannot alter explicit dates."""
        base = cls._base_query_analysis(question, normalization)
        candidate = normalize_query(
            str(parsed.get("normalized_question") or question),
        ).normalized_question
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("rewrite_confidence"))))
        except (TypeError, ValueError):
            confidence = 0.0
        ambiguous = bool(parsed.get("ambiguous"))
        original_years = set(re.findall(r"\b\d{3,4}\b", question))
        candidate_years = set(re.findall(r"\b\d{3,4}\b", candidate))
        length_ratio = len(candidate) / max(1, len(question))
        primary_entity = analyze_query(question).primary_entity
        primary_preserved = (
            not primary_entity
            or cls._normalize_search_text(primary_entity)
            in cls._normalize_search_text(candidate)
        )
        ambiguity_resolved = (
            not base["query_ambiguous"]
            or candidate.casefold() != question.casefold()
        )
        original_token_count = len(cls._normalize_search_text(question).split())
        candidate_token_count = len(cls._normalize_search_text(candidate).split())
        content_growth_allowed = (
            base["query_ambiguous"]
            or candidate_token_count <= original_token_count + 1
        )
        accepted = (
            bool(candidate)
            and confidence >= 0.85
            and not ambiguous
            and original_years == candidate_years
            and 0.45 <= length_ratio <= 1.8
            and primary_preserved
            and ambiguity_resolved
            and content_growth_allowed
        )

        notes = [
            str(note).strip()
            for note in (parsed.get("ambiguity_notes") or [])
            if str(note).strip()
        ]
        if not accepted:
            return {
                **base,
                "rewrite_confidence": confidence,
                "query_ambiguous": ambiguous or base["query_ambiguous"],
                "ambiguity_notes": notes or base["ambiguity_notes"],
            }

        corrections = list(base["query_corrections"])
        model_correction_count = 0
        for correction in parsed.get("corrections") or []:
            if not isinstance(correction, dict):
                continue
            original = str(correction.get("original") or "").strip()
            replacement = str(correction.get("replacement") or "").strip()
            reason = str(correction.get("reason") or "Sửa theo ngữ cảnh").strip()
            if (
                original
                and replacement
                and original != replacement
                and cls._normalize_search_text(original)
                in cls._normalize_search_text(question)
                and cls._normalize_search_text(replacement)
                in cls._normalize_search_text(candidate)
            ):
                item = {
                    "original": original,
                    "replacement": replacement,
                    "reason": reason,
                }
                if item not in corrections:
                    corrections.append(item)
                    model_correction_count += 1
        if candidate != question and model_correction_count == 0:
            corrections.append({
                "original": question,
                "replacement": candidate,
                "reason": "Chuẩn hóa câu hỏi theo ngữ cảnh",
            })
        return {
            "normalized_question": candidate,
            "query_corrections": corrections,
            "rewrite_confidence": confidence,
            "query_ambiguous": False,
            "ambiguity_notes": [],
        }

    @staticmethod
    def _preserve_direct_evidence(
        selected: list[dict],
        candidates: list[dict],
        top_k: int,
        signals: QuerySignals,
    ) -> list[dict]:
        """Keep one exact subject match so reranking cannot erase direct evidence."""
        direct_candidates = [
            item for item in candidates
            if item.get("directEvidence")
            and (not signals.primary_entity or item.get("entityMatch"))
        ]
        if not direct_candidates:
            return selected[:top_k]

        direct = direct_candidates[0]
        direct_id = str(direct.get("id") or "")
        selected_ids = {str(item.get("id") or "") for item in selected}
        if direct_id not in selected_ids:
            selected = [direct, *selected]
        return selected[:top_k]

    @classmethod
    def _select_with_facet_coverage(
        cls,
        selected: list[dict],
        candidates: list[dict],
        required_facets: list[str] | tuple[str, ...],
        top_k: int,
    ) -> list[dict]:
        """Reserve facet slots using concrete detail, then retain reranker order."""

        ordered_pool: list[dict] = []
        seen_pool: set[str] = set()
        for item in [*selected, *candidates]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in seen_pool:
                continue
            seen_pool.add(item_id)
            ordered_pool.append(item)

        chosen: list[dict] = []
        chosen_ids: set[str] = set()
        selected_ids = {
            str(item.get("id") or "") for item in selected
            if str(item.get("id") or "")
        }
        if not required_facets:
            for item in selected:
                item_id = str(item.get("id") or "")
                if not item_id or item_id in chosen_ids:
                    continue
                chosen.append(item)
                chosen_ids.add(item_id)
        for facet in required_facets:
            supported = [
                (index, item)
                for index, item in enumerate(ordered_pool)
                if str(item.get("id") or "") not in chosen_ids
                and text_supports_facet(
                    facet,
                    " ".join((
                        str(item.get("sourceTitle") or ""),
                        str(item.get("heading") or ""),
                        str(item.get("text") or ""),
                    )),
                    list(item.get("facets") or []),
                )
            ]
            detail_terms = cls._salient_terms(facet_detail_search_text(facet))

            def facet_quality(entry: tuple[int, dict]) -> float:
                index, item = entry
                searchable = " ".join((
                    str(item.get("heading") or ""),
                    str(item.get("text") or ""),
                ))
                overlap = len(detail_terms & cls._salient_terms(searchable))
                detail_match = min(1.0, overlap / 4)
                rank_quality = 1 - index / max(1, len(ordered_pool))
                return (
                    0.40 * cls._concrete_evidence_score(item)
                    + 0.35 * detail_match
                    + 0.15 * float(item.get("score") or 0)
                    + 0.05 * rank_quality
                    + 0.05 * (str(item.get("id") or "") in selected_ids)
                )

            candidate = max(supported, key=facet_quality)[1] if supported else None
            if candidate is not None:
                chosen.append(candidate)
                chosen_ids.add(str(candidate.get("id") or ""))

        if required_facets:
            for item in ordered_pool:
                item_id = str(item.get("id") or "")
                if item_id in chosen_ids:
                    continue
                chosen.append(item)
                chosen_ids.add(item_id)
                if len(chosen) >= top_k:
                    break

        # For open-ended questions without explicit facets, retain up to two
        # concrete chunks when the reranker selected only abstract summaries.
        target_concrete = min(2, max(1, top_k // 4))
        concrete_now = sum(
            cls._concrete_evidence_score(item) >= 0.32 for item in chosen
        )
        if concrete_now < target_concrete:
            additions = sorted(
                (
                    item for item in ordered_pool
                    if str(item.get("id") or "") not in chosen_ids
                    and cls._concrete_evidence_score(item) >= 0.40
                    and float(item.get("score") or 0) >= 0.30
                ),
                key=lambda item: (
                    cls._concrete_evidence_score(item),
                    float(item.get("score") or 0),
                ),
                reverse=True,
            )[:target_concrete - concrete_now]
            if additions:
                protected = min(len(required_facets), len(chosen))
                keep_count = max(protected, top_k - len(additions))
                chosen = [*chosen[:keep_count], *additions]
        return chosen[:top_k]

    @classmethod
    def _select_with_evolution_coverage(
        cls,
        selected: list[dict],
        candidates: list[dict],
        plan: QueryPlan,
        top_k: int,
    ) -> list[dict]:
        """Reserve one strong chunk for each known F9 phase, then keep rank."""
        if plan.intent != "historical_evolution" or not plan.evolution_periods:
            return selected[:top_k]

        pool: list[dict] = []
        seen: set[str] = set()
        for item in [*selected, *candidates]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            pool.append(item)

        chosen: list[dict] = []
        chosen_ids: set[str] = set()
        pool_by_id = {
            str(item.get("id") or ""): item for item in pool
        }
        # Khóa trước bằng chứng bước ngoặt đã được kiểm tra excerpt. Nhờ vậy
        # reranker không thể vô tình bỏ đoạn giữ đúng chủ thể – hành động.
        for boundary_id in plan.evolution_boundary_evidence_ids:
            item = pool_by_id.get(str(boundary_id or ""))
            if item is None or str(boundary_id) in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(str(boundary_id))
            if len(chosen) >= top_k:
                return chosen[:top_k]

        evolution_terms = cls._salient_terms(
            facet_detail_search_text("historical_evolution")
        )
        for period_index, (period_start, period_end) in enumerate(
            plan.evolution_periods
        ):
            planned_evidence_ids = (
                set(plan.evolution_period_evidence_ids[period_index])
                if period_index < len(plan.evolution_period_evidence_ids)
                else set()
            )
            supported: list[tuple[float, dict]] = []
            for item in pool:
                item_id = str(item.get("id") or "")
                if item_id in chosen_ids:
                    continue
                span = cls._candidate_content_year_span(item)
                if (
                    span is None
                    or span[0] > period_end
                    or span[1] < period_start
                ):
                    continue
                raw = " ".join((
                    str(item.get("heading") or ""),
                    str(item.get("text") or ""),
                ))
                explicit_years = {
                    int(value)
                    for value in re.findall(r"\b\d{4}\b", raw)
                }
                direct_period_year = any(
                    period_start <= year <= period_end
                    for year in explicit_years
                )
                span_width = max(1, span[1] - span[0] + 1)
                specificity = min(1.0, 12 / span_width)
                term_overlap = min(
                    1.0,
                    len(evolution_terms & cls._salient_terms(raw)) / 4,
                )
                quality = (
                    0.38 * (1.0 if item_id in planned_evidence_ids else 0.0)
                    + 0.22 * (1.0 if direct_period_year else 0.0)
                    + 0.10 * specificity
                    + 0.12 * cls._concrete_evidence_score(item)
                    + 0.10 * term_overlap
                    + 0.08 * float(item.get("score") or 0)
                )
                supported.append((quality, item))
            if supported:
                candidate = max(supported, key=lambda entry: entry[0])[1]
                chosen.append(candidate)
                chosen_ids.add(str(candidate.get("id") or ""))

        for item in [*selected, *pool]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(item_id)
            if len(chosen) >= top_k:
                break
        return chosen[:top_k]

    @classmethod
    def _comparison_side_indexes(
        cls,
        item: dict,
        plan: QueryPlan,
    ) -> tuple[int, ...]:
        """Identify which side of a comparison a chunk directly supports."""
        searchable_terms = cls._salient_terms(" ".join((
            str(item.get("sourceTitle") or ""),
            str(item.get("heading") or ""),
            str(item.get("text") or ""),
            " ".join(str(value) for value in item.get("relatedEntities") or []),
        )))
        ignored = {
            "cuoc", "cua", "phap", "khai", "thac", "thuoc", "dia",
            "lan", "thu", "su", "lich", "viet", "nam", "chien", "tranh",
        }
        ignored.update({
            "diplomatic_agreement": {"hiep", "dinh", "hoi", "nghi"},
            "military_campaign": {"chien", "dich", "tran"},
            "movement_revolution": {
                "phong", "trao", "cach", "mang", "khoi", "nghia",
            },
        }.get(plan.comparison_domain, set()))
        matches: list[int] = []
        for index, subject in enumerate(plan.comparison_subjects):
            variants = tuple(dict.fromkeys((
                subject,
                *expand_known_entity_aliases(subject),
            )))
            variant_matches = []
            for variant in variants:
                subject_terms = cls._salient_terms(variant) - ignored
                if not subject_terms:
                    continue
                overlap = len(subject_terms & searchable_terms)
                variant_matches.append(overlap >= min(2, len(subject_terms)))
            if any(variant_matches):
                matches.append(index)
        if matches:
            return tuple(matches)

        span = cls._candidate_year_span(item)
        if span and plan.comparison_date_ranges:
            overlaps = [
                max(0, min(span[1], end) - max(span[0], start) + 1)
                for start, end in plan.comparison_date_ranges
            ]
            largest_overlap = max(overlaps, default=0)
            if largest_overlap:
                return tuple(
                    index
                    for index, overlap in enumerate(overlaps)
                    if overlap == largest_overlap
                )
        return ()

    @classmethod
    def _select_with_comparison_coverage(
        cls,
        selected: list[dict],
        candidates: list[dict],
        plan: QueryPlan,
        top_k: int,
    ) -> list[dict]:
        """Greedily cover facet × comparison-side cells before filling rank slots."""
        if plan.intent != "comparison" or not plan.comparison_subjects:
            return selected[:top_k]

        pool: list[dict] = []
        seen: set[str] = set()
        for item in [*selected, *candidates]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            pool.append(item)

        selected_ids = {
            str(item.get("id") or "")
            for item in selected
            if str(item.get("id") or "")
        }
        uncovered = {
            (side, facet)
            for side in range(len(plan.comparison_subjects))
            for facet in plan.required_facets
        }
        chosen: list[dict] = []
        chosen_ids: set[str] = set()

        while uncovered and len(chosen) < top_k:
            best_item: dict | None = None
            best_cells: set[tuple[int, str]] = set()
            best_quality = -1.0
            for item in pool:
                item_id = str(item.get("id") or "")
                if item_id in chosen_ids:
                    continue
                sides = cls._comparison_side_indexes(item, plan)
                if not sides:
                    continue
                searchable = " ".join((
                    str(item.get("sourceTitle") or ""),
                    str(item.get("heading") or ""),
                    str(item.get("text") or ""),
                ))
                indexed_facets = list(item.get("facets") or [])
                cells = {
                    (side, facet)
                    for side in sides
                    for facet in plan.required_facets
                    if (side, facet) in uncovered
                    and text_supports_facet(facet, searchable, indexed_facets)
                }
                if not cells:
                    continue
                quality = (
                    len(cells) * 10
                    + (2 if item_id in selected_ids else 0)
                    + cls._concrete_evidence_score(item)
                    + float(item.get("score") or 0)
                )
                if quality > best_quality:
                    best_item = item
                    best_cells = cells
                    best_quality = quality
            if best_item is None:
                break
            chosen.append(best_item)
            chosen_ids.add(str(best_item.get("id") or ""))
            uncovered.difference_update(best_cells)

        for item in [*selected, *candidates]:
            item_id = str(item.get("id") or "")
            if not item_id or item_id in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(item_id)
            if len(chosen) >= top_k:
                break
        return chosen[:top_k]

    @classmethod
    def _comparison_evidence_details(
        cls,
        results: list[dict],
        plan: QueryPlan,
    ) -> dict:
        """Build the auditable object × facet evidence matrix."""
        if plan.intent != "comparison" or not plan.comparison_subjects:
            return {
                "comparison_evidence": {},
                "balanced_facets": [],
                "missing_comparison_cells": [],
            }

        matrix = {
            facet: {
                subject: []
                for subject in plan.comparison_subjects
            }
            for facet in plan.required_facets
        }
        for item in results:
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            sides = cls._comparison_side_indexes(item, plan)
            if not sides:
                continue
            searchable = " ".join((
                str(item.get("sourceTitle") or ""),
                str(item.get("heading") or ""),
                str(item.get("text") or ""),
            ))
            indexed_facets = list(item.get("facets") or [])
            for facet in plan.required_facets:
                if not text_supports_facet(
                    facet,
                    searchable,
                    indexed_facets,
                ):
                    continue
                for side in sides:
                    if side >= len(plan.comparison_subjects):
                        continue
                    subject = plan.comparison_subjects[side]
                    evidence_ids = matrix[facet][subject]
                    if item_id not in evidence_ids:
                        evidence_ids.append(item_id)

        balanced_facets = [
            facet
            for facet, by_subject in matrix.items()
            if by_subject and all(by_subject.values())
        ]
        missing_cells = [
            f"{facet}::{subject}"
            for facet, by_subject in matrix.items()
            for subject, evidence_ids in by_subject.items()
            if not evidence_ids
        ]
        return {
            "comparison_evidence": matrix,
            "balanced_facets": balanced_facets,
            "missing_comparison_cells": missing_cells,
        }

    def _rerank(
        self,
        question: str,
        candidates: list[dict],
        signals: QuerySignals | None = None,
        normalization: QueryNormalization | None = None,
        plan: QueryPlan | None = None,
    ) -> tuple[list[dict], dict]:
        signals = signals or analyze_query(question)
        plan = plan or build_query_plan(question)
        requested_facets = list(plan.required_facets) or self._requested_query_facets(
            question,
            plan,
        )
        top_k = (
            max(
                self.settings.ai_retrieval_top_k,
                min(14, max(6, 2 * len(requested_facets))),
            )
            if plan.intent == "comparison"
            else self.settings.ai_retrieval_top_k
        )
        fallback_results = candidates[:top_k]
        fallback_results = self._preserve_direct_evidence(
            fallback_results,
            candidates,
            top_k,
            signals,
        )
        fallback_results = self._select_with_facet_coverage(
            fallback_results,
            candidates,
            requested_facets,
            top_k,
        )
        fallback_results = self._select_with_comparison_coverage(
            fallback_results,
            candidates,
            plan,
            top_k,
        )
        fallback_facet_coverage = self._facet_evidence_map(
            requested_facets,
            fallback_results,
        )
        fallback_analysis = {
            "intent": plan.intent or "history_lookup",
            "scope": plan.inferred_scope,
            "date_range": plan.explicit_date_range,
            "answerable": bool(candidates),
            "requested_facets": requested_facets,
            "covered_facets": [
                facet for facet, ids in fallback_facet_coverage.items() if ids
            ],
            "facet_coverage": fallback_facet_coverage,
            "missing_facets": [
                facet for facet, ids in fallback_facet_coverage.items() if not ids
            ],
            "answer_structure": plan.answer_structure,
            "evidence_coverage": self._estimate_evidence_coverage(
                question,
                fallback_results,
                requested_facets,
            ),
            **self._base_query_analysis(question, normalization),
            **self._detail_analysis(requested_facets, fallback_results),
        }
        fallback_analysis.update(
            self._comparison_evidence_details(fallback_results, plan)
        )
        if not self.settings.ai_enable_rerank or len(candidates) <= 4:
            return fallback_results, fallback_analysis

        compact_candidates = [
            {
                "id": item["id"],
                "source": item["sourceTitle"],
                "page": item["pageStart"],
                "heading": item.get("heading") or "",
                "facets": item.get("facets") or [],
                "year_start": item.get("yearStart"),
                "year_end": item.get("yearEnd"),
                "source_priority": item.get("sourcePriority"),
                "extraction_status": item.get("extractionStatus") or "",
                "text": item["text"][:900],
                "direct_evidence": bool(item.get("directEvidence")),
                "entity_match": bool(item.get("entityMatch")),
                "matched_phrases": item.get("matchedPhrases") or [],
                "concrete_score": self._concrete_evidence_score(item),
            }
            for item in candidates[:self.settings.ai_retrieval_candidate_k]
        ]
        prompt = f"""
Phân tích câu hỏi lịch sử và chọn tối đa {top_k} đoạn thực sự hữu ích.
Không chọn đoạn chỉ nhắc thoáng qua một tên riêng nhưng không trả lời trọng tâm.
Đặc biệt, không chọn đoạn chỉ chứa mốc năm hoặc nêu tên chiến dịch
như một ví dụ so sánh. Mỗi selected_id phải trực tiếp hỗ trợ ít nhất một
vế mà người dùng yêu cầu.
Ưu tiên độ bao phủ, sự đa dạng nguồn/trang và đúng khoảng thời gian.
Một câu hỏi nhiều ý có thể được trả lời bằng nhiều đoạn khác nhau; không đánh dấu
answerable=false chỉ vì không có một đoạn duy nhất chứa toàn bộ câu trả lời.
Nếu câu hỏi thiếu mốc năm nhưng sự kiện lịch sử là duy nhất, hãy suy ra mốc
chuẩn trong scope/date_range; không buộc người dùng phải gõ năm.
evidence_coverage là tỷ lệ từ 0 đến 1 giữa số vế yêu cầu có bằng chứng
trực tiếp và tổng số vế. Chỉ nhắc tên sự kiện không được tính là đã
bao phủ diễn biến, lực lượng hoặc chỉ huy.
Kế hoạch đã được xác định độc lập trước retrieval và không được tự thu hẹp
theo nội dung của vài ứng viên đầu tiên. Với mỗi required_facet, ưu tiên chọn
ít nhất một đoạn có bằng chứng trực tiếp. Facet không có bằng chứng phải đưa
vào missing_facets, không được coi là đã bao phủ.
Trong các đoạn cùng liên quan, hãy giữ cân bằng giữa đoạn giải thích
khái quát và đoạn có minh họa cụ thể. concrete_score cao cho biết đoạn
có tên riêng, mốc thời gian, số liệu, biện pháp hoặc chuỗi sự kiện;
không vì vậy chọn một đoạn lệch chủ đề. Với câu hỏi rộng, cần ít
nhất hai đoạn cụ thể nếu kho ứng viên có.
Với intent event_phases hoặc answer_structure phases_with_dates_then_result,
hãy chọn tập đoạn cùng bao phủ: mốc bắt đầu–kết thúc của từng đợt/giai đoạn,
diễn biến hoặc mục tiêu chính, nhân vật then chốt, kết quả và ngày kết thúc
toàn sự kiện. Không coi một đoạn chỉ ghi "đợt 1/2/3" mà thiếu ngày tháng là
đã đủ chi tiết nếu ứng viên khác có mốc cụ thể.
Với intent comparison, phải chọn bằng chứng cân bằng cho cả hai đối tượng và
từng required_facet. Không dùng một đoạn của phía thứ nhất để suy đoán phía
thứ hai. Ưu tiên các trang chuyên sâu về từng phương diện thay vì chỉ chọn
trang mở đầu hoặc kết luận chung.
Với intent historical_evolution, phải chọn bằng chứng trải đều period plan
động đã cung cấp, ưu tiên evidence ID đã khóa cho từng bước ngoặt. Không để
nhiều đoạn của một giai đoạn thay thế cho bước ngoặt khác; ưu tiên đoạn cho
biết mốc thời gian, nguyên nhân chuyển đổi, trạng thái mới và yếu tố tiếp nối.
Phải giữ các boundary_evidence_id để khóa đúng chủ thể – hành động – thời điểm.
Ngoài bằng chứng bước ngoặt, ưu tiên thêm một đoạn hỗ trợ trạng thái chủ đạo
của mỗi giai đoạn. Không sửa hoặc tự tạo thêm giai đoạn.

Đồng thời chuẩn hóa câu hỏi tiếng Việt:
- Sửa lỗi gõ, lỗi chính tả, từ lặp và từ viết tắt khi cách hiểu rõ ràng.
- Giữ nguyên tên riêng, tổ chức và toàn bộ mốc năm; không thêm dữ kiện lịch sử.
- Nếu có nhiều cách hiểu, giữ câu hiện tại, đặt ambiguous=true và giải thích ngắn.
- normalized_question phải là câu tự nhiên để dùng cho tìm kiếm và trả lời.

Câu hỏi gốc: {normalization.original_question if normalization else question}
Câu đã làm sạch bằng quy tắc chắc chắn: {question}
Kế hoạch truy xuất: {json.dumps({
    "intent": plan.intent,
    "subject": plan.subject,
    "scope": plan.inferred_scope,
    "explicit_date_range": plan.explicit_date_range,
    "required_facets": requested_facets,
    "optional_facets": list(plan.optional_facets),
    "answer_structure": plan.answer_structure,
    "comparison_intent": plan.comparison_intent,
    "comparison_domain": plan.comparison_domain,
    "comparison_subjects": list(plan.comparison_subjects),
    "explicit_facets": list(plan.explicit_facets),
    "evolution_intent": plan.evolution_intent,
    "evolution_domain": plan.evolution_domain,
    "evolution_subject_type": plan.evolution_subject_type,
    "evolution_periods": list(plan.evolution_periods),
    "evolution_period_labels": list(plan.evolution_period_labels),
    "evolution_period_states": list(plan.evolution_period_states),
    "evolution_boundary_causes": list(plan.evolution_boundary_causes),
    "evolution_boundary_years": list(plan.evolution_boundary_years),
    "evolution_boundary_actors": list(plan.evolution_boundary_actors),
    "evolution_boundary_actions": list(plan.evolution_boundary_actions),
    "evolution_boundary_excerpts": list(plan.evolution_boundary_excerpts),
    "evolution_boundary_evidence_ids": list(
        plan.evolution_boundary_evidence_ids
    ),
    "evolution_period_evidence_ids": [
        list(values) for values in plan.evolution_period_evidence_ids
    ],
}, ensure_ascii=False)}
Ứng viên: {json.dumps(compact_candidates, ensure_ascii=False)}

Trả dữ liệu đúng schema đã yêu cầu, với nội dung tương đương mẫu:
{{
  "intent": "liệt kê/giải thích/so sánh/timeline/kể chuyện/khác",
  "scope": "phạm vi địa lý, tổ chức hoặc đối tượng",
  "date_range": "mốc thời gian nếu có",
  "answerable": true,
  "covered_facets": ["result", "leadership"],
  "facet_coverage": [
    {{"facet": "result", "evidence_ids": ["id-1"]}},
    {{"facet": "leadership", "evidence_ids": ["id-2"]}}
  ],
  "missing_facets": [],
  "evidence_coverage": 0.5,
  "selected_ids": ["id-1", "id-2"],
  "normalized_question": "câu hỏi đã sửa",
  "corrections": [{{"original": "từ sai", "replacement": "từ đúng", "reason": "lý do"}}],
  "rewrite_confidence": 0.98,
  "ambiguous": false,
  "ambiguity_notes": []
}}
""".strip()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=0,
                response_format=RERANK_RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": "Bạn là bộ xếp hạng nguồn RAG chính xác."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            answerable = bool(parsed.get("answerable", True))
            selected_ids = parsed.get("selected_ids")
            selected_ids = selected_ids if isinstance(selected_ids, list) else []
            by_id = {str(item["id"]).strip(): item for item in candidates}
            normalized_ids = [str(item_id).strip() for item_id in selected_ids]
            selected = [by_id[item_id] for item_id in normalized_ids if item_id in by_id]
            selected = self._preserve_direct_evidence(
                selected,
                candidates,
                top_k,
                signals,
            )
            selected = self._select_with_facet_coverage(
                selected,
                candidates,
                requested_facets,
                top_k,
            )
            selected = self._select_with_comparison_coverage(
                selected,
                candidates,
                plan,
                top_k,
            )
            has_direct_subject_evidence = any(
                item.get("directEvidence")
                and (not signals.primary_entity or item.get("entityMatch"))
                for item in candidates
            )
            if has_direct_subject_evidence:
                answerable = True
            deterministic_facet_coverage = self._facet_evidence_map(
                requested_facets,
                selected,
            )
            estimated_coverage = self._estimate_evidence_coverage(
                question,
                selected,
                requested_facets,
            )
            if candidates and (
                not requested_facets
                or any(deterministic_facet_coverage.values())
            ):
                answerable = True
            try:
                model_coverage = float(parsed.get("evidence_coverage"))
            except (TypeError, ValueError):
                model_coverage = estimated_coverage
            final_coverage = estimated_coverage if requested_facets else model_coverage
            model_date_range = str(parsed.get("date_range") or "")
            if not plan.allow_inferred_date_recovery and not plan.explicit_date_range:
                model_date_range = ""
            analysis = {
                "intent": plan.intent or str(parsed.get("intent") or fallback_analysis["intent"]),
                "scope": plan.inferred_scope or str(parsed.get("scope") or ""),
                "date_range": plan.explicit_date_range or model_date_range,
                "answerable": answerable,
                "requested_facets": requested_facets,
                "covered_facets": [
                    facet for facet, ids in deterministic_facet_coverage.items() if ids
                ],
                "facet_coverage": deterministic_facet_coverage,
                "missing_facets": [
                    facet for facet, ids in deterministic_facet_coverage.items() if not ids
                ],
                "answer_structure": plan.answer_structure,
                "evidence_coverage": max(0.0, min(1.0, final_coverage)),
                **self._model_query_analysis(question, parsed, normalization),
                **self._detail_analysis(requested_facets, selected),
            }
            # Khi model khẳng định có thể trả lời nhưng trả ID sai kiểu/sai format,
            # giữ top hybrid thay vì biến một tập nguồn tốt thành kết quả rỗng.
            if answerable and not selected:
                selected = candidates[:top_k]
                selected = self._preserve_direct_evidence(
                    selected,
                    candidates,
                    top_k,
                    signals,
                )
                selected = self._select_with_facet_coverage(
                    selected,
                    candidates,
                    requested_facets,
                    top_k,
                )
                selected = self._select_with_comparison_coverage(
                    selected,
                    candidates,
                    plan,
                    top_k,
                )
                analysis["facet_coverage"] = self._facet_evidence_map(
                    requested_facets,
                    selected,
                )
                analysis["covered_facets"] = [
                    facet for facet, ids in analysis["facet_coverage"].items() if ids
                ]
                analysis["missing_facets"] = [
                    facet for facet, ids in analysis["facet_coverage"].items() if not ids
                ]
                analysis["evidence_coverage"] = self._estimate_evidence_coverage(
                    question,
                    selected,
                    requested_facets,
                )
                analysis.update(self._detail_analysis(requested_facets, selected))
            if not answerable:
                selected = []
            analysis.update(self._comparison_evidence_details(selected, plan))
            return selected[:top_k], analysis
        except Exception:
            return fallback_results, fallback_analysis

    def answer(self, request: ChatRequest, user_id: str = "") -> ChatResponse:
        started_at = time.perf_counter()
        normalization = normalize_query(request.question)
        retrieval_question = normalization.normalized_question
        plan = build_query_plan(retrieval_question)
        plan, adaptive_comparison_planning = self._refine_comparison_plan(
            retrieval_question,
            plan,
        )
        plan, adaptive_evolution_planning = self._refine_evolution_plan(
            retrieval_question,
            plan,
        )
        signals = analyze_query(retrieval_question)
        if plan.intent == "historical_evolution" and plan.subject:
            signals = replace(
                signals,
                primary_entity=plan.subject,
                exact_phrases=tuple(dict.fromkeys((
                    plan.subject,
                    *signals.exact_phrases,
                ))),
            )
        adaptive_top_k = max(
            self.settings.ai_retrieval_top_k,
            (
                min(14, max(6, 2 * len(plan.required_facets)))
                if plan.intent == "comparison"
                else min(
                    12,
                    max(
                        8,
                        len(plan.evolution_periods) + 3,
                        len(plan.required_facets) + 2,
                    ),
                )
            ),
        )
        retrieval_candidate_k = (
            max(
                self.settings.ai_retrieval_candidate_k,
                min(48, max(24, 3 * adaptive_top_k)),
            )
            if plan.intent in {"comparison", "historical_evolution"}
            else self.settings.ai_retrieval_candidate_k
        )
        retrieval_top_k = (
            adaptive_top_k
            if plan.intent in {"comparison", "historical_evolution"}
            else self.settings.ai_retrieval_top_k
        )
        exact_phrases = self._expanded_exact_phrases(signals)
        question_embedding = self._embed([retrieval_question])[0]
        initial_retrieval = self.repository.retrieve_hybrid(
            retrieval_question,
            question_embedding,
            retrieval_candidate_k,
            exact_phrases=exact_phrases,
            primary_entity=signals.primary_entity,
        )
        retrieval = initial_retrieval
        retrieval_strategy = "hybrid_vector_fulltext_rerank"
        if plan.intent == "comparison":
            retrieval_strategy += "_comparison_matrix"
            if adaptive_comparison_planning:
                retrieval_strategy += "_adaptive_criteria"
        elif plan.intent == "historical_evolution":
            retrieval_strategy += "_evolution_timeline"
            if adaptive_evolution_planning:
                retrieval_strategy += "_adaptive_f9"
        planned_queries = self._expanded_recovery_queries(
            retrieval_question,
            {
                "scope": plan.inferred_scope,
                "date_range": plan.explicit_date_range,
            },
            plan,
        )[1:]
        if planned_queries:
            planned_embeddings = self._embed(planned_queries)
            planned_retrievals = [
                self.repository.retrieve_hybrid(
                    planned_query,
                    planned_embedding,
                    retrieval_candidate_k,
                    exact_phrases=exact_phrases,
                    primary_entity=signals.primary_entity,
                )
                for planned_query, planned_embedding in zip(
                    planned_queries,
                    planned_embeddings,
                    strict=True,
                )
            ]
            retrieval = self._merge_retrievals(
                [initial_retrieval, *planned_retrievals],
                retrieval_candidate_k,
            )
            retrieval_strategy += (
                "_facet_detail_expansion"
                if plan.required_facets
                else "_detail_expansion"
            )
        candidates = [
            result for result in retrieval["items"]
            if float(result["score"]) >= self.settings.ai_min_relevance_score
        ]
        candidates, comparison_period_filtered = (
            self._filter_comparison_period_candidates(candidates, plan)
        )
        if comparison_period_filtered:
            retrieval_strategy += "_comparison_period_gate"
        candidates, evolution_period_filtered = (
            self._filter_evolution_period_candidates(candidates, plan)
        )
        if evolution_period_filtered:
            retrieval_strategy += "_evolution_period_gate"
        dynamic_evolution_period_planning = False
        if plan.intent == "historical_evolution":
            timeline_items = self._retrieve_evolution_timeline_coverage(
                retrieval_question,
                plan,
            )
            if timeline_items:
                timeline_groups: dict[tuple[int, int], list[dict]] = {}
                for item in timeline_items:
                    key = (
                        int(item.get("timelineWindowStart") or 0),
                        int(item.get("timelineWindowEnd") or 0),
                    )
                    timeline_groups.setdefault(key, []).append(item)
                timeline_retrievals = [
                    {
                        "items": items,
                        "candidateCount": len(items),
                        "queryTerms": [
                            str(window_start),
                            str(window_end),
                        ],
                        "channels": ["timeline_window"],
                    }
                    for (window_start, window_end), items
                    in timeline_groups.items()
                ]
                retrieval = self._merge_retrievals(
                    [retrieval, *timeline_retrievals],
                    max(
                        retrieval_candidate_k,
                        min(64, len(timeline_items) + 24),
                    ),
                )
                candidates = [
                    result for result in retrieval["items"]
                    if float(result["score"])
                    >= self.settings.ai_min_relevance_score
                ]
                candidates, timeline_period_filtered = (
                    self._filter_evolution_period_candidates(
                        candidates,
                        plan,
                    )
                )
                if timeline_period_filtered:
                    retrieval_strategy += "_evolution_period_gate"
                retrieval_strategy += "_timeline_window_coverage"
            plan, dynamic_evolution_period_planning = (
                self._derive_evolution_period_plan(
                    retrieval_question,
                    plan,
                    candidates,
                    signals,
                )
            )
            if dynamic_evolution_period_planning:
                retrieval_strategy += "_dynamic_periods"
                # Sau khi có số chặng thật mới biết ngân sách bằng chứng cần
                # thiết. Giữ tối đa 12 chunk để có cả boundary và state
                # evidence, thay vì luôn ép mọi F9 vào 8 chunk.
                retrieval_top_k = max(
                    retrieval_top_k,
                    min(
                        12,
                        max(
                            len(plan.evolution_periods) + 3,
                            2 * len(plan.evolution_periods),
                        ),
                    ),
                )
                already_queried = {
                    value.casefold()
                    for value in (retrieval_question, *planned_queries)
                }
                dynamic_queries = [
                    value
                    for value in self._expanded_recovery_queries(
                        retrieval_question,
                        {
                            "scope": plan.inferred_scope,
                            "date_range": plan.explicit_date_range,
                        },
                        plan,
                    )[1:]
                    if value.casefold() not in already_queried
                ]
                if dynamic_queries:
                    dynamic_embeddings = self._embed(dynamic_queries)
                    dynamic_retrievals = [
                        self.repository.retrieve_hybrid(
                            dynamic_query,
                            dynamic_embedding,
                            retrieval_candidate_k,
                            exact_phrases=exact_phrases,
                            primary_entity=signals.primary_entity,
                        )
                        for dynamic_query, dynamic_embedding in zip(
                            dynamic_queries,
                            dynamic_embeddings,
                            strict=True,
                        )
                    ]
                    retrieval = self._merge_retrievals(
                        [retrieval, *dynamic_retrievals],
                        retrieval_candidate_k,
                    )
                    candidates = [
                        result for result in retrieval["items"]
                        if float(result["score"])
                        >= self.settings.ai_min_relevance_score
                    ]
                    candidates, dynamic_period_filtered = (
                        self._filter_evolution_period_candidates(
                            candidates,
                            plan,
                        )
                    )
                    if dynamic_period_filtered:
                        retrieval_strategy += "_evolution_period_gate"
                    retrieval_strategy += "_dynamic_period_expansion"
                revised_plan, coverage_verified = (
                    self._derive_evolution_period_plan(
                        retrieval_question,
                        plan,
                        candidates,
                        signals,
                        coverage_revision=True,
                    )
                )
                if coverage_verified:
                    plan = revised_plan
                    retrieval_strategy += "_periodization_coverage_verified"
                    retrieval_top_k = max(
                        retrieval_top_k,
                        min(
                            12,
                            max(
                                len(plan.evolution_periods) + 3,
                                2 * len(plan.evolution_periods),
                            ),
                        ),
                    )
        pre_ground_candidates = candidates
        candidates, subject_grounded = self._ground_candidates_by_known_subject(
            candidates,
            signals,
        )
        if subject_grounded:
            retrieval_strategy += "_subject_grounding"
            if (
                plan.intent == "historical_evolution"
                and plan.evolution_period_evidence_ids
            ):
                protected_ids = {
                    item_id
                    for period_ids in plan.evolution_period_evidence_ids
                    for item_id in period_ids
                }
                protected_ids.update(
                    plan.evolution_boundary_evidence_ids
                )
                retained_ids = {
                    str(item.get("id") or "") for item in candidates
                }
                candidates.extend(
                    item for item in pre_ground_candidates
                    if str(item.get("id") or "") in protected_ids
                    and str(item.get("id") or "") not in retained_ids
                )
                retrieval_strategy += "_f9_boundary_evidence_guard"
        results, analysis = self._rerank(
            retrieval_question,
            candidates,
            signals,
            normalization,
            plan,
        )

        normalized_question = str(
            analysis.get("normalized_question") or retrieval_question
        ).strip()
        normalized_signals = analyze_query(normalized_question)
        if plan.intent == "historical_evolution" and plan.subject:
            normalized_signals = replace(
                normalized_signals,
                primary_entity=plan.subject,
                exact_phrases=tuple(dict.fromkeys((
                    plan.subject,
                    *normalized_signals.exact_phrases,
                ))),
            )
        normalized_exact_phrases = self._expanded_exact_phrases(
            normalized_signals
        )
        rewrite_changed = (
            normalized_question.casefold() != retrieval_question.casefold()
        )

        if normalization.changed:
            retrieval_strategy += "_deterministic_normalization"
        recovery_query = self._recovery_query(normalized_question, analysis, plan)
        should_recover = self._needs_query_recovery(
            normalized_question,
            analysis,
            bool(results),
            plan,
        )
        if rewrite_changed and (
            not results or float(analysis.get("evidence_coverage") or 0) < 0.8
        ):
            should_recover = True
        if (
            should_recover
            and recovery_query.casefold() != retrieval_question.casefold()
        ):
            initial_results = results
            expanded_queries = self._expanded_recovery_queries(
                normalized_question,
                analysis,
                plan,
            )
            expanded_embeddings = self._embed(expanded_queries)
            expanded_retrievals = [
                self.repository.retrieve_hybrid(
                    expanded_query,
                    expanded_embedding,
                    retrieval_candidate_k,
                    exact_phrases=normalized_exact_phrases,
                    primary_entity=normalized_signals.primary_entity,
                )
                for expanded_query, expanded_embedding in zip(
                    expanded_queries,
                    expanded_embeddings,
                    strict=True,
                )
            ]
            retrieval = self._merge_retrievals(
                expanded_retrievals,
                retrieval_candidate_k,
            )
            candidates = [
                result for result in retrieval["items"]
                if float(result["score"]) >= self.settings.ai_min_relevance_score
            ]
            candidates, recovered_comparison_period_filtered = (
                self._filter_comparison_period_candidates(candidates, plan)
            )
            if recovered_comparison_period_filtered:
                retrieval_strategy += "_comparison_period_gate"
            candidates, recovered_evolution_period_filtered = (
                self._filter_evolution_period_candidates(candidates, plan)
            )
            if recovered_evolution_period_filtered:
                retrieval_strategy += "_evolution_period_gate"
            candidates, recovered_subject_grounded = (
                self._ground_candidates_by_known_subject(
                    candidates,
                    normalized_signals,
                )
            )
            if recovered_subject_grounded:
                retrieval_strategy += "_subject_grounding"
            recovered_results, recovered_analysis = self._rerank(
                normalized_question,
                candidates,
                normalized_signals,
                normalization,
                plan,
            )
            if not recovered_results:
                recovered_results = self._evidence_fallback(
                    recovery_query,
                    candidates,
                    retrieval_top_k,
                )
            results = recovered_results or initial_results
            if recovered_analysis.get("intent"):
                analysis["intent"] = recovered_analysis["intent"]
            analysis["scope"] = recovered_analysis.get("scope") or analysis.get("scope", "")
            analysis["date_range"] = (
                recovered_analysis.get("date_range") or analysis.get("date_range", "")
            )
            analysis["covered_facets"] = recovered_analysis.get("covered_facets") or []
            analysis["requested_facets"] = recovered_analysis.get("requested_facets") or list(plan.required_facets)
            analysis["facet_coverage"] = recovered_analysis.get("facet_coverage") or {}
            analysis["missing_facets"] = recovered_analysis.get("missing_facets") or []
            analysis["answer_structure"] = recovered_analysis.get("answer_structure") or plan.answer_structure
            analysis["evidence_coverage"] = recovered_analysis.get(
                "evidence_coverage",
                self._estimate_evidence_coverage(recovery_query, results),
            )
            for key in (
                "normalized_question",
                "query_corrections",
                "rewrite_confidence",
                "query_ambiguous",
                "ambiguity_notes",
                "concrete_evidence_count",
                "concrete_facets",
                "missing_concrete_facets",
                "detail_coverage",
            ):
                if key in recovered_analysis:
                    analysis[key] = recovered_analysis[key]
            normalized_question = str(
                analysis.get("normalized_question") or normalized_question
            ).strip()
            normalized_signals = analyze_query(normalized_question)
            if plan.intent == "historical_evolution" and plan.subject:
                normalized_signals = replace(
                    normalized_signals,
                    primary_entity=plan.subject,
                    exact_phrases=tuple(dict.fromkeys((
                        plan.subject,
                        *normalized_signals.exact_phrases,
                    ))),
                )
            retrieval_strategy += "_query_recovery"
        elif not results:
            results = self._evidence_fallback(
                normalized_question,
                candidates,
                retrieval_top_k,
            )
            if results:
                retrieval_strategy += "_evidence_fallback"

        if plan.intent == "event_phases":
            results = self._ground_event_phase_results(
                results,
                candidates,
                normalized_signals,
                self.settings.ai_retrieval_top_k,
            )
            retrieval_strategy += "_phase_grounding_gate"
            phase_coverage = self._facet_evidence_map(
                plan.required_facets,
                results,
            )
            analysis["facet_coverage"] = phase_coverage
            analysis["covered_facets"] = [
                facet for facet, ids in phase_coverage.items() if ids
            ]
            analysis["missing_facets"] = [
                facet for facet, ids in phase_coverage.items() if not ids
            ]
            analysis["evidence_coverage"] = self._estimate_evidence_coverage(
                normalized_question,
                results,
                plan.required_facets,
            )
        elif plan.intent == "historical_evolution":
            results = self._select_with_evolution_coverage(
                results,
                candidates,
                plan,
                retrieval_top_k,
            )
            retrieval_strategy += (
                "_period_coverage"
                if plan.evolution_periods
                else "_evolution_facet_coverage"
            )
            evolution_coverage = self._facet_evidence_map(
                plan.required_facets,
                results,
            )
            analysis["facet_coverage"] = evolution_coverage
            analysis["covered_facets"] = [
                facet for facet, ids in evolution_coverage.items() if ids
            ]
            analysis["missing_facets"] = [
                facet for facet, ids in evolution_coverage.items() if not ids
            ]
            analysis["evidence_coverage"] = self._estimate_evidence_coverage(
                normalized_question,
                results,
                plan.required_facets,
            )
        phase_model_fallback = plan.intent == "event_phases" and not results
        if phase_model_fallback:
            retrieval_strategy += "_model_knowledge_fallback"

        results, period_filtered = self._filter_results_by_known_period(
            results,
            normalized_signals,
        )
        if period_filtered:
            retrieval_strategy += "_period_gate"

        final_requested_facets = list(
            analysis.get("requested_facets") or plan.required_facets
        )
        analysis.update(self._detail_analysis(final_requested_facets, results))
        analysis.update(self._comparison_evidence_details(results, plan))

        confidence, confidence_factors = self._calibrate_confidence(
            normalized_signals,
            results,
            analysis,
            normalized_question,
            plan.required_facets,
        )
        diagnostics = RetrievalDiagnostics(
            strategy=retrieval_strategy,
            candidate_count=int(retrieval["candidateCount"]),
            selected_count=len(results),
            intent=analysis["intent"],
            scope=analysis["scope"],
            date_range=analysis["date_range"],
            query_terms=retrieval["queryTerms"],
            primary_entity=normalized_signals.primary_entity,
            exact_match_count=int(retrieval.get("exactMatchCount") or 0),
            channels=list(retrieval.get("channels") or []),
            confidence_factors={
                key: round(value, 3)
                for key, value in confidence_factors.items()
            },
            original_question=normalization.original_question,
            normalized_question=normalized_question,
            query_corrections=list(analysis.get("query_corrections") or []),
            rewrite_confidence=float(analysis.get("rewrite_confidence") or 0),
            query_ambiguous=bool(analysis.get("query_ambiguous")),
            ambiguity_notes=list(analysis.get("ambiguity_notes") or []),
            required_facets=list(plan.required_facets),
            optional_facets=list(plan.optional_facets),
            covered_facets=list(analysis.get("covered_facets") or []),
            missing_facets=list(analysis.get("missing_facets") or []),
            facet_coverage=dict(analysis.get("facet_coverage") or {}),
            concrete_evidence_count=int(
                analysis.get("concrete_evidence_count") or 0
            ),
            concrete_facets=list(analysis.get("concrete_facets") or []),
            missing_concrete_facets=list(
                analysis.get("missing_concrete_facets") or []
            ),
            detail_coverage=float(analysis.get("detail_coverage") or 0),
            answer_structure=str(
                analysis.get("answer_structure") or plan.answer_structure
            ),
            comparison_intent=plan.comparison_intent,
            comparison_domain=plan.comparison_domain,
            comparison_subjects=list(plan.comparison_subjects),
            comparison_object_types=list(plan.comparison_object_types),
            explicit_facets=list(plan.explicit_facets),
            comparison_evidence=dict(
                analysis.get("comparison_evidence") or {}
            ),
            balanced_facets=list(
                analysis.get("balanced_facets") or []
            ),
            missing_comparison_cells=list(
                analysis.get("missing_comparison_cells") or []
            ),
            answer_requirements={
                "includeSimilarities": plan.include_similarities,
                "includeDifferences": plan.include_differences,
                "includeExplanation": plan.include_explanation,
                "includeJudgement": plan.include_judgement,
            },
            evolution_intent=plan.evolution_intent,
            evolution_domain=plan.evolution_domain,
            evolution_subject_type=plan.evolution_subject_type,
            evolution_periods=[
                [start, end] for start, end in plan.evolution_periods
            ],
            evolution_period_labels=list(plan.evolution_period_labels),
            evolution_period_states=list(plan.evolution_period_states),
            evolution_boundary_causes=list(
                plan.evolution_boundary_causes
            ),
            evolution_boundary_years=list(plan.evolution_boundary_years),
            evolution_boundary_actors=list(plan.evolution_boundary_actors),
            evolution_boundary_actions=list(plan.evolution_boundary_actions),
            evolution_boundary_excerpts=list(
                plan.evolution_boundary_excerpts
            ),
            evolution_boundary_evidence_ids=list(
                plan.evolution_boundary_evidence_ids
            ),
            evolution_subject_lifetime=list(
                plan.evolution_subject_lifetime
            ),
            evolution_range_mismatch=plan.evolution_range_mismatch,
            evolution_range_resolution=plan.evolution_range_resolution,
        )

        if not results:
            fallback_answer = ""
            if phase_model_fallback:
                try:
                    fallback_answer = self._answer_event_phases_best_effort(
                        normalized_question,
                    )
                except Exception:
                    logger.exception(
                        "Không thể tạo câu trả lời best-effort cho event_phases"
                    )
            unavailable_answer = (
                "Xin lỗi, kho tri thức hiện chưa có đủ thông tin đáng tin cậy "
                "để trả lời câu hỏi này."
            )
            self._log_query({
                "question": request.question,
                "normalizedQuestion": normalized_question,
                "queryCorrections": [
                    f"{item.get('original', '')} -> {item.get('replacement', '')}"
                    for item in diagnostics.query_corrections
                ],
                "queryAmbiguous": diagnostics.query_ambiguous,
                "answerable": bool(fallback_answer),
                "grounded": False,
                "modelKnowledgeFallback": bool(fallback_answer),
                "intent": analysis["intent"],
                "comparisonIntent": diagnostics.comparison_intent,
                "comparisonDomain": diagnostics.comparison_domain,
                "comparisonSubjects": diagnostics.comparison_subjects,
                "comparisonObjectTypes": diagnostics.comparison_object_types,
                "explicitFacets": diagnostics.explicit_facets,
                "balancedFacets": diagnostics.balanced_facets,
                "missingComparisonCells": diagnostics.missing_comparison_cells,
                "answerStructure": diagnostics.answer_structure,
                "answerRequirements": diagnostics.answer_requirements,
                "evolutionIntent": diagnostics.evolution_intent,
                "evolutionDomain": diagnostics.evolution_domain,
                "evolutionSubjectType": diagnostics.evolution_subject_type,
                "evolutionPeriods": diagnostics.evolution_periods,
                "evolutionPeriodLabels": diagnostics.evolution_period_labels,
                "evolutionPeriodStates": diagnostics.evolution_period_states,
                "evolutionBoundaryCauses": diagnostics.evolution_boundary_causes,
                "evolutionBoundaryYears": diagnostics.evolution_boundary_years,
                "evolutionBoundaryActors": diagnostics.evolution_boundary_actors,
                "evolutionBoundaryActions": diagnostics.evolution_boundary_actions,
                "evolutionBoundaryEvidenceIds": (
                    diagnostics.evolution_boundary_evidence_ids
                ),
                "evolutionSubjectLifetime": diagnostics.evolution_subject_lifetime,
                "evolutionRangeMismatch": diagnostics.evolution_range_mismatch,
                "evolutionRangeResolution": diagnostics.evolution_range_resolution,
                "candidateCount": diagnostics.candidate_count,
                "selectedCount": 0,
                "latencyMs": round((time.perf_counter() - started_at) * 1000),
                "userHash": hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else "",
            })
            return ChatResponse(
                answer=fallback_answer or unavailable_answer,
                citations=[],
                confidence=0.45 if fallback_answer else 0,
                retrieval=diagnostics,
            )

        context_blocks = []
        citations = []
        related_entities: set[str] = set()
        all_plan_facets = list(dict.fromkeys([
            *plan.required_facets,
            *plan.optional_facets,
        ]))
        for index, result in enumerate(results, start=1):
            page_label = str(result["pageStart"])
            if result["pageEnd"] and result["pageEnd"] != result["pageStart"]:
                page_label += f"-{result['pageEnd']}"
            supported_facets = [
                facet for facet in all_plan_facets
                if text_supports_facet(
                    facet,
                    " ".join((
                        str(result.get("sourceTitle") or ""),
                        str(result.get("heading") or ""),
                        str(result.get("text") or ""),
                    )),
                    list(result.get("facets") or []),
                )
            ]
            facet_note = ", ".join(facet_label(facet) for facet in supported_facets)
            facet_line = f"\nPhương diện: {facet_note}" if facet_note else ""
            comparison_sides = (
                self._comparison_side_indexes(result, plan)
                if plan.intent == "comparison"
                else ()
            )
            comparison_labels = [
                plan.comparison_subjects[side]
                for side in comparison_sides
                if side < len(plan.comparison_subjects)
            ]
            comparison_line = (
                f"\nĐối tượng được hỗ trợ: {', '.join(comparison_labels)}"
                if comparison_labels
                else ""
            )
            concrete_score = self._concrete_evidence_score(result)
            concrete_line = (
                "\nLoại bằng chứng: có chi tiết cụ thể để minh họa"
                if concrete_score >= 0.32
                else "\nLoại bằng chứng: khái quát/giải thích"
            )
            context_blocks.append(
                f"[Nguồn {index}] {result['sourceTitle']}, trang {page_label}"
                f"{facet_line}"
                f"{comparison_line}"
                f"{concrete_line}"
                f"\n{result['text']}"
            )
            citations.append(Citation(
                chunk_id=str(result.get("id") or ""),
                source_id=result["sourceId"],
                source_title=result["sourceTitle"],
                page_start=result["pageStart"],
                page_end=result["pageEnd"],
                excerpt=result["text"][:260] + ("..." if len(result["text"]) > 260 else ""),
                score=round(float(result["score"]), 4),
                facets=[facet_label(facet) for facet in supported_facets],
            ))
            related_entities.update(result.get("relatedEntities") or [])

        history = "\n".join(
            f"{'Người dùng' if message.role == 'user' else 'Trợ lý'}: {message.content}"
            for message in request.messages[-6:]
        )
        active_prompt = self._active_prompt_layers()
        # Các hằng trên chỉ là bootstrap fallback. Khi admin kích hoạt một phiên bản,
        # cả năm lớp prompt đều được đọc từ cấu hình active trong Neo4j.
        system_prompt = active_prompt["system"]
        query_normalization_instruction = active_prompt["query_normalization"]
        answer_planning_instruction = active_prompt["answer_planning"]
        presentation_instruction = active_prompt["presentation"]
        output_contract = active_prompt["output_contract"]
        prompt = f"""
{presentation_instruction}

{query_normalization_instruction}

{answer_planning_instruction}

PHÂN TÍCH CÂU HỎI:
- Ý định: {analysis['intent']}
- Phạm vi: {analysis['scope'] or 'Tự xác định từ câu hỏi và nguồn'}
- Thời gian: {analysis['date_range'] or 'Không nêu rõ'}
- Đối tượng so sánh: {', '.join(plan.comparison_subjects) or 'Không áp dụng'}
- Kiểu yêu cầu so sánh: {plan.comparison_intent or 'Không áp dụng'}
- Miền đối tượng: {plan.comparison_domain or 'Không áp dụng'}
- Tiêu chí người dùng nêu rõ: {', '.join(facet_label(facet) for facet in plan.explicit_facets) or 'Không có'}
- Kiểu tiếp nối/thay đổi F9: {plan.evolution_intent or 'Không áp dụng'}
- Miền tiến trình F9: {plan.evolution_domain or 'Không áp dụng'}
- Loại đối tượng F9: {plan.evolution_subject_type or 'Không áp dụng'}
- Period plan động: {json.dumps([
    {
        "range": period,
        "label": plan.evolution_period_labels[index]
        if index < len(plan.evolution_period_labels) else "",
        "dominantState": plan.evolution_period_states[index]
        if index < len(plan.evolution_period_states) else "",
        "boundaryCause": plan.evolution_boundary_causes[index]
        if index < len(plan.evolution_boundary_causes) else "",
        "boundaryYear": plan.evolution_boundary_years[index]
        if index < len(plan.evolution_boundary_years) else None,
        "boundaryActor": plan.evolution_boundary_actors[index]
        if index < len(plan.evolution_boundary_actors) else "",
        "boundaryAction": plan.evolution_boundary_actions[index]
        if index < len(plan.evolution_boundary_actions) else "",
        "boundaryEvidenceExcerpt": plan.evolution_boundary_excerpts[index]
        if index < len(plan.evolution_boundary_excerpts) else "",
    }
    for index, period in enumerate(plan.evolution_periods)
], ensure_ascii=False) if plan.evolution_periods else 'Không đủ bằng chứng để phân kỳ động'}
- Trạng thái kiểm định phân kỳ F9: {
    'Đã kiểm định độ phủ timeline'
    if plan.intent == 'historical_evolution' and plan.evolution_periods
    else (
        'Không được tự phát minh period; chỉ nêu các thay đổi có bằng chứng'
        if plan.intent == 'historical_evolution'
        else 'Không áp dụng'
    )
}
- Vòng đời đối tượng: {json.dumps(plan.evolution_subject_lifetime, ensure_ascii=False) if plan.evolution_subject_lifetime else 'Chưa xác định'}
- Khoảng hỏi vượt vòng đời: {'Có — ' + plan.evolution_range_resolution if plan.evolution_range_mismatch else 'Không'}
- Điểm còn mơ hồ: {'; '.join(diagnostics.ambiguity_notes) or 'Không có'}

KẾ HOẠCH ĐỘ BAO PHỦ:
- Cấu trúc: {diagnostics.answer_structure}
- Yêu cầu đầu ra: {json.dumps(diagnostics.answer_requirements, ensure_ascii=False)}
- Phương diện bắt buộc: {', '.join(facet_label(facet) for facet in diagnostics.required_facets) or 'Không có yêu cầu đặc biệt'}
- Đã có bằng chứng: {', '.join(facet_label(facet) for facet in diagnostics.covered_facets) or 'Chưa có'}
- Còn thiếu bằng chứng: {', '.join(facet_label(facet) for facet in diagnostics.missing_facets) or 'Không thiếu'}
- Phương diện đủ bằng chứng cho cả hai phía: {', '.join(facet_label(facet) for facet in diagnostics.balanced_facets) or 'Chưa có'}
- Ô đối tượng × facet còn thiếu: {', '.join(diagnostics.missing_comparison_cells) or 'Không thiếu'}
- Số đoạn có minh họa cụ thể: {diagnostics.concrete_evidence_count}
- Phương diện có minh họa: {', '.join(facet_label(facet) for facet in diagnostics.concrete_facets) or 'Chưa xác định'}
- Phương diện thiếu minh họa: {', '.join(facet_label(facet) for facet in diagnostics.missing_concrete_facets) or 'Không thiếu'}

{output_contract}

HỘI THOẠI GẦN ĐÂY:
{history or '(Không có)'}

CÂU HỎI ĐÃ CHUẨN HÓA:
{normalized_question}

NGUỒN TRUY XUẤT:
{chr(10).join(context_blocks)}
""".strip()

        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.15,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n{query_normalization_instruction}"
                        f"\n\n{answer_planning_instruction}"
                        f"\n\n{output_contract}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = naturalize_source_meta_language(strip_inline_citations(
            response.choices[0].message.content or ""
        ))
        retrieval_quality = confidence_factors["retrieval_quality"]
        evidence_coverage = confidence_factors["evidence_coverage"]
        self._log_query({
            "question": request.question,
            "normalizedQuestion": normalized_question,
            "queryCorrections": [
                f"{item.get('original', '')} -> {item.get('replacement', '')}"
                for item in diagnostics.query_corrections
            ],
            "rewriteConfidence": diagnostics.rewrite_confidence,
            "queryAmbiguous": diagnostics.query_ambiguous,
            "answerable": True,
            "intent": analysis["intent"],
            "scope": analysis["scope"],
            "dateRange": analysis["date_range"],
            "comparisonIntent": diagnostics.comparison_intent,
            "comparisonDomain": diagnostics.comparison_domain,
            "comparisonSubjects": diagnostics.comparison_subjects,
            "comparisonObjectTypes": diagnostics.comparison_object_types,
            "explicitFacets": diagnostics.explicit_facets,
            "comparisonEvidence": diagnostics.comparison_evidence,
            "balancedFacets": diagnostics.balanced_facets,
            "missingComparisonCells": diagnostics.missing_comparison_cells,
            "answerStructure": diagnostics.answer_structure,
            "answerRequirements": diagnostics.answer_requirements,
            "evolutionIntent": diagnostics.evolution_intent,
            "evolutionDomain": diagnostics.evolution_domain,
            "evolutionSubjectType": diagnostics.evolution_subject_type,
            "evolutionPeriods": diagnostics.evolution_periods,
            "evolutionPeriodLabels": diagnostics.evolution_period_labels,
            "evolutionPeriodStates": diagnostics.evolution_period_states,
            "evolutionBoundaryCauses": diagnostics.evolution_boundary_causes,
            "evolutionBoundaryYears": diagnostics.evolution_boundary_years,
            "evolutionBoundaryActors": diagnostics.evolution_boundary_actors,
            "evolutionBoundaryActions": diagnostics.evolution_boundary_actions,
            "evolutionBoundaryEvidenceIds": (
                diagnostics.evolution_boundary_evidence_ids
            ),
            "evolutionSubjectLifetime": diagnostics.evolution_subject_lifetime,
            "evolutionRangeMismatch": diagnostics.evolution_range_mismatch,
            "evolutionRangeResolution": diagnostics.evolution_range_resolution,
            "candidateCount": diagnostics.candidate_count,
            "selectedCount": diagnostics.selected_count,
            "selectedChunkIds": [item["id"] for item in results],
            "retrievalScore": round(confidence, 3),
            "retrievalQuality": round(retrieval_quality, 3),
            "evidenceCoverage": round(evidence_coverage, 3),
            "detailCoverage": diagnostics.detail_coverage,
            "concreteEvidenceCount": diagnostics.concrete_evidence_count,
            "concreteFacets": diagnostics.concrete_facets,
            "missingConcreteFacets": diagnostics.missing_concrete_facets,
            "coveredFacets": analysis.get("covered_facets") or [],
            "requiredFacets": list(plan.required_facets),
            "missingFacets": analysis.get("missing_facets") or [],
            "facetCoverage": analysis.get("facet_coverage") or {},
            "primaryEntity": normalized_signals.primary_entity,
            "exactMatchCount": diagnostics.exact_match_count,
            "retrievalChannels": diagnostics.channels,
            "retrievalStrategy": diagnostics.strategy,
            "confidenceFactors": diagnostics.confidence_factors,
            "directEvidence": bool(confidence_factors["direct_evidence"]),
            "entityMatch": bool(confidence_factors["entity_match"]),
            "latencyMs": round((time.perf_counter() - started_at) * 1000),
            "userHash": hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else "",
        })
        return ChatResponse(
            answer=answer.strip(),
            citations=citations[:6] if request.include_citations else [],
            confidence=round(confidence, 3),
            related_entities=sorted(entity for entity in related_entities if entity),
            retrieval=diagnostics,
        )
