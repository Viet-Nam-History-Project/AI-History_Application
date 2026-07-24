"""Shared historical facet taxonomy for indexing and retrieval planning."""

from __future__ import annotations

from dataclasses import dataclass

from .entity_resolution import normalize_search_key


@dataclass(frozen=True)
class HistoryFacet:
    facet_id: str
    label: str
    search_text: str
    evidence_terms: tuple[str, ...]
    detail_search_text: str = ""


HISTORY_FACETS = {
    "progress": HistoryFacet(
        "progress",
        "diễn biến và các đợt tiến công",
        "diễn biến các đợt tiến công chiến đấu",
        (
            "dien bien cua", "dien bien chien", "tien cong", "tan cong",
            "chien dau", "dot tien cong", "dot 1", "dot 2", "dot 3",
            "giai doan 1", "giai doan 2", "giai doan 3",
            "ke hoach tac chien",
        ),
        "diễn biến từng đợt từng giai đoạn mốc thời gian bắt đầu kết thúc "
        "địa điểm mục tiêu trận đánh và bước ngoặt cụ thể",
    ),
    "forces": HistoryFacet(
        "forces",
        "lực lượng tham chiến và tương quan lực lượng",
        "lực lượng tham chiến quân đội tương quan lực lượng",
        ("luc luong", "tham chien", "quan so", "binh luc", "su doan", "dai doan", "trung doan", "quan doi", "bo doi"),
        "lực lượng tham chiến tên đơn vị quân số vũ khí tương quan cụ thể",
    ),
    "leadership": HistoryFacet(
        "leadership",
        "người lãnh đạo và chỉ huy",
        "người lãnh đạo chỉ huy tư lệnh bộ chỉ huy",
        (
            "lanh dao", "chi huy", "tu lenh", "bo chi huy", "dai tuong",
            "tuong linh", "chi huy truong",
        ),
        "lãnh đạo chỉ huy tên nhân vật chức vụ quyết định vai trò "
        "chỉ huy đối phương bị bắt đầu hàng cụ thể",
    ),
    "result": HistoryFacet(
        "result",
        "kết quả",
        "kết quả thắng lợi thất bại",
        (
            "ket qua", "chien thang", "thang loi", "that bai", "dau hang",
            "tieu diet", "giai phong", "bat song", "bi bat", "ket thuc",
        ),
        "kết quả ngày kết thúc thắng lợi thất bại đầu hàng bắt sống "
        "chỉ huy số liệu thiệt hại giải phóng hiệp định ký kết "
        "chấm dứt chiến tranh công nhận độc lập chủ quyền",
    ),
    "significance": HistoryFacet(
        "significance",
        "ý nghĩa lịch sử",
        "ý nghĩa ảnh hưởng tác động bài học lịch sử",
        ("y nghia", "anh huong", "tac dong", "bai hoc kinh nghiem"),
        "tác động cụ thể thay đổi bước ngoặt bài học số liệu hiệp định "
        "hệ quả chính trị quốc tế phong trào giải phóng dân tộc",
    ),
    "cause": HistoryFacet(
        "cause",
        "nguyên nhân và bối cảnh",
        "nguyên nhân bối cảnh hoàn cảnh mục tiêu",
        ("nguyen nhan", "boi canh", "hoan canh", "am muu", "muc tieu"),
        "sự kiện mốc thời gian điều kiện nguyên nhân trực tiếp cụ thể",
    ),
    "political_administrative": HistoryFacet(
        "political_administrative",
        "chính trị và hành chính",
        "chính sách chính trị hành chính bộ máy cai trị thuộc địa",
        (
            "chinh tri", "hanh chinh", "cai tri", "bo may", "chinh quyen",
            "chia de tri", "bao ho", "quan lai", "dan ap",
            "nam ky", "bac ky", "trung ky",
        ),
        "ba kỳ nam kỳ bắc kỳ trung kỳ triều đình "
        "bộ máy quan lại sắc lệnh đàn áp phong trào",
    ),
    "economic_taxation": HistoryFacet(
        "economic_taxation",
        "kinh tế và thuế khóa",
        "chính sách kinh tế khai thác thuế khóa nông nghiệp công nghiệp thương nghiệp",
        (
            "kinh te", "khai thac", "thue", "thue khoa", "ruong dat",
            "don dien", "nong nghiep", "cong nghiep", "thuong nghiep",
            "khai mo", "khoang san", "doc quyen", "cao su", "xuat khau",
        ),
        "thuế thân thuế ruộng độc quyền muối rượu thuốc phiện "
        "đồn điền cao su cà phê khai mỏ than công ty số liệu",
    ),
    "culture_education": HistoryFacet(
        "culture_education",
        "văn hóa và giáo dục",
        "chính sách văn hóa giáo dục trường học báo chí ngôn luận",
        (
            "van hoa", "giao duc", "truong hoc", "chu quoc ngu", "bao chi",
            "ngon luan", "tuyen truyen", "thuoc phien", "ruou", "co bac",
        ),
        "trường học chương trình giáo dục đào tạo công chức "
        "báo chí kiểm duyệt tuyên truyền chữ quốc ngữ",
    ),
    "social_transformation": HistoryFacet(
        "social_transformation",
        "xã hội và sự phân hóa giai cấp",
        "biến đổi xã hội giai cấp tầng lớp đời sống nhân dân",
        (
            "xa hoi", "giai cap", "tang lop", "nong dan", "cong nhan",
            "tu san", "tieu tu san", "dia chu", "doi song", "mat dat",
        ),
        "giai cấp công nhân nông dân địa chủ tư sản "
        "tiểu tư sản trí thức đời sống việc làm",
    ),
    "objectives_consequences": HistoryFacet(
        "objectives_consequences",
        "mục đích và hậu quả",
        "mục đích hậu quả tác động của chính sách thuộc địa",
        (
            "muc dich", "hau qua", "tac dong", "anh huong", "phu thuoc",
            "thong tri", "boc lot", "vo vet", "mau thuan", "chu quyen",
            "chinh quoc", "loi ich", "nguyen lieu", "lao dong gia re",
        ),
        "mục đích lợi ích chính quốc nguyên liệu lao động thị trường; "
        "hệ quả cụ thể đời sống mất đất phụ thuộc mâu thuẫn phong trào đấu tranh",
    ),
    "historical_evolution": HistoryFacet(
        "historical_evolution",
        "sự thay đổi qua các thời kỳ",
        "sự thay đổi chính sách qua giai đoạn và thời kỳ lịch sử",
        (
            "thay doi", "chuyen sang", "qua cac giai doan", "qua cac thoi ky",
            "giai doan", "thoi ky", "lan thu nhat", "lan thu hai",
        ),
        "mốc năm nhân vật chính sách sự kiện chuyển biến cụ thể từng giai đoạn",
    ),
    "governance_methods": HistoryFacet(
        "governance_methods",
        "đàn áp, kiểm soát và nhượng bộ",
        "phương thức cai trị đàn áp kiểm soát nhượng bộ cải cách",
        (
            "dan ap", "khung bo", "bat bo", "tu day", "kiem soat",
            "kiem duyet", "bao chi", "canh sat", "tu phap", "nhuong bo",
            "dan chu", "cai cach", "mi dan", "phat xit hoa",
        ),
        "đàn áp bắt bớ kiểm duyệt báo chí cảnh sát tư pháp nhượng bộ "
        "dân chủ cải cách mị dân và mức độ kiểm soát cụ thể",
    ),
    "wartime_mobilization": HistoryFacet(
        "wartime_mobilization",
        "huy động thuộc địa trong thời chiến",
        "huy động nhân lực tài chính vật lực thời chiến",
        (
            "thoi chien", "chien tranh the gioi", "huy dong", "dong vien",
            "trung dung", "cong phieu", "quan dich", "phuc vu chien tranh",
            "phap nhat", "nhat dao chinh", "9 3 1945",
        ),
        "huy động nhân lực tài chính vật lực thuế công phiếu trưng dụng "
        "quan hệ Pháp Nhật và bước ngoặt thời chiến cụ thể",
    ),
    "comparison_context": HistoryFacet(
        "comparison_context",
        "thời gian và bối cảnh",
        "thời gian bối cảnh giai đoạn lần thứ nhất lần thứ hai",
        (
            "thoi gian", "boi canh", "lan thu nhat", "lan thu hai",
            "chien tranh the gioi", "giai doan", "bat dau", "ket thuc",
        ),
        "mốc bắt đầu kết thúc hoàn cảnh lịch sử cụ thể của từng đối tượng",
    ),
    "organizer": HistoryFacet(
        "organizer",
        "người khởi xướng và tổ chức",
        "người tổ chức khởi xướng điều hành chương trình kế hoạch",
        (
            "nguoi to chuc", "khoi xuong", "dieu hanh", "toan quyen",
            "chuong trinh", "ke hoach", "paul doumer", "albert sarraut",
        ),
        "tên người tổ chức chức vụ chương trình kế hoạch và vai trò cụ thể",
    ),
    "scale_investment": HistoryFacet(
        "scale_investment",
        "quy mô và vốn đầu tư",
        "quy mô vốn đầu tư tốc độ phạm vi khai thác",
        (
            "quy mo", "von dau tu", "tu ban dau tu", "nguon von",
            "tang von", "trieu francs", "ty francs", "pham vi dau tu",
        ),
        "quy mô tốc độ vốn đầu tư số liệu và phạm vi triển khai cụ thể",
    ),
    "agriculture": HistoryFacet(
        "agriculture",
        "nông nghiệp",
        "nông nghiệp ruộng đất đồn điền cây công nghiệp cao su",
        (
            "nong nghiep", "ruong dat", "don dien", "cao su", "ca phe",
            "cay lay dau", "thuy nong", "nhuong dia", "dien chu",
        ),
        "nông nghiệp ruộng đất đồn điền cao su cà phê cây trồng vốn và diện tích cụ thể",
    ),
    "industry": HistoryFacet(
        "industry",
        "công nghiệp và khai mỏ",
        "công nghiệp khai mỏ than khoáng sản chế biến nhà máy",
        (
            "cong nghiep", "khai mo", "ham mo", "than", "khoang san",
            "quang", "nha may", "xi mang", "det", "che bien",
        ),
        "ngành công nghiệp mỏ than khoáng sản nhà máy sản lượng cụ thể",
    ),
    "commerce": HistoryFacet(
        "commerce",
        "thương nghiệp và thị trường",
        "thương nghiệp ngoại thương thị trường công ty thương mại",
        (
            "thuong nghiep", "thuong mai", "ngoai thuong", "thi truong",
            "hang hoa", "xuat khau", "nhap khau", "cong ty thuong mai",
        ),
        "thương nghiệp thị trường ngoại thương công ty hàng hóa xuất nhập khẩu và độc quyền",
    ),
    "transport": HistoryFacet(
        "transport",
        "giao thông vận tải",
        "giao thông vận tải đường sắt đường bộ đường thủy cảng biển",
        (
            "giao thong", "van tai", "duong sat", "duong bo",
            "duong thuy", "cang bien", "xe lua", "tau bien",
        ),
        "giao thông vận tải tuyến đường sắt đường bộ đường thủy cảng phương tiện và số liệu vận chuyển",
    ),
    "geographic_scope": HistoryFacet(
        "geographic_scope",
        "phạm vi và địa bàn",
        "phạm vi địa bàn chiến trường miền Nam miền Bắc khu vực",
        (
            "pham vi", "dia ban", "chien truong", "mien nam", "mien bac",
            "mo rong chien tranh", "chien tranh pha hoai", "khu vuc",
        ),
        "phạm vi địa bàn chiến trường khu vực miền Nam miền Bắc và sự mở rộng cụ thể",
    ),
    "strategy_methods": HistoryFacet(
        "strategy_methods",
        "âm mưu và biện pháp chủ yếu",
        "âm mưu chiến lược biện pháp thủ đoạn phương thức tác chiến",
        (
            "am muu", "chien luoc", "bien phap", "thu doan",
            "ap chien luoc", "truc thang van", "thiet xa van",
            "tim diet", "binh dinh", "chien thuat",
        ),
        "âm mưu biện pháp thủ đoạn tác chiến ấp chiến lược tìm diệt bình định "
        "trực thăng vận thiết xa vận cụ thể",
    ),
    "scale_intensity": HistoryFacet(
        "scale_intensity",
        "quy mô và mức độ chiến tranh",
        "quy mô mức độ leo thang lực lượng phương tiện chiến tranh",
        (
            "quy mo", "muc do", "leo thang", "quan my", "quan dong minh",
            "khong quan", "hai quan", "vu khi", "phuong tien chien tranh",
        ),
        "quy mô mức độ leo thang quân số quân Mỹ quân đồng minh vũ khí "
        "không quân hải quân và phương tiện cụ thể",
    ),
    "representative_events": HistoryFacet(
        "representative_events",
        "sự kiện và thắng lợi tiêu biểu",
        "trận đánh chiến thắng thất bại sự kiện tiêu biểu",
        (
            "tran danh", "chien thang", "thang loi tieu bieu",
            "ap bac", "binh gia", "ba gia", "dong xoai", "van tuong",
            "mua kho", "mau than", "tong tien cong",
        ),
        "trận đánh chiến thắng sự kiện tiêu biểu tên địa điểm mốc thời gian "
        "và tác động trực tiếp đến chiến lược",
    ),
    "participants": HistoryFacet(
        "participants",
        "các bên và chủ thể tham gia",
        "các bên tham gia quốc gia tổ chức lực lượng đại biểu",
        (
            "cac ben tham gia", "phai doan", "quoc gia tham gia",
            "dai bieu", "hoi nghi", "dam phan", "ky ket",
        ),
        "tên quốc gia tổ chức lực lượng phái đoàn đại biểu và vai trò của từng bên",
    ),
    "agreement_content": HistoryFacet(
        "agreement_content",
        "nội dung và điều khoản chính",
        "nội dung điều khoản hiệp định ngừng bắn rút quân chủ quyền",
        (
            "dieu khoan", "noi dung hiep dinh", "ngung ban", "rut quan",
            "chu quyen", "tong tuyen cu", "gioi tuyen", "ky ket",
        ),
        "điều khoản ngừng bắn rút quân chủ quyền tổng tuyển cử giới tuyến và thời hạn cụ thể",
    ),
    "implementation_mechanism": HistoryFacet(
        "implementation_mechanism",
        "cơ chế và mức độ thực hiện",
        "cơ chế thực hiện giám sát trách nhiệm thời hạn thi hành",
        (
            "thi hanh", "thuc hien hiep dinh", "giam sat", "uy ban quoc te",
            "thoi han", "trach nhiem", "vi pham hiep dinh",
        ),
        "cơ chế giám sát trách nhiệm thời hạn thi hành và tình trạng thực hiện cụ thể",
    ),
    "limitations": HistoryFacet(
        "limitations",
        "hạn chế và điểm chưa giải quyết",
        "hạn chế điểm yếu bất lợi chưa thực hiện chưa giải quyết",
        (
            "han che", "diem yeu", "bat loi", "chua giai quyet",
            "khong thuc hien", "that bai", "khung hoang", "suy yeu",
        ),
        "hạn chế bất lợi điều chưa thực hiện nguyên nhân yếu kém và hệ quả cụ thể",
    ),
    "role_contribution": HistoryFacet(
        "role_contribution",
        "vai trò và đóng góp",
        "vai trò đóng góp quyết định chủ trương hoạt động thành tựu",
        (
            "vai tro", "dong gop", "cong lao", "quyet dinh",
            "chu truong", "lanh dao", "to chuc", "anh huong",
        ),
        "hành động quyết định chủ trương thành tựu vai trò chính trị quân sự ngoại giao cụ thể",
    ),
    "ideology_goals": HistoryFacet(
        "ideology_goals",
        "tư tưởng, mục tiêu và con đường",
        "tư tưởng mục tiêu con đường phương pháp hoạt động cứu nước",
        (
            "tu tuong", "muc tieu", "con duong", "phuong phap",
            "chu truong", "cuu nuoc", "duong loi", "nhan thuc",
        ),
        "tư tưởng chủ trương mục tiêu con đường phương pháp hành động và điểm khác biệt cụ thể",
    ),
    "organization_structure": HistoryFacet(
        "organization_structure",
        "tổ chức và cơ cấu quyền lực",
        "cơ cấu tổ chức bộ máy trung ương địa phương quyền lực pháp luật",
        (
            "co cau", "to chuc bo may", "trung uong", "dia phuong",
            "quyen luc", "phap luat", "tap quyen", "quan lai",
        ),
        "người đứng đầu cơ cấu trung ương địa phương phân quyền pháp luật và công cụ quản lý",
    ),
    "source_perspective": HistoryFacet(
        "source_perspective",
        "quan điểm và cách giải thích",
        "quan điểm tác giả luận điểm bằng chứng góc nhìn thiên kiến",
        (
            "quan diem", "tac gia", "luan diem", "goc nhin",
            "bang chung", "tu lieu", "nguon so cap", "nguon thu cap",
            "thien kien",
        ),
        "tác giả thời điểm loại nguồn luận điểm bằng chứng góc nhìn và giới hạn của tư liệu",
    ),
    "continuity_change": HistoryFacet(
        "continuity_change",
        "tính tiếp nối và thay đổi",
        "kế thừa tiếp nối thay đổi phát triển điểm mới bước ngoặt",
        (
            "ke thua", "tiep noi", "thay doi", "phat trien",
            "diem moi", "buoc ngoat", "chuyen bien", "lien tuc",
        ),
        "yếu tố được kế thừa yếu tố mới nguyên nhân chuyển biến và bước ngoặt cụ thể",
    ),
    "resources_logistics": HistoryFacet(
        "resources_logistics",
        "nguồn lực, hậu cần và phương tiện",
        "nguồn lực hậu cần tiếp tế vũ khí phương tiện tổ chức lực lượng",
        (
            "hau can", "tiep te", "vu khi", "phuong tien", "quan so",
            "van chuyen", "dan duoc", "luong thuc", "dong vien",
        ),
        "quân số vũ khí phương tiện hậu cần tiếp tế vận chuyển và khả năng huy động cụ thể",
    ),
}


POLICY_OVERVIEW_FACETS = (
    "political_administrative",
    "economic_taxation",
    "culture_education",
    "social_transformation",
    "objectives_consequences",
)


# Taxonomy ứng viên cho F9 (tiếp nối và thay đổi). Đây là danh mục để planner
# lựa chọn theo đối tượng, không phải checklist buộc câu trả lời dùng toàn bộ.
EVOLUTION_CORE_FACETS = (
    "historical_evolution",
    "continuity_change",
    "cause",
    "result",
    "significance",
)

EVOLUTION_DOMAIN_FACETS = {
    "political_administration": (
        "political_administrative",
        "organization_structure",
        "governance_methods",
        "economic_taxation",
        "wartime_mobilization",
        "social_transformation",
    ),
    "economic_policy": (
        "economic_taxation",
        "scale_investment",
        "agriculture",
        "industry",
        "commerce",
        "transport",
        "social_transformation",
    ),
    "military_strategy": (
        "forces",
        "geographic_scope",
        "strategy_methods",
        "scale_intensity",
        "representative_events",
        "result",
    ),
    "military_campaign": (
        "leadership",
        "forces",
        "geographic_scope",
        "resources_logistics",
        "progress",
        "representative_events",
        "result",
    ),
    "movement_revolution": (
        "objectives_consequences",
        "leadership",
        "forces",
        "geographic_scope",
        "strategy_methods",
        "progress",
        "result",
    ),
    "state_dynasty": (
        "organization_structure",
        "geographic_scope",
        "forces",
        "economic_taxation",
        "culture_education",
        "social_transformation",
        "result",
    ),
    # Dùng riêng khi câu hỏi F9 theo dõi một tiến trình rất rộng như
    # "Việt Nam thay đổi thế nào...". Planner chỉ chọn vài facet có bằng
    # chứng tốt để trả lời tổng quan, không biến danh sách này thành checklist.
    "multi_domain": (
        "political_administrative",
        "organization_structure",
        "strategy_methods",
        "geographic_scope",
        "economic_taxation",
        "social_transformation",
        "result",
        "significance",
    ),
    "historical_event": (
        "cause",
        "leadership",
        "forces",
        "progress",
        "result",
        "significance",
    ),
}


ECONOMIC_COMPARISON_FACETS = (
    "comparison_context",
    "organizer",
    "objectives_consequences",
    "scale_investment",
    "agriculture",
    "industry",
    "commerce",
    "transport",
    "social_transformation",
)

MILITARY_STRATEGY_COMPARISON_FACETS = (
    "comparison_context",
    "objectives_consequences",
    "forces",
    "geographic_scope",
    "strategy_methods",
    "scale_intensity",
    "representative_events",
    "result",
)

GENERIC_EVENT_COMPARISON_FACETS = (
    "comparison_context",
    "cause",
    "objectives_consequences",
    "forces",
    "leadership",
    "progress",
    "result",
    "significance",
)

MILITARY_CAMPAIGN_COMPARISON_FACETS = (
    "comparison_context",
    "objectives_consequences",
    "leadership",
    "forces",
    "geographic_scope",
    "resources_logistics",
    "progress",
    "representative_events",
    "result",
    "significance",
)

DIPLOMATIC_AGREEMENT_COMPARISON_FACETS = (
    "comparison_context",
    "participants",
    "objectives_consequences",
    "agreement_content",
    "implementation_mechanism",
    "result",
    "significance",
    "limitations",
)

HISTORICAL_PERSON_COMPARISON_FACETS = (
    "comparison_context",
    "ideology_goals",
    "strategy_methods",
    "role_contribution",
    "representative_events",
    "result",
    "limitations",
    "significance",
)

MOVEMENT_REVOLUTION_COMPARISON_FACETS = (
    "comparison_context",
    "cause",
    "objectives_consequences",
    "leadership",
    "forces",
    "geographic_scope",
    "strategy_methods",
    "progress",
    "result",
    "significance",
    "limitations",
)

POLITICAL_ADMINISTRATION_COMPARISON_FACETS = (
    "comparison_context",
    "organizer",
    "objectives_consequences",
    "organization_structure",
    "geographic_scope",
    "strategy_methods",
    "result",
    "social_transformation",
    "limitations",
)

STATE_DYNASTY_COMPARISON_FACETS = (
    "comparison_context",
    "organization_structure",
    "geographic_scope",
    "forces",
    "economic_taxation",
    "culture_education",
    "social_transformation",
    "result",
    "limitations",
    "significance",
)

SOURCE_INTERPRETATION_COMPARISON_FACETS = (
    "comparison_context",
    "source_perspective",
    "cause",
    "result",
    "significance",
    "limitations",
)

COMPARISON_DOMAIN_FACETS = {
    "economic_policy": ECONOMIC_COMPARISON_FACETS,
    "military_strategy": MILITARY_STRATEGY_COMPARISON_FACETS,
    "military_campaign": MILITARY_CAMPAIGN_COMPARISON_FACETS,
    "diplomatic_agreement": DIPLOMATIC_AGREEMENT_COMPARISON_FACETS,
    "historical_person": HISTORICAL_PERSON_COMPARISON_FACETS,
    "movement_revolution": MOVEMENT_REVOLUTION_COMPARISON_FACETS,
    "political_administration": POLITICAL_ADMINISTRATION_COMPARISON_FACETS,
    "state_dynasty": STATE_DYNASTY_COMPARISON_FACETS,
    "source_interpretation": SOURCE_INTERPRETATION_COMPARISON_FACETS,
    "historical_event": GENERIC_EVENT_COMPARISON_FACETS,
}

# Toàn bộ tiêu chí mà comparison planner được phép chọn. Danh sách này là
# taxonomy, không phải checklist bắt buộc cho mọi câu hỏi.
COMPARISON_FACETS = tuple(dict.fromkeys((
    *(
        facet_id
        for domain_facets in COMPARISON_DOMAIN_FACETS.values()
        for facet_id in domain_facets
    ),
)))


def comparison_domain_for_question(text: str) -> str:
    """Infer a conservative domain; the model planner may refine it later."""
    normalized = normalize_search_key(text)
    domain_markers = (
        ("source_interpretation", (
            "hai tu lieu", "hai tai lieu", "hai nguon", "quan diem",
            "cach danh gia", "cach giai thich",
        )),
        ("diplomatic_agreement", (
            "hiep dinh", "hoi nghi", "dam phan", "tuyen ngon",
        )),
        ("economic_policy", (
            "khai thac thuoc dia", "kinh te thuoc dia",
            "chinh sach kinh te", "cuoc khai thac",
        )),
        ("military_strategy", (
            "chien luoc", "chien tranh dac biet", "chien tranh cuc bo",
            "chien tranh don phuong", "viet nam hoa chien tranh",
        )),
        ("military_campaign", (
            "chien dich", "tran danh", "tran chien",
        )),
        ("movement_revolution", (
            "phong trao", "khoi nghia", "cach mang", "dong khoi",
            "tong tien cong",
        )),
        ("political_administration", (
            "bo may nha nuoc", "bo may cai tri", "cai cach hanh chinh",
            "chinh sach cai tri",
        )),
        ("state_dynasty", (
            "trieu dai", "nha nuoc van lang", "au lac", "nha le",
            "nha mac", "nha nguyen",
        )),
        ("historical_person", (
            "vai tro cua", "chu truong cuu nuoc", "hai nhan vat",
        )),
    )
    for domain, markers in domain_markers:
        if any(marker in normalized for marker in markers):
            return domain
    return "historical_event"


def comparison_facets_for_domain(domain: str) -> tuple[str, ...]:
    return COMPARISON_DOMAIN_FACETS.get(
        domain,
        GENERIC_EVENT_COMPARISON_FACETS,
    )


def comparison_facets_for_question(text: str) -> tuple[str, ...]:
    """Return a domain-aware candidate set, not a universal checklist."""
    return comparison_facets_for_domain(comparison_domain_for_question(text))


def evolution_facets_for_domain(domain: str) -> tuple[str, ...]:
    """Return F9 structural facets plus domain-relevant candidates."""
    return tuple(dict.fromkeys((
        *EVOLUTION_CORE_FACETS,
        *EVOLUTION_DOMAIN_FACETS.get(
            domain,
            EVOLUTION_DOMAIN_FACETS["historical_event"],
        ),
    )))


def facet_label(facet_id: str) -> str:
    facet = HISTORY_FACETS.get(facet_id)
    return facet.label if facet else facet_id


def facet_search_text(facet_id: str) -> str:
    facet = HISTORY_FACETS.get(facet_id)
    return facet.search_text if facet else facet_id.replace("_", " ")


def facet_detail_search_text(facet_id: str) -> str:
    """Return a focused query that favors concrete, teachable evidence."""
    facet = HISTORY_FACETS.get(facet_id)
    if facet is None:
        return "ví dụ cụ thể tên riêng mốc thời gian số liệu"
    return facet.detail_search_text or (
        f"{facet.search_text} ví dụ cụ thể tên riêng mốc thời gian số liệu"
    )


def infer_history_facets(text: str) -> tuple[str, ...]:
    """Infer deterministic facets from chunk text without an AI call."""
    normalized = normalize_search_key(text)
    return tuple(
        facet_id
        for facet_id, facet in HISTORY_FACETS.items()
        if any(term in normalized for term in facet.evidence_terms)
    )


def text_supports_facet(
    facet_id: str,
    text: str,
    indexed_facets: tuple[str, ...] | list[str] = (),
) -> bool:
    if facet_id in indexed_facets:
        return True
    facet = HISTORY_FACETS.get(facet_id)
    if facet is None:
        return False
    normalized = normalize_search_key(text)
    return any(term in normalized for term in facet.evidence_terms)
