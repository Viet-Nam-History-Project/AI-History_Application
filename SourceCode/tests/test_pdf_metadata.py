import unittest

from src.api.pdf_ingestion import (
    detect_content_page_range,
    extract_heading,
    infer_chunk_metadata,
    strip_page_footnotes,
    strip_page_running_header,
)


class PdfChunkMetadataTest(unittest.TestCase):
    def test_infers_facets_and_year_range(self) -> None:
        word_count, facets, year_start, year_end = infer_chunk_metadata(
            "Năm 1954, lực lượng tham chiến mở ba đợt tiến công. "
            "Kết quả là tập đoàn cứ điểm phải đầu hàng vào năm 1954."
        )

        self.assertGreater(word_count, 10)
        self.assertIn("forces", facets)
        self.assertIn("progress", facets)
        self.assertIn("result", facets)
        self.assertEqual(year_start, 1954)
        self.assertEqual(year_end, 1954)

    def test_infers_policy_facets_without_ai(self) -> None:
        _, facets, _, _ = infer_chunk_metadata(
            "Chính quyền thuộc địa tổ chức bộ máy cai trị và đặt nhiều loại thuế. "
            "Trong giáo dục, trường học phục vụ chủ yếu cho bộ máy hành chính. "
            "Quá trình khai thác làm xã hội phân hóa thành nhiều giai cấp."
        )

        self.assertIn("political_administrative", facets)
        self.assertIn("economic_taxation", facets)
        self.assertIn("culture_education", facets)
        self.assertIn("social_transformation", facets)

    def test_heading_is_conservative(self) -> None:
        self.assertEqual(
            extract_heading(
                "CHƯƠNG III\nCHIẾN DỊCH ĐIỆN BIÊN PHỦ\nNội dung chương."
            ),
            "CHƯƠNG III",
        )
        self.assertEqual(
            extract_heading(
                "động, hối hả như những ngày tháng ba.\nĐoạn tiếp theo."
            ),
            "",
        )

    def test_detects_history_body_and_excludes_references(self) -> None:
        result = detect_content_page_range([
            "BÌA SÁCH\nTác giả Trần Đức Cường",
            "NHÓM BIÊN SOẠN\nNguyễn Hữu Đạo: Chương I",
            "LỜI NÓI ĐẦU\nNội dung giới thiệu cuốn sách.",
            "Chuong I\nMIỀN BẮC TRONG THỜI KỲ KHÔI PHỤC\nNội dung lịch sử.",
            "Chương I. Miền Bắc\nNội dung tiếp theo.",
            "KẾT LUẬN\nTổng kết nội dung lịch sử.",
            "TÀI LIỆU THAM KHẢO\n1. Tên sách, Nhà xuất bản.",
            "Tài liệu tham khảo\n2. Tên sách khác.",
        ])

        self.assertEqual(result.start_page, 4)
        self.assertEqual(result.end_page, 6)
        self.assertEqual(result.detection, "chapter_1_to_references")

    def test_does_not_crop_when_markers_are_only_inline(self) -> None:
        result = detect_content_page_range([
            "NHÓM BIÊN SOẠN\nNguyễn Hữu Đạo phụ trách Chương I.",
            "LỜI NÓI ĐẦU\nTài liệu tham khảo được liệt kê ở cuối sách.",
            "Nội dung lịch sử không chia chương.",
        ])

        self.assertEqual(result.start_page, 1)
        self.assertEqual(result.end_page, 3)
        self.assertEqual(result.detection, "full_document")

    def test_removes_numbered_citations_at_page_bottom(self) -> None:
        body_lines = [
            "CHƯƠNG I. TỪNG BƯỚC KHẮC PHỤC KHỦNG HOẢNG",
            *[
                f"Nội dung lịch sử chính của trang, dòng số {index}."
                for index in range(1, 14)
            ],
        ]
        text = "\n".join([
            *body_lines,
            "1 . Tổng cục Thống kê, Số liệu thống kê Việt Nam thế kỷ XX, "
            "quyển 2, Sđd, tr. 126.",
            "2. Tổng cục Thống kê, Số liệu thống kê Việt Nam thế kỷ XX, "
            "quyển 2, Sđd, tr. 128.",
            "3. Tổng cục Thống kê, Số liệu thống kê Việt Nam thế kỷ XX, "
            "quyển 2, Sđd, tr. 753.",
        ])

        stripped, removed_chars = strip_page_footnotes(text)

        self.assertEqual(stripped, "\n".join(body_lines))
        self.assertGreater(removed_chars, 100)
        self.assertNotIn("Tổng cục Thống kê", stripped)

    def test_keeps_numbered_historical_list_in_body(self) -> None:
        text = "\n".join([
            "CHƯƠNG II",
            "Nội dung dẫn nhập về các nhiệm vụ lịch sử.",
            "1. Khôi phục sản xuất nông nghiệp.",
            "2. Phát triển công nghiệp.",
            "3. Củng cố quốc phòng.",
            "Phần phân tích tiếp tục sau danh sách nhiệm vụ.",
            "Kết quả của kế hoạch được trình bày trong đoạn cuối.",
            "Nội dung kết luận của trang.",
        ])

        stripped, removed_chars = strip_page_footnotes(text)

        self.assertEqual(stripped, text)
        self.assertEqual(removed_chars, 0)

    def test_removes_ocr_running_chapter_header(self) -> None:
        body_lines = [
            "Trong văn học nghệ thuật, Nhà nước đã chú ý đến đời sống.",
            *[
                f"Nội dung lịch sử chính của trang, dòng số {index}."
                for index in range(1, 12)
            ],
        ]
        text = "\n".join([
            "Chưcmg III. Miền Bắc xây dựng chủ nghĩa xã hội..,",
            "------------------------------",
            *body_lines,
        ])

        stripped, removed_chars = strip_page_running_header(text)

        self.assertEqual(stripped, "\n".join(body_lines))
        self.assertGreater(removed_chars, 40)
        self.assertNotIn("Chưcmg III", stripped)

    def test_keeps_real_chapter_opening_heading(self) -> None:
        text = "\n".join([
            "Chương I",
            "MIỀN BẮC TRONG THỜI KỲ KHÔI PHỤC",
            *[
                f"Nội dung mở đầu chương, dòng số {index}."
                for index in range(1, 12)
            ],
        ])

        stripped, removed_chars = strip_page_running_header(text)

        self.assertEqual(stripped, text)
        self.assertEqual(removed_chars, 0)


if __name__ == "__main__":
    unittest.main()
