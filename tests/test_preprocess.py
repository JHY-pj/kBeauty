"""인터넷 연결 없이 가상 데이터로 실행하는 테스트입니다. GitHub 원본 전체 검증은 아닙니다."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess as p


class PreprocessTests(unittest.TestCase):
    def test_broken_byte_repair(self):
        text, repaired, ambiguous = p.restore_latin1("I can?\x99t use this")
        self.assertEqual(text, "I can’t use this")
        self.assertEqual((repaired, ambiguous), (1, 0))
        txt, repaired, amb = p.restore_latin1("My ?\xa4 skin")
        self.assertGreater(amb, 0)
        self.assertEqual(repaired, 0)

    def test_volume_conservative(self):
        self.assertEqual(p.volume_in_ml("1.0 fl oz / 30 mL"), 30)
        self.assertAlmostEqual(p.volume_in_ml("1 fl oz"), 29.5735)
        self.assertIsNone(p.volume_in_ml("30 g"))
        self.assertIsNone(p.volume_in_ml("30 mL + 20 mL"))
        self.assertIsNone(p.volume_in_ml("Hydrating Kit 50 mL"))

    def test_full_pipeline(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, out = root / "data", root / "outputs"
            data.mkdir()
            with (data / "product_info.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["product_id", "product_name", "price_usd", "size", "rating", "reviews", "primary_category", "secondary_category"])
                writer.writeheader()
                writer.writerow({"product_id": "P1", "product_name": "Serum One", "price_usd": "30", "size": "1 fl oz / 30 mL", "rating": "4.5", "reviews": "90", "primary_category": "Skincare", "secondary_category": "Serums"})
                writer.writerow({"product_id": "P2", "product_name": "Serum Two", "price_usd": "70", "size": "50 g", "rating": "4", "reviews": "51", "primary_category": "Skincare", "secondary_category": "Serums"})
            rows = [
                ("a", 2019, "Works on my face", "P1", 1),
                ("a", 2022, "Works even better now", "P1", 1),  # 앞 기간에 등장한 작성자: 이후 기간 모델 입력에서 제외
                ("b", 2022, "Nice skin improvement", "P1", 1),
                ("c", 2023, "Nice skin improvement", "P2", 1),  # 앞 기간과 완전히 같은 문장: 이후 기간 모델 입력에서 제외
                ("d", 2023, "My skin can?\x99t tolerate perfume", "P2", 0),
                ("e", 2023, "Unrelated fresh honest text", "P2", 0),
                ("b", 2022, "Nice skin improvement", "P1", 1),  # 같은 작성자·본문 중복
                ("f", 2023, "Great product but bad ID", "INVALID", 1),
            ]
            review_path = data / "reviews_500-750.csv"
            # 이름이 있는 리뷰 데이터 열 18개와 첫 번째 이름 없는 인덱스 열
            with review_path.open("w", newline="", encoding="latin-1") as f:
                writer = csv.writer(f)
                writer.writerow(["Unnamed: 0", *p.REVIEW_FIELDS])
                for idx, (author, year, review_text, pid, label) in enumerate(rows):
                    review = {k: "" for k in p.REVIEW_FIELDS}
                    review.update({"author_id": author, "rating": "4", "is_recommended": str(label),
                                   "submission_time": f"{year}-02-01", "review_text": review_text,
                                   "product_id": pid, "product_name": "Test product", "brand_name": "Test brand",
                                   "price_usd": "30.0"})
                    writer.writerow([idx, *[review[k] for k in p.REVIEW_FIELDS]])
            rc = p.main(["--data-dir", str(data), "--out-dir", str(out), "--min-review-chars", "1"])
            self.assertEqual(rc, 0)
            report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
            s = report["stats"]
            self.assertEqual(s["product_rows_valid"], 2)
            self.assertEqual(s["review_rows_valid"], 7)
            self.assertEqual(s["review_rows_quarantined"], 1)
            self.assertEqual(s["cross_period_exclusions"], 2)
            self.assertEqual(s["m4_model_input_rows"], 4)
            self.assertEqual(s["m5_model_input_rows"], 4)
            self.assertGreater(s["review_text_repair_pairs"], 0)
            with (out / "reviews_m4_input.csv").open(encoding="utf-8") as f:
                m4 = list(csv.DictReader(f))
            self.assertEqual(Counter(r["split"] for r in m4), Counter(train=1, validation=1, test=2))
            self.assertTrue(any("can’t" in r["review_text"] for r in m4))
            with (out / "products_clean.csv").open(encoding="utf-8") as f:
                products = list(csv.DictReader(f))
            self.assertEqual(products[0]["volume_ml"], "30.0")
            self.assertEqual(products[0]["unit_price_usd_per_ml"], "1.0")
            self.assertEqual(products[1]["volume_ml"], "")

    def test_corrupt_title_does_not_drop_clean_review(self):
        # 회귀 테스트: v1.0은 제목만 손상돼도 정상 리뷰 본문을 학습 후보에서 제외했습니다.
        with TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"; out = root / "output"; data.mkdir()
            with (data / "product_info.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["product_id", "product_name", "size", "price_usd"])
                w.writeheader(); w.writerow({"product_id":"P1","product_name":"Serum","size":"30 mL","price_usd":"50"})
            with (data / "reviews_1250-end.csv").open("w", encoding="latin-1", newline="") as f:
                w = csv.writer(f); w.writerow(["Unnamed: 0", *p.REVIEW_FIELDS])
                for i, (author, body, title) in enumerate([
                    ("author_a", "Completely readable review body", "Bad title ?\xa4"),
                    ("author_b", "Broken review body ?\xa4 here", "Clean title")
                ]):
                    row = {k:"" for k in p.REVIEW_FIELDS}
                    row.update({"author_id":author,"rating":"4","is_recommended":"1",
                                "submission_time":"2023-02-01", "review_text":body,
                                "review_title":title,"product_id":"P1"})
                    w.writerow([i,*[row[k] for k in p.REVIEW_FIELDS]])
            self.assertEqual(p.main(["--data-dir",str(data),"--out-dir",str(out)]), 0)
            with (out / "reviews_m4_input.csv").open(encoding="utf-8",newline="") as f:
                model_rows = list(csv.DictReader(f))
            with (out / "reviews_clean.csv").open(encoding="utf-8",newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(model_rows), 1)
            self.assertEqual(model_rows[0]["review_text"], "Completely readable review body")
            self.assertEqual(rows[0]["text_ambiguous"], "0")
            self.assertGreater(int(rows[0]["metadata_ambiguous"]), 0)
            self.assertGreater(int(rows[1]["text_ambiguous"]), 0)
            report = json.loads((out / "quality_report.json").read_text())
            self.assertEqual(report["stats"]["unresolved_text_excluded_from_model_inputs"], 1)
            self.assertEqual(report["stats"]["review_metadata_ambiguous_chars"], 2)


    def test_identical_text_different_authors_same_period(self):
        """같은 기간에 서로 다른 작성자가 동일 문장을 올려도 모델 입력에는 한 번만 들어갑니다."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, out = root / "data", root / "out"
            data.mkdir()
            with (data / "product_info.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["product_id", "price_usd", "size"])
                w.writeheader()
                w.writerow({"product_id": "P1", "price_usd": "20", "size": "30 mL"})
            with (data / "reviews_1250-end.csv").open("w", encoding="latin-1", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Unnamed: 0", *p.REVIEW_FIELDS])
                examples = [
                    ("reviewer_a", "The same very detailed review text", ""),
                    ("reviewer_b", "The same very detailed review text", "1"),
                    ("reviewer_c", "Another separately written review", "0"),
                ]
                for idx, (author, body, label) in enumerate(examples):
                    row = {name: "" for name in p.REVIEW_FIELDS}
                    row.update({"author_id": author, "rating": "4",
                                "is_recommended": label, "submission_time": "2022-06-01",
                                "review_text": body, "product_id": "P1"})
                    w.writerow([idx, *[row[name] for name in p.REVIEW_FIELDS]])
            self.assertEqual(p.main(["--data-dir", str(data), "--out-dir", str(out)]), 0)
            with (out / "reviews_clean.csv").open(encoding="utf-8", newline="") as f:
                clean = list(csv.DictReader(f))
            with (out / "reviews_m4_input.csv").open(encoding="utf-8", newline="") as f:
                m4 = list(csv.DictReader(f))
            with (out / "reviews_m5_input.csv").open(encoding="utf-8", newline="") as f:
                m5 = list(csv.DictReader(f))
            self.assertEqual(len(clean), 3)
            self.assertEqual(len(m4), 2)
            self.assertEqual(len(m5), 2)  # 첫 문장 중 라벨이 있는 작성자 B는 포함
            self.assertEqual(clean[1]["is_duplicate_text"], "1")
            self.assertEqual(clean[1]["is_repeated_author_text"], "0")
            self.assertEqual({row["review_text"] for row in m4},
                             {"The same very detailed review text", "Another separately written review"})
            report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["stats"]["m4_duplicate_text_excluded"], 1)
            self.assertEqual(report["stats"].get("m5_duplicate_text_excluded", 0), 0)


from collections import Counter
if __name__ == "__main__":
    unittest.main()
