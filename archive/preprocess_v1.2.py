#!/usr/bin/env python3
"""Kaggle Sephora 상품·리뷰 CSV 전처리 도구 (모델링 정의서 v1.0, M0~M5 대상).

실행 예시: python preprocess.py --data-dir ./data --out-dir ./outputs/preprocessed
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime
from hashlib import sha1, sha256
import json
import math
import re
from pathlib import Path
import sys
from typing import Any

# 리뷰 파일 구조: 첫 번째 이름 없는 인덱스 열 + 이름이 있는 데이터 열 18개 (03번 품질 보고서 기준)
REVIEW_FIELDS = [
    "author_id", "rating", "is_recommended", "helpfulness",
    "total_feedback_count", "total_neg_feedback_count", "total_pos_feedback_count",
    "submission_time", "review_text", "review_title", "skin_tone", "eye_color",
    "skin_type", "hair_color", "product_id", "product_name", "brand_name", "price_usd",
]
REQUIRED_REVIEW_FIELDS = {"author_id", "rating", "submission_time", "review_text", "product_id"}
PRODUCT_ID = re.compile(r"^P\d+$", re.I)
# 품질 보고서의 손상 패턴: '?' 다음에 스마트 문장부호의 잘못 남은 바이트가 이어짐.
BROKEN_PUNCTUATION = {
    0x99: "’",  # 굽은 작은따옴표
    0x9C: "“",  # 여는 큰따옴표
    0x9D: "”",  # 닫는 큰따옴표
    0x94: "—",  # 긴 대시
    0x93: "–",  # 짧은 대시
    0xA6: "…",  # 말줄임표
}
BROKEN_PAIR = re.compile(rb"\?([\x80-\xbf])")
SIZE_ML = re.compile(r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)\s*(?:m[lL]|millilit(?:er|re)s?)\b", re.I)
SIZE_FLOZ = re.compile(r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)\s*(?:fl\.?\s*oz\.?|fluid\s*ounces?)\b", re.I)
KIT_PATTERN = re.compile(r"\b(?:kit|set|bundle|duo|trio|mini\s+set|value\s+set)\b", re.I)
# 검증 설계에 따른 시기별 데이터 분할: 2021년까지 / 2022년 / 2023년 이후
SPLIT_ORD = {"train": 0, "validation": 1, "test": 2}
STAGING_COLS = ["review_id", "source_file", "source_row", "source_index", *REVIEW_FIELDS,
                "text_sha", "author_text_sha", "text_repaired", "text_ambiguous",
                "metadata_repaired", "metadata_ambiguous"]
CLEAN_COLS = STAGING_COLS + ["raw_split", "split", "cross_split_exclusion_reason",
                              "is_repeated_author_text", "is_duplicate_text"]
MODEL_REVIEW_COLS = ["review_id", "product_id", "submission_time", "review_text", "rating",
                     "skin_type", "skin_tone", "split"]
M5_COLS = ["review_id", "product_id", "submission_time", "review_text", "is_recommended", "split"]


def setup_csv_limit() -> None:
    limit = sys.maxsize
    while limit:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10


def normalize_header(value: str) -> str:
    value = value.replace("\ufeff", "").strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    return value


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "null", "none") else s


def restore_latin1(value: str) -> tuple[str, int, int]:
    """latin-1 방식으로 보존한 바이트를 유니코드로 변환하고 알려진 손상 패턴 복원.

    복원 결과는 추정값이며 원문과 일치한다는 보증이 아님.
    해석할 수 없는 바이트와 잘못된 UTF-8 문자는 별도 집계.
    """
    raw = value.encode("latin-1")
    repaired = 0
    ambiguous = 0

    def replace(match: re.Match[bytes]) -> bytes:
        nonlocal repaired, ambiguous
        cp = match.group(1)[0]
        mapped = BROKEN_PUNCTUATION.get(cp)
        if mapped is None:
            ambiguous += 1
            return b"?" + bytes([cp])
        repaired += 1
        return mapped.encode("utf-8")

    restored = BROKEN_PAIR.sub(replace, raw)
    # 복원할 수 없는 바이트는 대체 문자(U+FFFD)로 남김.
    unicode_text = restored.decode("utf-8", errors="replace")
    ambiguous += unicode_text.count("\ufffd")
    return unicode_text.strip(), repaired, ambiguous


def as_float(value: str) -> float | None:
    s = text_or_empty(value).replace(",", "")
    if not s:
        return None
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def as_int(value: str) -> int | None:
    f = as_float(value)
    return int(f) if f is not None and f.is_integer() else None


def as_boolean(value: str) -> str:
    s = text_or_empty(value).lower()
    if not s:
        return ""
    if s in ("1", "1.0", "true", "yes"):
        return "1"
    if s in ("0", "0.0", "false", "no"):
        return "0"
    return ""  # 잘못된 값을 임의 변환하지 않으며, 품질 통계에서 별도 집계.


def as_date(value: str) -> date | None:
    s = text_or_empty(value)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def category_key(row: dict[str, Any]) -> str:
    return " > ".join(filter(None, (
        text_or_empty(row.get("primary_category", "")),
        text_or_empty(row.get("secondary_category", "")),
        text_or_empty(row.get("tertiary_category", "")),
    )))


def volume_in_ml(size: str) -> float | None:
    """용량을 보수적으로 환산. 명시된 mL를 우선하고 없으면 fl oz를 사용.

    g를 mL와 동일시하지 않음. 서로 다른 용량이 여러 개이거나 세트·키트 제품이면
    비교 가능한 단일 용량이 아니므로 결측으로 남겨 추후 검토 예정.
    """
    s = text_or_empty(size)
    if not s or KIT_PATTERN.search(s) or "+" in s or " x " in s.lower():
        return None
    values_ml = [float(x) for x in SIZE_ML.findall(s)]
    if len(set(values_ml)) > 1:
        return None
    if values_ml:
        return values_ml[0] if values_ml[0] > 0 else None
    values_oz = [float(x) for x in SIZE_FLOZ.findall(s)]
    if len(set(values_oz)) > 1:
        return None
    return round(values_oz[0] * 29.5735, 6) if values_oz and values_oz[0] > 0 else None


def file_order(file: Path) -> tuple[int, int]:
    name = file.name
    order = {"reviews_0-250": 0, "reviews_250-500": 1,
             "reviews_500-750": 2, "reviews_750-1250": 3,
             "reviews_1250-end": 4}
    for prefix, part in order.items():
        if name.startswith(prefix):
            m = re.search(r"\((\d+)\)\.csv$", name)
            return part, int(m.group(1)) if m else 0
    return (99, 0)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_products(path: Path, out: Path, bad_out: Path, stats: Counter) -> set[str]:
    """M0~M3 모델링에 필요한 원래 상품 정보 열을 삭제하지 않고 모두 보존."""
    try:
        path.read_bytes()[:65536].decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        encoding = "latin-1"
    stats["product_encoding_fallback_latin1"] = 1 if encoding == "latin-1" else 0
    product_ids: set[str] = set()
    with path.open("r", encoding=encoding, newline="") as source, \
         out.open("w", encoding="utf-8", newline="") as dest, \
         bad_out.open("w", encoding="utf-8", newline="") as rejected:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("product_info.csv 헤더가 없습니다")
        renamed = {v: normalize_header(v) for v in reader.fieldnames if v is not None}
        if "product_id" not in renamed.values():
            raise ValueError("product_info.csv 에 product_id가 없습니다")
        cols = list(renamed.values()) + ["category_path", "volume_ml", "unit_price_usd_per_ml",
                                         "volume_conversion_status", "product_price_valid"]
        writer = csv.DictWriter(dest, fieldnames=cols, extrasaction="ignore")
        invalid = csv.DictWriter(rejected, fieldnames=["row_number", "reason", "raw_json"])
        writer.writeheader()
        invalid.writeheader()
        for n, source_row in enumerate(reader, start=2):
            stats["product_rows_read"] += 1
            row = {renamed[k]: (v if v is not None else "") for k, v in source_row.items() if k in renamed}
            if encoding == "latin-1":
                row = {k: restore_latin1(v)[0] for k, v in row.items()}
            pid = text_or_empty(row.get("product_id"))
            if not PRODUCT_ID.fullmatch(pid):
                stats["product_invalid_id"] += 1
                invalid.writerow({"row_number": n, "reason": "invalid_product_id", "raw_json": json.dumps(row, ensure_ascii=False)})
                continue
            if pid in product_ids:
                stats["product_duplicate_id"] += 1
                invalid.writerow({"row_number": n, "reason": "duplicate_product_id", "raw_json": json.dumps(row, ensure_ascii=False)})
                continue
            product_ids.add(pid)
            raw_price = row.get("price_usd", "")
            price = as_float(raw_price)
            row["price_usd"] = price if price is not None else ""
            row["product_price_valid"] = int(price is not None and price > 0)
            if not row["product_price_valid"]:
                stats["product_missing_or_invalid_price"] += 1
            for numeric in ("rating", "reviews", "loves_count", "sale_price_usd", "value_price_usd"):
                if numeric in row:
                    val = as_float(row[numeric])
                    row[numeric] = "" if val is None else val
            volume = volume_in_ml(row.get("size", ""))
            row["category_path"] = category_key(row)
            row["volume_ml"] = volume if volume is not None else ""
            row["unit_price_usd_per_ml"] = round(price / volume, 8) if volume and price and price > 0 else ""
            row["volume_conversion_status"] = "converted" if volume is not None else "not_comparable_or_missing"
            if volume is None:
                stats["product_volume_not_ml_comparable"] += 1
            writer.writerow(row)
            stats["product_rows_valid"] += 1
    return product_ids


def stage_reviews(files: list[Path], staging: Path, quarantine: Path, product_ids: set[str],
                  stats: Counter) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    first_author: dict[str, int] = {}
    first_text: dict[str, int] = {}
    file_stats: dict[str, Any] = {}
    with staging.open("w", encoding="utf-8", newline="") as valid_out, \
         quarantine.open("w", encoding="utf-8", newline="") as bad_out:
        writer = csv.DictWriter(valid_out, fieldnames=STAGING_COLS)
        writer.writeheader()
        reject = csv.DictWriter(bad_out, fieldnames=["source_file", "source_row", "source_line", "source_index", "reason", "raw_json"])
        reject.writeheader()
        for file in files:
            per = Counter()
            with file.open("r", encoding="latin-1", newline="") as f:
                rdr = csv.reader(f, strict=False)
                try:
                    headers = next(rdr)
                except StopIteration:
                    raise ValueError(f"빈 리뷰 파일: {file.name}")
                headers = [normalize_header(x) for x in headers]
                index_col = 0 if headers and (not headers[0] or headers[0].startswith("unnamed") or headers[0] in {"index", ""}) else None
                core_headers = [h for i, h in enumerate(headers) if i != index_col]
                core_columns = [h for h in core_headers if h]
                missing = REQUIRED_REVIEW_FIELDS - set(core_columns)
                if missing:
                    raise ValueError(f"{file.name}: 필수 리뷰 컬럼 없음: {sorted(missing)} / {core_headers[:22]}")
                meaningful_indices = [i for i,h in enumerate(headers) if h in REVIEW_FIELDS]
                # 열의 고정 위치를 임의로 가정하지 않고 실제 헤더 이름을 기준으로 처리.
                for source_row, values in enumerate(rdr, start=2):
                    per["rows_parsed"] += 1
                    stats["review_rows_parsed"] += 1
                    ref_index = values[index_col] if index_col is not None and len(values) > index_col else ""
                    reasons: list[str] = []
                    # 맨 끝의 이름 없는 빈 열은 빠져도 허용하지만 필수 데이터 열이 없으면 오류로 격리.
                    required_width = max(i for i, name in enumerate(headers) if name in REVIEW_FIELDS) + 1
                    if len(values) < required_width:
                        reasons.append("missing_core_columns")
                    if len(values) < len(headers):
                        values = values + [""] * (len(headers) - len(values))
                    # 헤더 범위를 벗어나 값이 존재하면 CSV의 필드 밀림을 의심.
                    if len(values) > len(headers) and any(text_or_empty(x) for x in values[len(headers):]):
                        reasons.append("extra_nonempty_columns")
                    row_dict = {}
                    for idx, h in enumerate(headers):
                        if h in REVIEW_FIELDS:
                            row_dict[h] = values[idx] if idx < len(values) else ""
                        elif idx != index_col and idx < len(values) and text_or_empty(values[idx]):
                            reasons.append("unexpected_named_or_unnamed_column")
                    # CSV 행이 깨지면 product_id 자리에 제품명이 들어가는 경우가 있어 형식을 검증.
                    if not PRODUCT_ID.fullmatch(text_or_empty(row_dict.get("product_id", ""))):
                        reasons.append("invalid_product_id")
                    score = as_float(row_dict.get("rating", ""))
                    if score is None or not (1 <= score <= 5) or not score.is_integer():
                        reasons.append("invalid_rating")
                    day = as_date(row_dict.get("submission_time", ""))
                    if day is None or day > date.today() or day < date(2000, 1, 1):
                        reasons.append("invalid_date")
                    if not reasons and row_dict["product_id"] not in product_ids:
                        reasons.append("unmatched_product_id")
                    if reasons:
                        per["quarantined"] += 1
                        stats["review_rows_quarantined"] += 1
                        for reason in set(reasons):
                            stats["reject_" + reason] += 1
                        reject.writerow({
                            "source_file": file.name, "source_row": source_row, "source_line": rdr.line_num,
                            "source_index": ref_index, "reason": ";".join(sorted(set(reasons))),
                            "raw_json": json.dumps(values, ensure_ascii=False),
                        })
                        continue
                    normalized: dict[str, str] = {}
                    # M4/M5의 입력으로 실제 사용하는 리뷰 본문(review_text)의 손상만 모델 입력 제외 기준으로 삼음
                    # 제목·브랜드·피부 특성이 손상됐다는 이유만으로 정상 리뷰 본문까지 제외하지 않음.
                    text_repaired = text_ambiguous = 0
                    metadata_repaired = metadata_ambiguous = 0
                    for col in REVIEW_FIELDS:
                        txt, fixed, unresolved = restore_latin1(row_dict.get(col, ""))
                        normalized[col] = txt
                        if col == "review_text":
                            text_repaired += fixed
                            text_ambiguous += unresolved
                        else:
                            metadata_repaired += fixed
                            metadata_ambiguous += unresolved
                    normalized["rating"] = str(int(score))
                    normalized["submission_time"] = day.isoformat()
                    original_label = normalized["is_recommended"]
                    normalized["is_recommended"] = as_boolean(original_label)
                    if original_label and not normalized["is_recommended"]:
                        stats["review_invalid_recommendation_label"] += 1
                    for col in ["helpfulness", "price_usd", "total_feedback_count", "total_neg_feedback_count", "total_pos_feedback_count"]:
                        val = as_float(normalized[col])
                        normalized[col] = "" if val is None else str(int(val) if col.startswith("total_") and val.is_integer() else val)
                    normalized_text = re.sub(r"\s+", " ", normalized["review_text"]).casefold().strip()
                    txt_hash = sha256(normalized_text.encode("utf-8")).hexdigest() if normalized_text else ""
                    author = text_or_empty(normalized["author_id"])
                    author_text_sha = sha256((author + "\0" + normalized_text).encode("utf-8")).hexdigest() if author and normalized_text else ""
                    split = "train" if day.year <= 2021 else "validation" if day.year == 2022 else "test"
                    ordv = SPLIT_ORD[split]
                    if author:
                        first_author[author] = min(ordv, first_author.get(author, ordv))
                    if txt_hash:
                        first_text[txt_hash] = min(ordv, first_text.get(txt_hash, ordv))
                    normalized.update({
                        "review_id": sha1((file.name + "|" + str(source_row)).encode("utf-8")).hexdigest(),
                        "source_file": file.name, "source_row": source_row, "source_index": ref_index,
                        "text_sha": txt_hash, "author_text_sha": author_text_sha,
                        "text_repaired": text_repaired, "text_ambiguous": text_ambiguous,
                        "metadata_repaired": metadata_repaired,
                        "metadata_ambiguous": metadata_ambiguous,
                    })
                    writer.writerow(normalized)
                    per["valid"] += 1
                    per["text_repair_pairs"] += text_repaired
                    per["text_ambiguous_count"] += text_ambiguous
                    per["metadata_repair_pairs"] += metadata_repaired
                    per["metadata_ambiguous_count"] += metadata_ambiguous
                    if text_repaired:
                        per["rows_with_repair"] += 1
                    if text_ambiguous:
                        per["rows_with_ambiguous_text"] += 1
                    if metadata_repaired or metadata_ambiguous:
                        per["rows_with_metadata_damage"] += 1
                    stats["review_rows_valid"] += 1
                    stats["review_text_repair_pairs"] += text_repaired
                    stats["review_text_ambiguous_chars"] += text_ambiguous
                    stats["review_metadata_repair_pairs"] += metadata_repaired
                    stats["review_metadata_ambiguous_chars"] += metadata_ambiguous
                    if text_repaired:
                        stats["review_rows_with_text_repair"] += 1
            file_stats[file.name] = {"size_bytes": file.stat().st_size, **dict(per)}
    return first_author, first_text, file_stats


def publish_review_datasets(staging: Path, clean_path: Path, m4_path: Path, m5_path: Path,
                            first_author: dict[str, int], first_text: dict[str, int],
                            stats: Counter, min_review_chars: int) -> None:
    """리뷰 작성일 순서로 데이터를 나누고 시기 사이의 작성자·문장 중복 제외.

    앞선 기간에 나온 작성자 또는 동일 문장이 뒤 기간에 다시 등장하면 뒤 기간 행을 모델 입력에서 제외.
    임시 저장본은 학습에 사용하지 않으며, 분할 정보가 부여된 적합한 행만 모델 입력 파일에 저장.
    """
    seen_author_text: set[str] = set()
    seen_text: set[str] = set()
    with staging.open("r", encoding="utf-8", newline="") as stage, \
         clean_path.open("w", encoding="utf-8", newline="") as clean_file, \
         m4_path.open("w", encoding="utf-8", newline="") as m4_file, \
         m5_path.open("w", encoding="utf-8", newline="") as m5_file:
        reader = csv.DictReader(stage)
        clean = csv.DictWriter(clean_file, fieldnames=CLEAN_COLS, extrasaction="ignore")
        m4 = csv.DictWriter(m4_file, fieldnames=MODEL_REVIEW_COLS, extrasaction="ignore")
        m5 = csv.DictWriter(m5_file, fieldnames=M5_COLS, extrasaction="ignore")
        clean.writeheader(); m4.writeheader(); m5.writeheader()
        for row in reader:
            day = as_date(row["submission_time"])
            assert day is not None
            raw_split = "train" if day.year <= 2021 else "validation" if day.year == 2022 else "test"
            rank = SPLIT_ORD[raw_split]
            reasons: list[str] = []
            author = text_or_empty(row["author_id"])
            if author and first_author[author] < rank:
                reasons.append("author_seen_in_earlier_period")
            if row["text_sha"] and first_text[row["text_sha"]] < rank:
                reasons.append("text_seen_in_earlier_period")
            split = raw_split if not reasons else "excluded_cross_split"
            author_text_key = row["author_text_sha"]
            is_repeated = int(bool(author_text_key and author_text_key in seen_author_text))
            is_duplicate_text = int(bool(row["text_sha"] and row["text_sha"] in seen_text))
            if author_text_key:
                seen_author_text.add(author_text_key)
            if row["text_sha"]:
                seen_text.add(row["text_sha"])
            row.update({"raw_split": raw_split, "split": split,
                        "cross_split_exclusion_reason": ";".join(reasons),
                        "is_repeated_author_text": is_repeated,
                        "is_duplicate_text": is_duplicate_text})
            clean.writerow(row)
            stats[f"split_{split}"] += 1
            if reasons:
                stats["cross_period_exclusions"] += 1
                continue
            if is_repeated:
                stats["duplicate_author_text_rows"] += 1
                continue  # 전체 정제본에는 남기되 모델 입력에서는 제외.
            if len(text_or_empty(row["review_text"])) < min_review_chars:
                stats["short_or_empty_review_excluded_from_model_inputs"] += 1
                continue
            if int(row["text_ambiguous"] or 0) > 0:
                stats["unresolved_text_excluded_from_model_inputs"] += 1
                continue  # 복원 불가 문자가 있으면 기본적으로 모델 입력에서 제외.
            m4.writerow(row)
            stats["m4_model_input_rows"] += 1
            if day.year >= 2018 and row["is_recommended"] in ("0", "1"):
                m5.writerow(row)
                stats["m5_model_input_rows"] += 1


def checksum_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="K-뷰티/Sephora CSV 재현 가능한 데이터 전처리")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="원본 CSV가 있는 data 폴더 경로")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/preprocessed"), help="전처리 결과 저장 폴더 경로")
    p.add_argument("--min-review-chars", type=int, default=20, help="모델 입력에 포함할 리뷰 본문의 최소 글자 수(기본 20자)")
    p.add_argument("--hash-inputs", action="store_true", help="입력 CSV 전체 파일의 SHA-256 해시값 계산(추가 읽기 발생)")
    args = p.parse_args(argv)
    setup_csv_limit()
    data_dir = args.data_dir.resolve()
    out = args.out_dir.resolve()
    if out == data_dir or data_dir in out.parents:
        raise ValueError("출력 폴더는 원본 data 폴더 바깥에 지정하세요")
    product_path = data_dir / "product_info.csv"
    reviews = sorted(data_dir.glob("reviews_*.csv"), key=file_order)
    if not product_path.is_file() or not reviews:
        raise FileNotFoundError(f"product_info.csv 또는 reviews_*.csv 없음: {data_dir}")
    if args.min_review_chars < 1:
        raise ValueError("min-review-chars는 1 이상이어야 합니다")
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    paths = {
        "products_clean": out / "products_clean.csv",
        "products_quarantine": out / "products_quarantine.csv",
        "reviews_staging": out / "_reviews_staging.csv",
        "reviews_quarantine": out / "reviews_quarantine.csv",
        "reviews_clean": out / "reviews_clean.csv",
        "reviews_m4": out / "reviews_m4_input.csv",
        "reviews_m5": out / "reviews_m5_input.csv",
    }
    print(f"[1/3] 상품 정보 처리: {product_path.name}", flush=True)
    product_ids = load_products(product_path, paths["products_clean"], paths["products_quarantine"], stats)
    print(f"[2/3] 리뷰 처리: {len(reviews)}개 파일 / {sum(f.stat().st_size for f in reviews)/1e6:.1f} MB", flush=True)
    first_author, first_text, per_file = stage_reviews(reviews, paths["reviews_staging"],
                                                      paths["reviews_quarantine"], product_ids, stats)
    print("[3/3] 시기별 분할 / 중복 제외 / 모델 입력 파일 저장", flush=True)
    publish_review_datasets(paths["reviews_staging"], paths["reviews_clean"], paths["reviews_m4"], paths["reviews_m5"],
                            first_author, first_text, stats, args.min_review_chars)
    paths["reviews_staging"].unlink(missing_ok=True)
    input_files = [product_path, *reviews]
    # 기존 품질 보고서는 리뷰 파일 19개 기준이지만, 현재 전처리 파일은 2개가 병합된 17개임.
    expected = {"review_rows_parsed": 1_092_530, "review_rows_valid": 1_089_073}
    historical_comparison = {
        key: {"historical_report": val, "this_run": stats.get(key, 0),
              "difference": stats.get(key, 0) - val,
              "equal": stats.get(key, 0) == val}
        for key, val in expected.items()
    }
    manifest = {
        "pipeline": "KBeauty Sephora v1.1",
        "data_source": "data/ 폴더(출처:Kaggle)",
        "generated_utc": datetime.now().astimezone().isoformat(),
        "split_policy": {
            "train": "리뷰 작성일이 2021-12-31 이전 또는 당일", "validation": "2022-01-01부터 2022-12-31까지",
            "test": "리뷰 작성일이 2023-01-01 이후 또는 당일",
            "leakage_control": "이전 기간에 나온 작성자 또는 정규화된 동일 문장을 이후 기간 모델 입력에서 제외하며, 동일 작성자·문장 복제본도 모델 입력에서 제외",
            "m5_label": "2018년 이후 is_recommended 값 사용, M5 입력에는 리뷰 별점을 포함하지 않음",
            "m4_label": "속성별(Aspect) 정답 라벨을 임의 생성하지 않음. 지도학습 평가에는 별도의 수작업 라벨링 필요",
            "caveat": "시간 기준 분할과 작성자·문장 중복 제외는 보수적이므로 제외 후 표본 수와 분포 확인 필요",
        },
        "input_files": [{"file": f.name, "bytes": f.stat().st_size, **({"sha256": checksum_file(f)} if args.hash_inputs else {})} for f in input_files],
        "outputs": {k: v.name for k,v in paths.items() if k != "reviews_staging"},
        "stats": dict(stats),
        "per_review_file": per_file,
        "comparison_to_2026_10_22_quality_report": historical_comparison,
        "historical_count_warning": "기존 보고서 수치는 비교 기준일 뿐입니다. CSV 파서 차이, 파일 손상, 분할 파일 통합에 따라 실제 행 수가 달라질 수 있습니다.",
        "limitations": [
            "실제 폴더에 존재하는 리뷰 파일만 읽음. 기존 보고서는 분할 파일이 19개지만, 현재 파일은 2개가 병합되어 17개가 기준임.",
            "실제 원본 파일 수와 행 수가 기존 품질 보고서와 같다고 가정하지 않음",
            "손상된 CSV 따옴표 때문에 여러 행이 합쳐지거나 유실됐을 수 있으며 격리 작업만으로 유실 행을 되살릴 수 없음.",
            "손상 문장부호 복원은 규칙 기반 추정이므로 일부 표본을 사람이 원문과 대조해야 함.",
            "리뷰는 자발적 작성 자료이며 기존 보고서상 2023년 3월까지의 과거 자료임. 현재 소비자 수요로 일반화하지 않음.",
            "제품 가격·용량의 정제 상태는 프리미엄 등급 정답이 아님. 브랜드의 K-뷰티 여부도 임의 추정하지 않음.",
            "평가지표와 합격선은 모델링 정의서 v1.0을 따름. 이 코드에서는 모델을 학습하지 않습니다.",
            "격리본에는 리뷰 원문과 피부·건강 관련 서술이 포함될 수 있습니다. 공개 저장소에 올리기 전에 데이터 재배포 조건과 개인정보를 점검하세요.",
        ],
    }
    write_json(out / "quality_report.json", manifest)
    print(json.dumps({"outputs": str(out), "stats": dict(stats)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
