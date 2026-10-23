#!/usr/bin/env python3
"""대용량 전처리 CSV에서 검증에 필요한 작은 표본만 추출합니다.

사용법:
    python 검증용_샘플_추출.py --input-dir ./outputs/preprocessed

원본 및 전처리 결과 파일을 수정하지 않습니다. Python 표준 라이브러리만 사용합니다.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import tempfile
import zipfile


# 재현 가능하도록 난수를 고정합니다. 행을 전부 메모리에 올리지 않습니다.
RANDOM_SEED = 20261023
MAX_SAMPLES_PER_TYPE = 12
MAX_SAMPLES_PER_REASON = 4
MAX_TEXT_LENGTH = 280


def increase_csv_field_limit() -> None:
    import sys
    size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(size)
            return
        except OverflowError:
            size //= 10


def sample_add(samples, counts, category, row, max_items, rng):
    """저장 공간을 일정하게 유지하면서 전체 행에 걸쳐 균등하게 추출합니다."""
    counts[category] += 1
    n = counts[category]
    bucket = samples[category]
    if len(bucket) < max_items:
        bucket.append(row)
    else:
        target = rng.randrange(n)
        if target < max_items:
            bucket[target] = row


def safe_text(value: object, length: int = MAX_TEXT_LENGTH) -> str:
    """이메일·전화번호를 가리고 너무 긴 문자열은 잘라 저장합니다.

    피부·건강 관련 개인적 서술까지 자동으로 식별하지는 못합니다.
    첨부하기 전에 사용자가 예문을 직접 확인해야 합니다.
    """
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "[이메일 숨김]", text)
    text = re.sub(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)", "[전화번호 숨김]", text)
    return text[:length] + ("…[이하 생략]" if len(text) > length else "")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def summarize_clean(clean_path: Path, product_groups: dict[str, str], rng):
    samples = defaultdict(list)
    counts = Counter()
    splits = Counter()
    product_counts = Counter()
    split_by_group = Counter()
    for row in read_csv(clean_path):
        split = row.get("split", "미상") or "미상"
        splits[split] += 1
        product = row.get("product_id", "")
        group = product_groups.get(product, "상품군 확인 불가")
        product_counts[group] += 1
        split_by_group[(group, split)] += 1

        fixed = int(row.get("text_repaired") or 0)
        unclear = int(row.get("text_ambiguous") or 0)
        if unclear:
            category = "손상 미복원"
        elif fixed:
            category = "문장 복원"
        elif split == "excluded_cross_split":
            category = "기간 간 중복 제외"
        else:
            category = "손상 없음"
        slim = {
            "검증 유형": category,
            "원본 파일": row.get("source_file", ""),
            "원본 행 번호": row.get("source_row", ""),
            "상품 ID": product,
            "상품군": group,
            "데이터 분할": split,
            "복원된 문장부호 수": fixed,
            "미복원 문자 수": unclear,
            "문장 중복 여부": row.get("is_duplicate_text", ""),
            "검토용 리뷰 본문": safe_text(row.get("review_text", "")),
        }
        sample_add(samples, counts, category, slim, MAX_SAMPLES_PER_TYPE, rng)
    return samples, splits, product_counts, split_by_group, counts


def summarize_quarantine(path: Path, rng):
    samples = defaultdict(list)
    counts = Counter()
    for row in read_csv(path):
        reason = row.get("reason", "미상") or "미상"
        raw = row.get("raw_json", "")
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = []
        # 원본 리뷰의 일반적인 위치(첫 열은 인덱스, 둘째 열은 작성자 식별자).
        # 원본 구조가 손상됐으면 위치가 달라질 수 있으므로 검토용 단서로만 사용합니다.
        if isinstance(parsed, list):
            preview = ["[작성자 식별자 숨김]" if i == 1 else safe_text(v, 90)
                       for i, v in enumerate(parsed[:21])]
            field_count = len(parsed)
        else:
            preview = ["파싱 가능한 배열이 아님"]
            field_count = -1
        slim = {
            "오류 유형": reason,
            "원본 파일": row.get("source_file", ""),
            "원본 행 번호": row.get("source_row", ""),
            "원본 실제 줄 번호": row.get("source_line", ""),
            "해석한 필드 수": field_count,
            "검토용 일부 필드(작성자 숨김)": json.dumps(preview, ensure_ascii=False),
        }
        sample_add(samples, counts, reason, slim, MAX_SAMPLES_PER_REASON, rng)
    return samples, counts


def model_splits(path: Path) -> dict[str, int]:
    counts = Counter()
    if path.exists():
        for row in read_csv(path):
            counts[row.get("split", "미상") or "미상"] += 1
    return dict(sorted(counts.items()))


def save_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="K-뷰티 전처리 결과에서 소용량 검증 표본 추출")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/preprocessed"),
                        help="전처리 결과(products_clean.csv 등)가 저장된 폴더")
    parser.add_argument("--output", type=Path, default=Path("검증용_소용량_샘플.zip"),
                        help="생성할 ZIP 파일 경로")
    args = parser.parse_args()
    source = args.input_dir
    required = [source / x for x in ("products_clean.csv", "reviews_clean.csv", "reviews_quarantine.csv")]
    absent = [p.name for p in required if not p.is_file()]
    if absent:
        parser.error("다음 전처리 파일이 없습니다: " + ", ".join(absent))
    increase_csv_field_limit()
    rng = random.Random(RANDOM_SEED)

    product_groups = {
        row.get("product_id", ""): row.get("primary_category", "미분류") or "미분류"
        for row in read_csv(source / "products_clean.csv")
    }
    print("리뷰 정제본에서 표본·상품군 통계를 추출합니다. (전체 파일은 읽지만 메모리에 올리지는 않습니다.)", flush=True)
    reviews, splits, groups, cross, review_counts = summarize_clean(source / "reviews_clean.csv", product_groups, rng)
    print("격리된 리뷰에서 오류 유형별 표본을 추출합니다.", flush=True)
    quarantines, quarantine_counts = summarize_quarantine(source / "reviews_quarantine.csv", rng)

    review_rows = [record for label, records in sorted(reviews.items()) for record in records]
    quarantine_rows = [record for label, records in sorted(quarantines.items()) for record in records]
    summary = {
        "안내": "개별 리뷰 본문을 첨부 전 직접 확인하고 민감한 내용이 있으면 가리거나 해당 행을 삭제하세요.",
        "상품군별 정상 리뷰 건수": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
        "전체 정제본 분할별 건수": dict(sorted(splits.items())),
        "상품군별 분할 건수": [
            {"상품군": name, "분할": split, "건수": n}
            for (name, split), n in sorted(cross.items())
        ],
        "M4 모델 입력 분할별 건수": model_splits(source / "reviews_m4_input.csv"),
        "M5 모델 입력 분할별 건수": model_splits(source / "reviews_m5_input.csv"),
        "오류 유형별 격리 건수": dict(quarantine_counts),
        "표본 유형별 해당 리뷰 건수": dict(review_counts),
        "실제 추출 표본": {"리뷰": len(review_rows), "격리": len(quarantine_rows)},
    }

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        save_csv(folder / "리뷰_검증_표본.csv", review_rows,
                 ["검증 유형", "원본 파일", "원본 행 번호", "상품 ID", "상품군", "데이터 분할",
                  "복원된 문장부호 수", "미복원 문자 수", "문장 중복 여부", "검토용 리뷰 본문"])
        save_csv(folder / "격리_오류_표본.csv", quarantine_rows,
                 ["오류 유형", "원본 파일", "원본 행 번호", "원본 실제 줄 번호", "해석한 필드 수",
                  "검토용 일부 필드(작성자 숨김)"])
        (folder / "상품군_및_모델분할_집계.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "첨부전_확인.txt").write_text(
            "이 압축 파일은 대용량 전체 CSV가 아닌 검증용 표본 및 집계입니다.\n"
            "표본에 리뷰 문장 또는 피부·건강 관련 개인적 서술이 포함될 수 있습니다.\n"
            "업로드하기 전에 CSV를 열고 민감한 부분을 가리거나 행을 삭제하세요.\n"
            "작성자 ID는 가렸지만 모든 개인정보가 자동 제거된 것은 아닙니다.\n",
            encoding="utf-8")
        dest = args.output.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for p in sorted(folder.iterdir()):
                archive.write(p, p.name)
    print(f"완료: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    print("ZIP 안의 예시 문장을 확인한 뒤 업로드해 주세요.")


if __name__ == "__main__":
    main()
