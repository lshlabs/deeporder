import os
import sys
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Iterable

import numpy as np

try:
    from PIL import ImageGrab
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("Pillow(ImageGrab) 모듈이 필요합니다. 'pip install pillow'로 설치하세요.") from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV 라이브러리가 필요합니다. 'pip install opencv-python'으로 설치하세요.") from exc

try:
    import easyocr
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("easyocr 라이브러리가 필요합니다. 'pip install easyocr'로 설치하세요.") from exc


BAEMIN_KEYWORD = "신규주문"
COUPANG_KEYWORD = "거절"

BAEMIN_PATTERNS: Dict[str, Dict[str, Iterable[str]]] = {
    "accept_button": {"type": "exact", "values": ["접수"]},
    "reject_button": {"type": "exact", "values": ["거부"]},
    "time": {"type": "regex", "values": [r"\d{1,2}~\d{1,2}분", r"\d{1,2}\s*분"]},
    "increase_time": {"type": "exact", "values": ["+"]},
    "reduce_time": {"type": "exact", "values": ["-"]},
}

COUPANG_PATTERNS: Dict[str, Dict[str, Iterable[str]]] = {
    "accept_button": {"type": "exact", "values": ["수락"]},
    "reject_button": {"type": "exact", "values": ["거절"]},
    "time": {"type": "regex", "values": [r"\d{1,2}\s*분"]},
    "increase_time": {"type": "exact", "values": ["+5", "+ 5"]},
    "reduce_time": {"type": "exact", "values": ["-5", "- 5"]},
}

CAPTURE_DELAY = 1.5  # seconds between screen captures


def initialize_reader() -> "easyocr.Reader":
    print("[INFO] easyocr Reader 초기화 중...", flush=True)
    reader = easyocr.Reader(["ko", "en"], gpu=False, quantize=False)
    print("[INFO] easyocr Reader 초기화 완료", flush=True)
    return reader


def capture_screen() -> np.ndarray:
    image = ImageGrab.grab()
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def run_ocr(reader: "easyocr.Reader", image: np.ndarray) -> List[Dict[str, object]]:
    results = reader.readtext(image)
    parsed = []
    for bbox, text, confidence in results:
        parsed.append({
            "bbox": bbox,
            "text": text.strip(),
            "confidence": confidence,
        })
    return parsed


def normalize_text(text: str) -> str:
    return text.replace(" ", "")


def match_exact(source: str, targets: Iterable[str]) -> bool:
    normalized_source = normalize_text(source)
    for target in targets:
        if normalize_text(target) == normalized_source:
            return True
    return False


def match_regex(source: str, patterns: Iterable[str]) -> bool:
    normalized_source = normalize_text(source)
    for pattern in patterns:
        if re.fullmatch(pattern, normalized_source):
            return True
    return False


def collect_matches(
    entries: List[Dict[str, object]],
    patterns: Dict[str, Dict[str, Iterable[str]]],
) -> Dict[str, List[Dict[str, object]]]:
    matches: Dict[str, List[Dict[str, object]]] = {key: [] for key in patterns}

    for entry in entries:
        text = str(entry["text"])
        for element_key, descriptor in patterns.items():
            match_type = descriptor.get("type")
            values = descriptor.get("values", [])
            if match_type == "exact" and match_exact(text, values):
                matches[element_key].append(entry)
            elif match_type == "regex" and match_regex(text, values):
                matches[element_key].append(entry)

    return matches


def annotate_image(
    image: np.ndarray,
    keyword_matches: List[Dict[str, object]],
    element_matches: Dict[str, List[Dict[str, object]]],
    keyword_label: str,
) -> np.ndarray:
    annotated = image.copy()

    def draw_box(bbox: List[Tuple[float, float]], color: Tuple[int, int, int], label: str) -> None:
        points = np.array(bbox).astype(int)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
        x_min, y_min = points.min(axis=0)
        cv2.putText(
            annotated,
            label,
            (int(x_min), int(y_min) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    for entry in keyword_matches:
        draw_box(entry["bbox"], (0, 255, 255), f"keyword:{keyword_label}")

    color_map = {
        "accept_button": (0, 255, 0),
        "reject_button": (0, 0, 255),
        "time": (255, 255, 0),
        "increase_time": (255, 0, 255),
        "reduce_time": (255, 128, 0),
    }

    for element, entries in element_matches.items():
        for entry in entries:
            color = color_map.get(element, (255, 255, 255))
            draw_box(entry["bbox"], color, element)

    return annotated


def ensure_output_path(root: str) -> str:
    timestamp = datetime.now().strftime("%m%d%H%M")
    output_dir = os.path.join(root, "test_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_result(image: np.ndarray, output_dir: str, prefix: str) -> str:
    filename = f"{prefix}_{datetime.now().strftime('%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def monitor_screen(reader: "easyocr.Reader") -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    last_save_time: Dict[str, float] = {}

    while True:
        try:
            frame = capture_screen()
            ocr_entries = run_ocr(reader, frame)

            if not ocr_entries:
                time.sleep(CAPTURE_DELAY)
                continue

            matches_baemin_keyword = [entry for entry in ocr_entries if normalize_text(entry["text"]) == normalize_text(BAEMIN_KEYWORD)]
            matches_coupang_keyword = [entry for entry in ocr_entries if normalize_text(entry["text"]) == normalize_text(COUPANG_KEYWORD)]

            detected = False

            if matches_baemin_keyword:
                element_matches = collect_matches(ocr_entries, BAEMIN_PATTERNS)
                annotated = annotate_image(frame, matches_baemin_keyword, element_matches, "baemin")
                output_dir = ensure_output_path(project_root)
                filepath = save_result(annotated, output_dir, "baemin")
                detected = True
                print(f"[INFO] Baemin 주문 감지 - 저장 경로: {filepath}", flush=True)

            if matches_coupang_keyword:
                element_matches = collect_matches(ocr_entries, COUPANG_PATTERNS)
                annotated = annotate_image(frame, matches_coupang_keyword, element_matches, "coupang")
                output_dir = ensure_output_path(project_root)
                filepath = save_result(annotated, output_dir, "coupang")
                detected = True
                print(f"[INFO] Coupang 주문 감지 - 저장 경로: {filepath}", flush=True)

            if not detected:
                time.sleep(CAPTURE_DELAY)

        except KeyboardInterrupt:
            print("\n[INFO] 사용자가 모니터링을 중지했습니다.", flush=True)
            break
        except Exception as exc:  # pragma: no cover - 보호 로깅
            print(f"[ERROR] 예기치 못한 오류가 발생했습니다: {exc}", file=sys.stderr, flush=True)
            time.sleep(CAPTURE_DELAY)


def main() -> None:
    reader = initialize_reader()
    print("[INFO] 화면 모니터링을 시작합니다. 종료하려면 Ctrl+C를 누르세요.", flush=True)
    monitor_screen(reader)


if __name__ == "__main__":
    main()

