import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from cccd_model import CCCDData
from db import save_cccd_to_db
from preprocessing import preprocess_for_ocr

# ---------------------------------------------------------------------------
# Models – loaded once, reused across threads
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    yolo = YOLO("best.pt")
    print("YOLO classes:", yolo.names)
    cfg = Cfg.load_config_from_name('vgg_transformer')
    cfg['device'] = 'cpu'
    cfg['predictor']['beamsearch'] = False
    ocr = Predictor(cfg)
    return yolo, ocr

# ---------------------------------------------------------------------------
# OCR pipeline (shared by camera and upload modes)
# ---------------------------------------------------------------------------

def run_ocr_pipeline(image: Image.Image, yolo_model, reader) -> CCCDData:
    """
    Chạy YOLO + preprocessing + VietOCR trên một PIL Image.
    Nhận model trực tiếp để an toàn khi gọi từ background thread.
    Trả về CCCDData đã được gán đầy đủ field.
    """
    iw, ih = image.size
    pad = 6

    results = yolo_model.predict(image, conf=0.25, verbose=False)
    cccd_data = CCCDData()

    if not (results and len(results[0].boxes) > 0):
        return cccd_data

    # --- Pass 1: OCR tất cả box ---
    detections = []
    for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        class_name = yolo_model.names[int(cls_id)]
        x1, y1, x2, y2 = map(int, box)
        x1p = max(0, x1 - pad);  y1p = max(0, y1 - pad)
        x2p = min(iw, x2 + pad); y2p = min(ih, y2 + pad)
        cropped = image.crop((x1p, y1p, x2p, y2p))
        cropped_bgr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        processed_img, _ = preprocess_for_ocr(cropped_bgr)
        text = reader.predict(Image.fromarray(processed_img))
        detections.append({
            "class_name": class_name,
            "text": text,
            "box_coords": (x1, y1, x2, y2),
        })

    # --- Pass 2: reassign gender / current_place ---
    def _read_label_above(box_coords):
        x1, y1, x2, y2 = box_coords
        label_crop = image.crop((x1, max(0, y1 - 120), x2, y1))
        label_bgr = cv2.cvtColor(np.array(label_crop), cv2.COLOR_RGB2BGR)
        label_proc, _ = preprocess_for_ocr(label_bgr)
        return reader.predict(Image.fromarray(label_proc))

    ambiguous = {"gender", "current_place"}
    for d in detections:
        if "/" in d["text"]:
            d["class_name"] = "dob"
            continue
        if d["class_name"] not in ambiguous:
            continue
        label_lower = _read_label_above(d["box_coords"]).lower()
        if "sex" in label_lower or "tính" in label_lower or "tinh" in label_lower:
            d["class_name"] = "gender"
        else:
            d["class_name"] = "current_place"

    # --- Pass 3: gán vào CCCDData ---
    for d in detections:
        t = d["text"]
        match d["class_name"]:
            case "name":          cccd_data.name = t
            case "id":            cccd_data.id = t
            case "dob":           cccd_data.dob = t
            case "gender":        cccd_data.gender = t
            case "origin_place":  cccd_data.origin_place = t
            case "current_place":
                cccd_data.current_place = (
                    t if cccd_data.current_place is None
                    else cccd_data.current_place + ", " + t
                )

    return cccd_data

# ---------------------------------------------------------------------------
# Video processor – state machine chạy trong background thread
# ---------------------------------------------------------------------------

STILL_THRESHOLD    = 1.5   # mean pixel diff (grayscale blurred) dưới mức này = đứng im
STILL_FRAMES_NEEDED = 8    # cần N frame liên tiếp đứng im mới chụp

class CCCDScanner(VideoProcessorBase):
    """
    State machine:
      searching    → YOLO quét mỗi frame, chờ vật thể vào
      stabilizing  → vật thể vào, đo độ rung; hiện progress bar
      scanning     → vật thể đứng im, OCR chạy background thread
      scanned      → OCR xong, chỉ hiện camera + kết quả, không reset

    Transitions:
      searching   → stabilizing : YOLO detect lần đầu
      stabilizing → searching   : vật thể rời khỏi frame
      stabilizing → scanning    : đứng im đủ STILL_FRAMES_NEEDED frames
      scanning    → scanned     : OCR thread hoàn thành
      scanned     → (terminal)  : chỉ focus, không scan thêm
    """

    def __init__(self):
        self.yolo_model, self.reader = load_models()
        self._lock = threading.Lock()
        self.state = "searching"
        self.cccd_result: CCCDData | None = None
        self._prev_gray = None       # frame trước (grayscale blur) để đo độ rung
        self._still_count = 0        # số frame liên tiếp đứng im
        self._scanned_at: float = 0  # timestamp khi chuyển sang scanned

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.yolo_model.predict(img, conf=0.4, verbose=False)
        detected = bool(results and len(results[0].boxes) > 0)

        # Grayscale blur để đo độ rung chính xác hơn
        gray_blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        with self._lock:
            if detected:
                # Vẽ bounding box
                for box in results[0].boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if self.state == "searching":
                    self.state = "stabilizing"
                    self._still_count = 0
                    self._prev_gray = gray_blur

                elif self.state == "stabilizing":
                    # Đo độ rung so với frame trước
                    if self._prev_gray is not None:
                        diff = cv2.absdiff(gray_blur, self._prev_gray).mean()
                        if diff < STILL_THRESHOLD:
                            self._still_count += 1
                        else:
                            self._still_count = 0
                    self._prev_gray = gray_blur

                    # Hiện progress bar độ ổn định
                    progress = min(self._still_count / STILL_FRAMES_NEEDED, 1.0)
                    bar_w = int(220 * progress)
                    cv2.rectangle(img, (10, 55), (230, 75), (60, 60, 60), -1)
                    cv2.rectangle(img, (10, 55), (10 + bar_w, 75), (0, 220, 220), -1)
                    cv2.putText(img, "Giu yen...", (10, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)

                    if self._still_count >= STILL_FRAMES_NEEDED:
                        self.state = "scanning"
                        threading.Thread(
                            target=self._run_ocr,
                            args=(img.copy(),),
                            daemon=True,
                        ).start()

                if self.state == "scanning":
                    cv2.putText(img, "Dang quet...", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
                elif self.state == "scanned":
                    remaining = max(0, 10 - int(time.time() - self._scanned_at))
                    cv2.putText(img, f"Da quet! Quet lai sau {remaining}s", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if remaining == 0:
                        self.state = "searching"
                        self.cccd_result = None
                        self._still_count = 0
                        self._prev_gray = None

            else:
                # Vật thể rời frame
                self._prev_gray = None
                self._still_count = 0
                # Chỉ reset khi chưa scan xong
                if self.state in ("searching", "stabilizing"):
                    self.state = "searching"
                    cv2.putText(img, "Dang tim CCCD...", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                elif self.state == "scanned":
                    cv2.putText(img, "Da quet xong!", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _run_ocr(self, img_bgr: np.ndarray):
        """Chạy pipeline OCR trong thread riêng, không block video stream."""
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # --- Debug: kiểm tra YOLO trước ---
        results = self.yolo_model.predict(pil_img, conf=0.4, verbose=False)
        if not (results and len(results[0].boxes) > 0):
            print("[OCR] YOLO không detect được box nào — thử hạ conf xuống")
            # Thử lại với conf thấp hơn
            results = self.yolo_model.predict(pil_img, conf=0.2, verbose=False)

        if results and len(results[0].boxes) > 0:
            print(f"[OCR] YOLO detect {len(results[0].boxes)} boxes:")
            for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                print(f"  class={self.yolo_model.names[int(cls_id)]}  box={[int(v) for v in box]}")
        else:
            print("[OCR] YOLO không detect được box nào dù conf=0.2")

        cccd_data = run_ocr_pipeline(pil_img, self.yolo_model, self.reader)
        print("========== CCCD ==========")
        print(cccd_data)
        save_cccd_to_db(cccd_data)
        print("========== CCCD ==========")
        print(cccd_data)
        with self._lock:
            self.cccd_result = cccd_data
            self.state = "scanned"
            self._scanned_at = time.time()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CCCD OCR Demo", layout="centered")
st.title("📄 The recruitment system extracts data from the ID card photo.")

input_mode = st.radio(
    "Chọn nguồn ảnh:",
    ["📤 Upload file", "📷 Camera realtime"],
    horizontal=True,
)

# ── Camera realtime ─────────────────────────────────────────────────────────
if input_mode == "📷 Camera realtime":
    ctx = webrtc_streamer(
        key="cccd-scanner",
        video_processor_factory=CCCDScanner,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    status_ph = st.empty()
    result_ph = st.empty()

    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                with ctx.video_processor._lock:
                    state  = ctx.video_processor.state
                    result = ctx.video_processor.cccd_result

                if state == "searching":
                    status_ph.info("🔍 Đang tìm CCCD...")
                    result_ph.empty()
                elif state == "scanning":
                    status_ph.warning("⏳ Đang quét...")
                elif state == "scanned" and result:
                    status_ph.success("✅ Đã quét xong!")
                    with result_ph.container():
                        st.write("**Họ tên:**",        result.name)
                        st.write("**Số CCCD:**",       result.id)
                        st.write("**Ngày sinh:**",     result.dob)
                        st.write("**Giới tính:**",     result.gender)
                        st.write("**Quê quán:**",      result.origin_place)
                        st.write("**Nơi thường trú:**",result.current_place)
                        st.success("🎉 Đã lưu vào PostgreSQL!")

            time.sleep(0.5)

# ── Upload file ──────────────────────────────────────────────────────────────
else:
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã upload", use_column_width=True)

        with st.spinner("Đang nhận diện..."):
            yolo_model, reader = load_models()
            cccd_data = run_ocr_pipeline(image, yolo_model, reader)
            save_cccd_to_db(cccd_data)

        if cccd_data.id:
            st.success("✅ Nhận diện thành công!")
            st.write("**Họ tên:**",         cccd_data.name)
            st.write("**Số CCCD:**",        cccd_data.id)
            st.write("**Ngày sinh:**",      cccd_data.dob)
            st.write("**Giới tính:**",      cccd_data.gender)
            st.write("**Quê quán:**",       cccd_data.origin_place)
            st.write("**Nơi thường trú:**", cccd_data.current_place)
            st.success("🎉 Đã lưu vào PostgreSQL!")
        else:
            st.warning("Không tìm thấy CCCD trong ảnh!")
