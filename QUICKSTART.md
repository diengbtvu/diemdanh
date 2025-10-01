# ⚡ QUICKSTART - Hướng dẫn chạy nhanh

## 🚀 Chạy ngay trong 3 bước

### Bước 1: Cài đặt packages
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dataset
Đặt dataset vào folder `aligned_faces/` theo cấu trúc:
```
aligned_faces/
├── student_001/
│   ├── img1.jpg
│   └── img2.jpg
├── student_002/
│   └── ...
└── ...
```

### Bước 3: Chạy!

**Windows:**
```bash
run.bat
```
hoặc
```bash
python main.py
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```
hoặc
```bash
python3 main.py
```

## ✅ Chương trình sẽ tự động:

1. ✅ Kiểm tra GPU và dataset
2. ✅ Train 4 models với Early Stopping
3. ✅ Evaluate và tính toán metrics
4. ✅ Tạo visualizations
5. ✅ Export reports (CSV, JSON, Markdown)
6. ✅ Đưa ra recommendations

## 📁 Kết quả

Tất cả kết quả trong folder `face_detection_results/`:

- **Models:** `*_best_model.pth` (hoặc `.pkl`)
- **Reports:** `SUMMARY_REPORT.md`, `*.csv`, `*.json`
- **Charts:** `*.png` files

## ⏱️ Thời gian chạy

- **GPU:** ~30-60 phút
- **CPU:** ~2-4 giờ

## 🔧 Troubleshooting nhanh

### Không tìm thấy dataset?
➡️ Kiểm tra folder `aligned_faces/` có đúng vị trí không

### Out of memory?
➡️ Mở `config.py`, giảm `BATCH_SIZE` xuống 16 hoặc 8

### Lỗi multiprocessing trên Windows?
➡️ Mở `config.py`, đặt `NUM_WORKERS = 0`

## 📊 Xem kết quả

Sau khi chạy xong, mở file `face_detection_results/SUMMARY_REPORT.md` để xem tổng quan.

---

**Chúc may mắn! 🎓**

