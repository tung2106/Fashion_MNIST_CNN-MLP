"""Xuất đúng ảnh dataset[1] để xem khi chưa có torch/matplotlib.

Chạy: python 01_kham_pha_du_lieu/xuat_anh_khong_can_thu_vien.py
"""
from pathlib import Path
import struct
import zlib


def main():
    folder = Path(__file__).resolve().parent
    raw = folder.parent / "du_lieu_chung" / "FashionMNIST" / "raw"
    with (raw / "train-images-idx3-ubyte").open("rb") as stream:
        magic, count, rows, cols = struct.unpack(">4I", stream.read(16))
        if magic != 2051 or count < 2 or (rows, cols) != (28, 28):
            raise ValueError("Invalid Fashion-MNIST image file")
        stream.seek(rows * cols, 1)  # Bỏ ảnh 0, lấy ảnh 1 như bài học.
        pixels = stream.read(rows * cols)
    if len(pixels) != rows * cols:
        raise ValueError("Incomplete image data")
    with (raw / "train-labels-idx1-ubyte").open("rb") as stream:
        magic, count = struct.unpack(">2I", stream.read(8))
        if magic != 2049 or count < 2:
            raise ValueError("Invalid Fashion-MNIST label file")
        stream.read(1)
        label = stream.read(1)[0]

    # Phóng to 10 lần, giữ nguyên từng pixel để quan sát rõ.
    scale = 10
    scanlines = b"".join(
        b"\x00" + bytes(pixels[(y // scale) * cols + x // scale]
                        for x in range(cols * scale))
        for y in range(rows * scale)
    )

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    output = folder / "anh_du_lieu_1.png"
    output.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", cols * scale, rows * scale, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    print(f"Image index: 1; label: {label}; saved: {output}")


if __name__ == "__main__":
    main()
