"""Lossless PNG16 and native float EXR boundaries for the BF16 delivery.

No AI, GPU, AE, network or color-transfer operations. PNG consumers inspect the
actual IHDR and CRCs. EXR is read/written as planar IEEE float32 data by the
installed FFmpeg, with an independent channel-storage header check.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(argv, *, data=None, timeout=120):
    result = subprocess.run([str(v) for v in argv], input=data, capture_output=True,
                            timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-8000:])
    return result.stdout


def png_chunks(path):
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    offset, chunks, ended = 8, [], False
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("Truncated PNG payload")
        value = data[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + value) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch in " + repr(kind))
        chunks.append((kind, value))
        offset = end
        if kind == b"IEND":
            if value or offset != len(data):
                raise ValueError("Invalid IEND or trailing PNG bytes")
            ended = True
            break
    if not ended or not chunks or chunks[0][0] != b"IHDR":
        raise ValueError("PNG requires first IHDR and final IEND")
    if sum(k == b"IHDR" for k, _ in chunks) != 1:
        raise ValueError("Duplicate PNG IHDR")
    return chunks


def png_header(path):
    chunks = png_chunks(path)
    width, height, depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunks[0][1])
    if width < 1 or height < 1 or depth not in (8, 16) or color_type not in CHANNELS:
        raise ValueError("Only nonpalette RGB/RGBA/gray/GA PNG8 or PNG16 supported")
    if compression != 0 or filtering != 0 or interlace != 0:
        raise ValueError("PNG must use standard compression and noninterlaced rows")
    unknown_critical = [k for k, _ in chunks if not k[0] & 32 and k not in (b"IHDR", b"PLTE", b"IDAT", b"IEND")]
    if unknown_critical:
        raise ValueError("Unknown critical PNG chunk")
    return {"width": width, "height": height, "bit_depth": depth,
            "color_type": color_type, "channels": CHANNELS[color_type],
            "chunk_types": [k.decode("ascii") for k, _ in chunks]}


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def read_png(path):
    header = png_header(path)
    chunks = png_chunks(path)
    if any(k in (b"tRNS", b"acTL") for k, _ in chunks):
        raise ValueError("Implicit tRNS alpha or animated PNG is not supported")
    w, h, c, depth = (header[k] for k in ("width", "height", "channels", "bit_depth"))
    if depth == 8 or (depth == 16 and c == 1):
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        expected_dtype = np.uint8 if depth == 8 else np.uint16
        if pixels.shape != (h, w, c) or (depth == 8 and pixels.dtype != expected_dtype):
            raise ValueError("PNG consumer changed shape/precision")
        if depth == 16:
            if not np.issubdtype(pixels.dtype, np.integer) or pixels.min() < 0 or pixels.max() > 65535:
                raise ValueError("Invalid gray16 decoder values")
            pixels = pixels.astype(np.uint16)
        return pixels, header
    # FFmpeg retains RGB/RGBA16 values; Pillow intentionally exposes these as8bit.
    fmt = {2: "ya16le", 3: "rgb48le", 4: "rgba64le"}[c]
    data = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                    "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "pipe:1"])
    if len(data) != h * w * c * 2:
        raise ValueError("PNG16 native decoder byte count mismatch")
    return np.frombuffer(data, dtype="<u2").reshape(h, w, c).copy(), header


def read_png_independent_rows(path):
    """Small fixture control for the standard-library PNG filter decoder."""
    header = png_header(path)
    chunks = png_chunks(path)
    w, h, c, depth = (header[k] for k in ("width", "height", "channels", "bit_depth"))
    # Standard row reconstruction is used for PNG16; Pillow downcasts RGBA16.
    bpp, stride = c * depth // 8, w * c * depth // 8
    encoded = zlib.decompress(b"".join(v for k, v in chunks if k == b"IDAT"))
    if len(encoded) != h * (1 + stride):
        raise ValueError("PNG decoded row byte count mismatch")
    restored = np.empty((h, stride), dtype=np.uint8)
    for y in range(h):
        start = y * (stride + 1)
        filter_type = encoded[start]
        row = np.frombuffer(encoded, np.uint8, stride, start + 1).copy()
        prev = restored[y - 1] if y else np.zeros(stride, np.uint8)
        if filter_type == 1:
            for channel in range(bpp):
                row[channel::bpp] = np.cumsum(row[channel::bpp], dtype=np.uint64).astype(np.uint8)
        elif filter_type == 2:
            row = (row.astype(np.uint16) + prev).astype(np.uint8)
        elif filter_type in (3, 4):
            for x in range(stride):
                a = int(row[x - bpp]) if x >= bpp else 0
                b = int(prev[x])
                cvalue = int(prev[x - bpp]) if x >= bpp else 0
                predictor = (a + b) // 2 if filter_type == 3 else _paeth(a, b, cvalue)
                row[x] = (int(row[x]) + predictor) & 255
        elif filter_type != 0:
            raise ValueError("Unknown PNG row filter")
        restored[y] = row
    if depth == 16:
        pixels = np.frombuffer(restored.tobytes(), dtype=">u2").astype(np.uint16).reshape(h, w, c)
    else:
        pixels = restored.reshape(h, w, c)
    return pixels, header


def _chunk(kind, value):
    return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)


def write_png(path, pixels):
    path, pixels = Path(path), np.asarray(pixels)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.ndim != 3 or pixels.shape[2] not in (1, 2, 3, 4) or pixels.dtype not in (np.uint8, np.uint16):
        raise ValueError("PNG pixels must be HWC uint8/uint16 with 1..4 channels")
    h, w, channels = pixels.shape
    depth, color_type = pixels.dtype.itemsize * 8, {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    encoded = pixels.astype(">u2", copy=False) if depth == 16 else pixels
    rows = encoded.view(np.uint8).reshape(h, -1)
    filtered = np.zeros((h, rows.shape[1] + 1), np.uint8)
    filtered[:, 1:] = rows
    data = PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 0))
    data += _chunk(b"IDAT", zlib.compress(filtered.tobytes(), 6)) + _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    decoded, header = read_png(path)
    if not np.array_equal(decoded, pixels) or header["bit_depth"] != depth:
        raise ValueError("PNG write/readback did not preserve every code value")
    return header


def exr_header(path):
    with Path(path).open("rb") as stream:
        if stream.read(4) != b"v/1\x01":
            raise ValueError("Invalid OpenEXR magic")
        version = struct.unpack("<I", stream.read(4))[0]
        def cstring():
            chars = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("Truncated EXR header")
                if char == b"\x00":
                    return chars.decode("ascii")
                chars += char
        attributes = {}
        while True:
            name = cstring()
            if not name:
                break
            kind = cstring()
            size = struct.unpack("<I", stream.read(4))[0]
            value = stream.read(size)
            if len(value) != size:
                raise ValueError("Truncated EXR attribute")
            attributes[name] = (kind, value)
    channels = {}
    kind, channel_data = attributes["channels"]
    if kind != "chlist":
        raise ValueError("Invalid EXR channel attribute")
    offset = 0
    while offset < len(channel_data) and channel_data[offset]:
        end = channel_data.index(b"\x00", offset)
        name = channel_data[offset:end].decode("ascii")
        pixel_type, linear, xsample, ysample = struct.unpack_from("<iB3xii", channel_data, end + 1)
        if xsample != 1 or ysample != 1:
            raise ValueError("Subsampled EXR channel unsupported")
        channels[name] = pixel_type
        offset = end + 17
    box = struct.unpack("<iiii", attributes["dataWindow"][1])
    if box[0:2] != (0, 0):
        raise ValueError("EXR data window origin must be zero")
    return {"width": box[2] + 1, "height": box[3] + 1,
            "channel_pixel_types": channels, "version_flags": version,
            "compression_code": attributes["compression"][1][0],
            "all_channels_float32": bool(channels) and all(v == 2 for v in channels.values())}


def read_exr(path, alpha=False):
    header = exr_header(path)
    if not header["all_channels_float32"]:
        raise ValueError("Native EXR storage must be FLOAT32 for every channel")
    required = {"R", "G", "B", "A"} if alpha else {"R", "G", "B"}
    if not required.issubset(header["channel_pixel_types"]):
        raise ValueError("EXR channel names do not match required RGB(A)")
    try:
        import OpenImageIO as oiio
    except ImportError:
        oiio = None
    if oiio is not None:
        reader = oiio.ImageInput.open(str(path))
        if reader is None:
            raise ValueError("OpenImageIO cannot open native EXR")
        try:
            spec = reader.spec()
            names = list(spec.channelnames)
            data = reader.read_image(format=oiio.FLOAT)
        finally:
            reader.close()
        order = [names.index(c) for c in ("RGBA" if alpha else "RGB")]
        pixels = np.asarray(data, dtype=np.float32)[:, :, order].copy()
        if pixels.shape[:2] != (header["height"], header["width"]) or not np.isfinite(pixels).all():
            raise ValueError("Native EXR OIIO dimensions/nonfinite data")
        header["decoder"] = "OpenImageIO " + oiio.VERSION_STRING + "; channel names"
        return pixels, header
    if header["compression_code"] == 4:
        raise ValueError("PIZ FLOAT32 requires OpenImageIO; installed FFmpeg PIZ32 decoder is inaccurate")
    # Keep alpha while decoding an RGBA file even when caller asks for RGB.
    # FFmpeg's RGBA-float -> RGB-float swscale conversion adds quantization.
    decode_alpha = "A" in header["channel_pixel_types"]
    fmt, count = ("gbrapf32le", 4) if decode_alpha else ("gbrpf32le", 3)
    data = command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                    "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "pipe:1"])
    expected = header["height"] * header["width"] * count * 4
    if len(data) != expected:
        raise ValueError("EXR decoded float byte count mismatch")
    planes = np.frombuffer(data, dtype="<f4").reshape(count, header["height"], header["width"])
    rgb_order = [2, 0, 1, 3] if decode_alpha else [2, 0, 1]
    pixels = np.moveaxis(planes[rgb_order], 0, -1).copy()
    if not alpha:
        pixels = pixels[:, :, :3].copy()
    if not np.isfinite(pixels).all():
        raise ValueError("EXR contains nonfinite pixels")
    return pixels, header


def write_exr(path, pixels):
    path, pixels = Path(path), np.asarray(pixels)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError("EXR requires finite HWC float32 RGB(A)")
    if path.exists():
        raise FileExistsError(path)
    h, w, channels = pixels.shape
    order = [1, 2, 0, 3] if channels == 4 else [1, 2, 0]
    data = np.ascontiguousarray(np.moveaxis(pixels[:, :, order], -1, 0), dtype="<f4").tobytes()
    fmt = "gbrapf32le" if channels == 4 else "gbrpf32le"
    path.parent.mkdir(parents=True, exist_ok=True)
    command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", fmt,
             "-s", f"{w}x{h}", "-framerate", "1", "-i", "pipe:0", "-frames:v", "1",
             "-c:v", "exr", "-format", "float", "-compression", "zip16", "-gamma", "1", str(path)], data=data)
    actual, header = read_exr(path, channels == 4)
    if not np.array_equal(actual, pixels):
        raise ValueError("EXR float32 write/readback changed values")
    return header


def matte_float(raw_rgb):
    # Match the inspected BF16 quality producer's float32 arithmetic order.
    rgb = np.asarray(raw_rgb, dtype=np.float32)
    return np.clip(np.float32(.299) * rgb[:, :, 0] + np.float32(.587) * rgb[:, :, 1]
                   + np.float32(.114) * rgb[:, :, 2], np.float32(0), np.float32(1))


def quantize16(alpha):
    # Deliberately preserve the producer's FLOAT32 multiply/add/round contract.
    return np.floor(np.asarray(alpha, dtype=np.float32) * np.float32(65535)
                    + np.float32(.5)).astype(np.uint16)


def quantize8(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype == np.uint8:
        return pixels.copy()
    if pixels.dtype == np.uint16:
        return ((pixels.astype(np.uint32) + 128) // 257).astype(np.uint8)
    return np.floor(np.clip(pixels, 0, 1).astype(np.float64) * 255 + .5).astype(np.uint8)
