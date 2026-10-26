# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Minimal PDF writer for the audit report: the plain-text report on A4 pages in Courier (one of the
14 standard PDF fonts, so nothing is embedded and no extra dependency is needed), optionally with a
JPEG thumbnail of the image on the first page.

The standard fonts only cover Western European characters (Windows-1252). Box-drawing characters
and arrows are mapped to ASCII; anything else that cannot be shown (e.g. Chinese, Japanese, Cyrillic)
becomes '?', and a note about it is added at the top of the PDF. The .txt and .json keep the full text."""

import io
import re
import zlib

PAGE_W, PAGE_H = 595.28, 841.89  # A4 in points
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 48.0, 50.0, 46.0
FONT_SIZE = 9.0
LEADING = 11.4
MAX_COLS = int((PAGE_W - 2 * MARGIN_X) / (0.6 * FONT_SIZE))  # Courier: every glyph is 0.6 em wide
_MAP = {"─": "\u2014", "═": "=", "━": "-", "│": "|", "→": "->", "←": "<-", "✓": "v", "✗": "x", "≈": "~",
        "≤": "<=", "≥": ">=", "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\t": "    "}
NON_LATIN_NOTE = ("Note: this PDF uses a standard font that cannot show some characters of the report "
                  "(e.g. Chinese, Japanese, Cyrillic or emoji). They are replaced by '?'. The .audit.txt and .audit.json "
                  "files contain the complete text.")


def _encode_line(line):
    """-> (cp1252 bytes, number of replaced characters)"""
    out, bad = bytearray(), 0
    for ch in line:
        ch = _MAP.get(ch, ch)
        try:
            b = ch.encode("cp1252")
        except UnicodeEncodeError:
            b, bad = b"?", bad + 1
        out += b
    return bytes(out), bad


def _pdf_str(b):
    return b"(" + b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)").replace(b"\r", b"") + b")"


_LEADER_RE = re.compile(r"(?<=\S) {3,}(?=\S)")
_DOTS_RE = re.compile(rb"(?<=\S) \.+ (?=\S)")


def _is_heading(line):
    s = line.rstrip()
    return bool(s) and (s.endswith("────") or s.endswith("----") or s.endswith("\u2014" * 4)) and not s.startswith(" ")


def _thumbnail_jpeg(image_rgb_uint8, max_w=360, max_h=240):
    """numpy HxWx3 uint8 -> (jpeg bytes, w, h) or None."""
    try:
        from PIL import Image

        im = Image.fromarray(image_rgb_uint8)
        im.thumbnail((max_w * 2, max_h * 2))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), im.width, im.height
    except Exception:
        return None


def write_pdf(path, text, title="Audit report", thumbnail=None, author=None):
    """Writes `text` (the plain-text report) to `path`. `thumbnail`: optional HxWx3 uint8 array.
    Returns the number of characters that had to be replaced."""
    lines_in = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        # the page holds MAX_COLS characters: long tokens (URLs, paths) are wrapped, never cut off
        while len(ln) > MAX_COLS:
            cut = ln.rfind(" ", 0, MAX_COLS)
            if cut <= len(ln) - len(ln.lstrip()) + 8:
                cut = MAX_COLS
            lines_in.append(ln[:cut])
            indent = len(ln) - len(ln.lstrip())
            ln = " " * min(indent + 4, 30) + ln[cut:].lstrip()
        lines_in.append(ln)
    # Column gaps of 3+ spaces become dot leaders ("File ....... name.png"). PDF viewers split text with wide
    # gaps into separate columns, so selecting a line would jump between label and value blocks; the leader
    # keeps each line one connected run of text. Two spaces (word gap) are left as they are.
    lines_in = [_LEADER_RE.sub(lambda m: " " + "." * (len(m.group()) - 2) + " ", ln) for ln in lines_in]
    encoded, bad = [], 0
    for ln in lines_in:
        b, n = _encode_line(ln)
        encoded.append((b, _is_heading(ln)))
        bad += n
    if bad:
        import textwrap

        note = [(_encode_line(x)[0], False) for x in textwrap.wrap(NON_LATIN_NOTE, 86)] + [(b"", False)]
        encoded = encoded[:1] + [(b"", False)] + note + encoded[1:]
    if encoded:
        encoded[0] = (encoded[0][0], True)  # report title in bold

    thumb = _thumbnail_jpeg(thumbnail) if thumbnail is not None else None
    usable = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
    per_page = int(usable // LEADING)
    thumb_box = None
    if thumb:
        jw, jh = thumb[1], thumb[2]
        scale = min(220.0 / jw, 150.0 / jh)
        dw, dh = jw * scale, jh * scale
        thumb_box = (dw, dh)

    pages, cur = [], []
    cap = per_page - (int((thumb_box[1] + 14) // LEADING) + 1 if thumb_box else 0)
    keep = 5  # a section heading starts a new page unless at least this many lines of it fit below
    for i, item in enumerate(encoded):
        heading = item[1] and i > 0
        if len(cur) >= cap or (heading and len(cur) > cap - keep):
            pages.append(cur)
            cur = []
            cap = per_page
        if not cur and not item[0].strip() and pages:
            continue  # no empty lines at the top of a page
        cur.append(item)
    pages.append(cur)

    objs = []  # index 0 -> object 1

    def add(body):
        objs.append(body)
        return len(objs)

    catalog = add(None)
    pages_id = add(None)
    # Explicit glyph widths (Courier: every glyph 600/1000 em). Without them some viewers measure the text
    # with a substitute font, and selecting / copying text is offset from what is shown.
    widths = b"/FirstChar 32 /LastChar 255 /Widths [" + b" ".join([b"600"] * 224) + b"]"
    font_r = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding " + widths + b" >>")
    font_b = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding " + widths + b" >>")
    img_id = None
    if thumb:
        jpg, jw, jh = thumb
        img_id = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                     b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (jw, jh, len(jpg))
                     + jpg + b"\nendstream")
    page_ids = []
    total = len(pages)
    for pi, pg in enumerate(pages):
        c = bytearray()
        y = PAGE_H - MARGIN_TOP
        if pi == 0 and thumb_box:
            dw, dh = thumb_box
            c += b"q %.2f 0 0 %.2f %.2f %.2f cm /Im1 Do Q\n" % (dw, dh, MARGIN_X, y - dh)
            y -= dh + 14
        c += b"BT\n"
        font = None
        cw = 0.6 * FONT_SIZE  # Courier advance per character
        ty = y - FONT_SIZE
        for b, bold in pg:
            want = b"/F2" if bold else b"/F1"
            if want != font:
                c += want + b" %.1f Tf\n" % FONT_SIZE
                font = want
            # every word is placed at its exact column, so the text layer (selection, copy) matches the page
            # even in viewers that measure with a substitute font
            pos = 0
            parts = []
            for mt in _DOTS_RE.finditer(b):
                parts.append((pos, b[pos:mt.start()], False))
                parts.append((mt.start(), mt.group(), True))
                pos = mt.end()
            parts.append((pos, b[pos:], False))
            for start, seg, dots in parts:
                if not seg:
                    continue
                if dots:
                    c += b"0.62 g "
                c += b"1 0 0 1 %.2f %.2f Tm " % (MARGIN_X + start * cw, ty) + _pdf_str(seg) + b" Tj\n"
                if dots:
                    c += b"0 g "
            ty -= LEADING
        c += b"ET\n"
        foot = _encode_line(f"{title} - page {pi + 1} of {total}")[0]
        c += b"BT /F1 7 Tf 0.45 g %.2f %.2f Td " % (MARGIN_X, MARGIN_BOTTOM - 22) + _pdf_str(foot) + b" Tj ET\n"
        data = zlib.compress(bytes(c))
        cid = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream")
        res = b"<< /Font << /F1 %d 0 R /F2 %d 0 R >>" % (font_r, font_b)
        if pi == 0 and img_id:
            res += b" /XObject << /Im1 %d 0 R >>" % img_id
        res += b" >>"
        page_ids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] /Resources %s /Contents %d 0 R >>"
                            % (pages_id, PAGE_W, PAGE_H, res, cid)))
    objs[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id
    objs[pages_id - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % p for p in page_ids), len(page_ids))
    info = b"<< /Title " + _pdf_str(_encode_line(title)[0]) + b" /Producer (Pertineo VFX for ComfyUI)"
    if author:
        info += b" /Author " + _pdf_str(_encode_line(author)[0])
    info_id = add(info + b" >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offsets:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, catalog, info_id, xref)
    with open(path, "wb") as fh:
        fh.write(out)
    return bad
