# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""PDF of the Audit Chain: overview page with the final result, the strictest triage and the chain as small images,
then one block per step (image + facts), models, outside sources, open points. Plain PDF 1.4 with the standard
fonts Helvetica / Helvetica-Bold (nothing embedded) and JPEG images – no extra dependency."""

import datetime
import io
import os
import zlib

from . import chain as CH

PAGE_W, PAGE_H = 595.28, 841.89
M_L, M_R, M_T, M_B = 42.0, 42.0, 46.0, 50.0
CONTENT_W = PAGE_W - M_L - M_R

# Helvetica widths (1/1000 em) for 32..126 – standard AFM metrics, for line wrapping
_HELV = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556,
         556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
         500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469,
         556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
         278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]

GRADE_COLORS = {"A": (0.18, 0.60, 0.32), "B": (0.45, 0.68, 0.25), "C": (0.92, 0.60, 0.10), "D": (0.55, 0.55, 0.55),
                "X": (0.80, 0.16, 0.16)}
USE_COLORS = {"yes": (0.18, 0.60, 0.32), "restricted": (0.92, 0.60, 0.10), "no": (0.80, 0.16, 0.16)}
GREY = (0.45, 0.45, 0.45)
LIGHT = (0.93, 0.93, 0.93)
DARK = (0.12, 0.12, 0.12)
_MAP = {"\u2192": "->", "\u2190": "<-", "\u2248": "~", "\u2009": " ", "\u202f": " ", "\u00a0": " ", "\u2500": "-"}


def _enc(text):
    out = bytearray()
    for ch in str(text):
        ch = _MAP.get(ch, ch)
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            out += b"?"
    return bytes(out).replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def text_width(text, size, bold=False):
    w = 0
    for ch in str(text):
        o = ord(ch)
        w += _HELV[o - 32] if 32 <= o <= 126 else 556
    return w * size / 1000.0 * (1.07 if bold else 1.0)


def wrap(text, width, size, bold=False):
    """Splits text into lines that fit `width` points (long words are cut)."""
    lines = []
    for para in str(text).split("\n"):
        line = ""
        for word in para.split(" "):
            cand = word if not line else line + " " + word
            if text_width(cand, size, bold) <= width:
                line = cand
                continue
            if line:
                lines.append(line)
            while text_width(word, size, bold) > width and len(word) > 1:
                cut = len(word)
                while cut > 1 and text_width(word[:cut], size, bold) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            line = word
        lines.append(line)
    return lines


class _Doc:
    """Minimal PDF builder: pages with a content stream, two fonts, JPEG images."""

    def __init__(self):
        self.pages, self.images = [], []
        self.ops = None

    def new_page(self):
        self.ops = []
        self.pages.append(self.ops)

    # drawing (y = from the top of the page)
    def text(self, x, y, s, size=9, bold=False, color=DARK):
        self.ops.append(b"BT /%s %.2f Tf %.3f %.3f %.3f rg %.2f %.2f Td (%s) Tj ET" % (
            b"F2" if bold else b"F1", size, *color, x, PAGE_H - y, _enc(s)))

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.6):
        cmd = b""
        if fill:
            cmd += b"%.3f %.3f %.3f rg " % fill
        if stroke:
            cmd += b"%.3f %.3f %.3f RG %.2f w " % (*stroke, lw)
        op = b"B" if fill and stroke else (b"f" if fill else b"S")
        self.ops.append(cmd + b"%.2f %.2f %.2f %.2f re %s" % (x, PAGE_H - y - h, w, h, op))

    def line(self, x1, y1, x2, y2, color=GREY, lw=0.6):
        self.ops.append(b"%.3f %.3f %.3f RG %.2f w %.2f %.2f m %.2f %.2f l S" % (
            *color, lw, x1, PAGE_H - y1, x2, PAGE_H - y2))

    def arrow(self, x1, y, x2, color=GREY):
        self.line(x1, y, x2 - 3, y, color, 0.9)
        self.ops.append(b"%.3f %.3f %.3f rg %.2f %.2f m %.2f %.2f l %.2f %.2f l f" % (
            *color, x2, PAGE_H - y, x2 - 5, PAGE_H - y + 3, x2 - 5, PAGE_H - y - 3))

    def dot(self, x, y, r, color):
        k = 0.5523 * r
        cy = PAGE_H - y
        self.ops.append(b"%.3f %.3f %.3f rg %.2f %.2f m %.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c "
                        b"%.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c f" % (
                            *color, x + r, cy, x + r, cy + k, x + k, cy + r, x, cy + r, x - k, cy + r, x - r, cy + k,
                            x - r, cy, x - r, cy - k, x - k, cy - r, x, cy - r, x + k, cy - r, x + r, cy - k, x + r, cy))

    def image(self, pil, x, y, w, h):
        """Draws a PIL image fitted into the box (aspect kept, centred); returns the drawn rect."""
        iw, ih = pil.size
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        dx, dy = x + (w - dw) / 2, y + (h - dh) / 2
        buf = io.BytesIO()
        im = pil.convert("RGB")
        im.thumbnail((int(dw * 3), int(dh * 3)))
        im.save(buf, "JPEG", quality=85)
        self.images.append((buf.getvalue(), im.width, im.height))
        name = b"Im%d" % len(self.images)
        self.ops.append(b"q %.2f 0 0 %.2f %.2f %.2f cm /%s Do Q" % (dw, dh, dx, PAGE_H - dy - dh, name))
        return dx, dy, dw, dh

    def save(self, path, title, author):
        objs = []

        def add(body):
            objs.append(body)
            return len(objs)

        cat = add(None)
        pages = add(None)
        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        img_ids = []
        for data, w, h in self.images:
            img_ids.append(add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                               b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (w, h, len(data))
                               + data + b"\nendstream"))
        xobj = b" ".join(b"/Im%d %d 0 R" % (i + 1, oid) for i, oid in enumerate(img_ids))
        res = b"<< /Font << /F1 %d 0 R /F2 %d 0 R >> /XObject << %s >> >>" % (f1, f2, xobj)
        kids = []
        for ops in self.pages:
            raw = zlib.compress(b"\n".join(ops))
            cs = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(raw) + raw + b"\nendstream")
            kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] /Resources %s /Contents %d 0 R >>"
                            % (pages, PAGE_W, PAGE_H, res, cs)))
        objs[cat - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages
        objs[pages - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
            b" ".join(b"%d 0 R" % k for k in kids), len(kids))
        info = add(b"<< /Title (%s) /Author (%s) /Producer (Pertineo VFX) /CreationDate (D:%s) >>" % (
            _enc(title), _enc(author or ""), datetime.datetime.now().strftime("%Y%m%d%H%M%S").encode()))
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offs = []
        for i, body in enumerate(objs, 1):
            offs.append(len(out))
            out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
        out += b"".join(b"%010d 00000 n \n" % o for o in offs)
        out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
            len(objs) + 1, cat, info, xref)
        with open(path, "wb") as f:
            f.write(out)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
class _Writer:
    def __init__(self, doc, footer):
        self.doc, self.footer, self.y = doc, footer, 0.0
        self.new_page()

    def new_page(self):
        self.doc.new_page()
        self.y = M_T

    def need(self, h):
        if self.y + h > PAGE_H - M_B:
            self.new_page()

    def heading(self, s):
        self.need(90)  # a section heading never stands alone at the bottom of a page
        self.y += 10
        self.doc.text(M_L, self.y + 10, s.upper(), 10, True)
        self.doc.line(M_L, self.y + 15, PAGE_W - M_R, self.y + 15, (0.75, 0.75, 0.75), 0.8)
        self.y += 24

    def pairs(self, x, y, w, rows, size=8.3, label_w=62):
        """label / value rows, value wrapped; rows: (label, value, color or None). Continues on a new page when the
        page is full. Returns the new y."""
        lh = size * 1.32
        for label, value, color in rows:
            lines = wrap(value, w - label_w, size)
            for i, ln in enumerate(lines):
                if y + size + 2 > PAGE_H - M_B:
                    self.new_page()
                    y = self.y
                if i == 0:
                    self.doc.text(x, y + size, label, size, False, GREY)
                    # light dotted leader: label and value stay one line of text for PDF viewers (a wide gap
                    # makes them read all labels first, then all values – selecting / copying jumps)
                    lw_ = text_width(label, size) if label else 0.0
                    dot_w = text_width(". ", size)
                    start = x + lw_ + (2 if label else 0)
                    n = int((x + label_w - 3 - start) / dot_w)
                    if n > 0:
                        self.doc.text(start, y + size, ". " * n, size, False, (0.85, 0.85, 0.85))
                self.doc.text(x + label_w, y + size, ln, size, False, color or DARK)
                y += lh
            y += 1.5
        return y

    def finish(self):
        n = len(self.doc.pages)
        for i, ops in enumerate(self.doc.pages, 1):
            self.doc.ops = ops
            self.doc.line(M_L, PAGE_H - M_B + 14, PAGE_W - M_R, PAGE_H - M_B + 14, LIGHT, 0.6)
            self.doc.text(M_L, PAGE_H - M_B + 26, self.footer, 7, False, GREY)
            p = f"{i} / {n}"
            self.doc.text(PAGE_W - M_R - text_width(p, 7), PAGE_H - M_B + 26, p, 7, False, GREY)


NO_GRADE = (0.80, 0.80, 0.80)


def _badge(doc, x, y, grade, size=22):
    col = GRADE_COLORS.get(grade, NO_GRADE)
    doc.rect(x, y, size, size, fill=col)
    doc.text(x + (size - text_width(grade or "-", size * 0.62, True)) / 2, y + size * 0.73, grade or "-",
             size * 0.62, True, (1, 1, 1))


def _placeholder(doc, x, y, w, h, label="no image"):
    doc.rect(x, y, w, h, fill=LIGHT)
    doc.text(x + (w - text_width(label, 7)) / 2, y + h / 2 + 2, label, 7, False, GREY)


def _thumb(doc, pil, x, y, w, h, label="no image"):
    if pil is None:
        _placeholder(doc, x, y, w, h, label)
        return
    try:
        doc.image(pil, x, y, w, h)
    except Exception:
        _placeholder(doc, x, y, w, h, label)


def _use(m):
    cu = CH._val(m.get("commercial_use"))
    return cu, USE_COLORS.get(cu.lower(), GREY)


RAW_EXTS = (".rw2", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".raf", ".pef", ".srw", ".rwl",
            ".3fr", ".iiq", ".x3f", ".raw")


def _camera(doc, cx, cy, w):
    """A simple camera symbol (body, viewfinder bump, lens) centred on cx / cy, w wide."""
    h = w * 0.62
    dark, light = (0.35, 0.35, 0.35), (0.96, 0.96, 0.96)
    doc.rect(cx - w * 0.22, cy - h / 2 - h * 0.18, w * 0.30, h * 0.2, fill=dark)
    doc.rect(cx - w / 2, cy - h / 2, w, h, fill=dark)
    doc.dot(cx, cy + h * 0.04, h * 0.34, light)
    doc.dot(cx, cy + h * 0.04, h * 0.22, dark)
    doc.rect(cx + w * 0.28, cy - h * 0.36, w * 0.12, h * 0.12, fill=light)


def _arrow_down(doc, x, y, color=GREY):
    doc.ops.append(b"%.3f %.3f %.3f rg %.2f %.2f m %.2f %.2f l %.2f %.2f l f" % (
        *color, x, PAGE_H - y, x - 3, PAGE_H - y + 5, x + 3, PAGE_H - y + 5))


def _draw_graph(doc, w, steps, thumbs, assess=True):
    nodes, edges = CH.graph_layout(steps)
    ncols = max(n["col"] for n in nodes.values()) + 1
    nlay = max(n["layer"] for n in nodes.values()) + 1
    gx, label_h, gy = 14.0, 22.0, 16.0
    room = PAGE_H - M_B - w.y - 6
    full = PAGE_H - M_B - M_T - 30  # a new page starts with the heading again

    def fit(h_avail):
        """box width that fits the width and the given height."""
        by_w = (CONTENT_W - gx * (ncols - 1)) / ncols
        by_h = ((h_avail + gy) / nlay - label_h - gy) / 0.62
        return min(118.0, by_w, by_h)

    bw = fit(room - 34)  # minus the heading
    if bw < 70 and fit(full) > bw:  # too small here: the graph gets a page of its own
        w.new_page()
    w.heading("The chain")
    bw = fit(PAGE_H - M_B - w.y - 6)
    bw = max(bw, 30.0)
    bh = bw * 0.62
    row = bh + label_h + gy
    x0 = M_L + (CONTENT_W - (ncols * bw + (ncols - 1) * gx)) / 2
    top = w.y
    pos = {k: (x0 + n["col"] * (bw + gx), top + n["layer"] * row) for k, n in nodes.items()}
    # edges first, boxes on top
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        sx, sy = xa + bw / 2, ya + bh + label_h - 1
        ex, ey = xb + bw / 2, yb - 1
        col = (0.55, 0.55, 0.55)
        la, lb, ca = nodes[a]["layer"], nodes[b]["layer"], nodes[a]["col"]
        # a line that would run through other boxes of its column takes the lane in the gap to the left
        blocked = any(n["col"] == ca and la < n["layer"] < lb for k2, n in nodes.items() if k2 not in (a, b))
        my = ey - gy / 2 + 2
        if blocked:
            lane = xa - gx / 2
            y1 = sy + 3
            doc.line(sx, sy, sx, y1, col, 0.9)
            doc.line(sx, y1, lane, y1, col, 0.9)
            doc.line(lane, y1, lane, my, col, 0.9)
            doc.line(lane, my, ex, my, col, 0.9)
            doc.line(ex, my, ex, ey - 4, col, 0.9)
        elif abs(sx - ex) < 0.5:
            doc.line(sx, sy, ex, ey - 4, col, 0.9)
        else:
            doc.line(sx, sy, sx, my, col, 0.9)
            doc.line(sx, my, ex, my, col, 0.9)
            doc.line(ex, my, ex, ey - 4, col, 0.9)
        _arrow_down(doc, ex, ey, col)
    for k, n in nodes.items():
        x, y = pos[k]
        if n["kind"] == "step":
            s = n["step"]
            _thumb(doc, thumbs.get(s["id"]), x, y, bw, bh)
            if thumbs.get(s["id"]) is None and s["report"].get("kind") == CH.PHOTO:
                doc.rect(x, y, bw, bh, fill=(0.96, 0.96, 0.96))
                _camera(doc, x + bw / 2, y + bh / 2, min(bw * 0.42, bh * 0.6))
            g = s["report"]["triage"].get("grade") if assess else None
            doc.dot(x + 4, y + bh + 8, 3.2, GRADE_COLORS.get(g, GREY) if assess else LIGHT)
            ext = CH.is_external(s["report"])
            doc.text(x + 10, y + bh + 11, f"{s['id']}  " + (("photo" if s["report"].get("kind") == CH.PHOTO else
                                                             "external") if ext else (g or ("-" if assess else ""))),
                     7.5, True)
            stp = (s["report"]["manual"].get("step") or "").strip() or \
                (s["report"]["file"].get("name") or "")
            ln = wrap(stp, bw, 6.5)
            doc.text(x, y + bh + 20, ln[0] + (" …" if len(ln) > 1 else ""), 6.5, False, GREY)
        else:
            amber = GRADE_COLORS["C"] if n["kind"] != "declared" else GREY
            fill = (0.99, 0.95, 0.88) if n["kind"] != "declared" else (0.96, 0.96, 0.96)
            doc.rect(x, y, bw, bh, fill=fill, stroke=amber, lw=0.8)
            head = {"gap": "GAP – not verified", "outside": "outside source"}.get(n["kind"], n["hint"])
            doc.text(x + 5, y + 11, head, 6.5, True, amber)
            fn = CH.asset_name(n["asset"])
            pic = thumbs.get(k)
            if pic is not None:  # the file itself was found: show it, framed in the colour of its status
                _thumb(doc, pic, x, y, bw, bh)
                doc.rect(x, y, bw, bh, stroke=amber, lw=1.2)
                doc.rect(x, y, bw, 12, fill=fill)
                doc.text(x + 4, y + 9, head, 6.3, True, amber)
                short = fn if text_width(fn, 6.3) <= bw else "…" + fn[-max(4, int((bw - 8) / 3.3)):]
                doc.text(x, y + bh + 11, short, 6.3, False, GREY)
                doc.text(x, y + bh + 19, "declared" if n["kind"] == "declared" else "not audited", 6.3, False, GREY)
                continue
            if fn.lower().endswith(RAW_EXTS):  # camera raw: a camera symbol instead of the picture
                _camera(doc, x + bw / 2, y + bh / 2 + 3, min(bw * 0.42, bh * 0.55))
                short = fn if text_width(fn, 6.3) <= bw - 10 else "…" + fn[-max(4, int((bw - 14) / 3.3)):]
                doc.text(x + 5, y + bh - 5, short, 6.3, False, DARK)
                doc.text(x, y + bh + 11, "declared", 6.5, False, GREY)
                continue
            lines = wrap(fn, bw - 10, 6.3)
            fit_n = max(1, int((bh - 17) // 8))
            if len(lines) > fit_n:
                lines = lines[:fit_n - 1] + ["…" + fn[-max(4, int((bw - 14) / 3.3)):]] if fit_n > 1 else \
                    ["…" + fn[-max(4, int((bw - 14) / 3.3)):]]
            for i, ln in enumerate(lines):
                doc.text(x + 5, y + 21 + i * 8, ln, 6.3, False, DARK)
            doc.text(x, y + bh + 11, "declared" if n["kind"] == "declared" else "not audited", 6.5, False, GREY)
    w.y = top + nlay * row - gy + 10


# --------------------------------------------------------------------------
# Audit by topic
# --------------------------------------------------------------------------
SEV_COLORS = {"critical": GRADE_COLORS["X"], "high": GRADE_COLORS["C"], "medium": (0.70, 0.55, 0.10)}
STATUS_COLORS = {"yes": GRADE_COLORS["A"], "no": GRADE_COLORS["X"], "unknown": GRADE_COLORS["C"]}


def _sha(h, label=True):
    """A full SHA256 in two halves of 32 characters – it never gets cut or broken at a random place."""
    h = str(h or "").strip()
    if len(h) != 64:
        return ("sha256 " if label and h else "") + (h or CH.UNK)
    return ("sha256 " if label else "") + h[:32] + "\n" + ("       " if label else "") + h[32:]


def _sub(w, doc, s, keep=70):
    """A sub heading inside a section – never alone at the bottom of a page: keep = room it needs below it."""
    w.need(30 + keep)
    w.y += 4
    doc.text(M_L, w.y + 9, s, 8.6, True, DARK)
    w.y += 15


def _para(w, doc, text, size=8.3, color=DARK, indent=0):
    for ln in wrap(text, CONTENT_W - indent, size):
        w.need(size * 1.4)
        doc.text(M_L + indent, w.y + size, ln, size, False, color)
        w.y += size * 1.38
    w.y += 3


def _table(w, doc, widths, header, rows, size=7.6):
    """A table over the content width. widths: relative column widths; rows: (cells, colour or None) – colour
    for the whole row, or a list of colours per cell. Long cells wrap; the header repeats after a page break."""
    total = float(sum(widths))
    ws = [CONTENT_W * x / total for x in widths]
    xs = [M_L + sum(ws[:i]) for i in range(len(ws))]
    lh = size * 1.32

    def head():
        w.need(16)
        for x, h in zip(xs, header):
            doc.text(x, w.y + 8, h, 7, True, GREY)
        w.y += 11
        doc.line(M_L, w.y, PAGE_W - M_R, w.y, (0.80, 0.80, 0.80), 0.6)
        w.y += 3

    if rows:  # the header never stands alone: it needs room for the first row too
        first = max(len(wrap(str(c if c is not None else ""), cw - 6, size)) for c, cw in zip(rows[0][0], ws))
        w.need(16 + min(first, 6) * lh + 6)
    head()
    for cells, color in rows:
        cols = [wrap(str(c if c is not None else ""), cw - 6, size) for c, cw in zip(cells, ws)]
        n = max(len(c) for c in cols)
        h = n * lh + 4
        if w.y + min(h, 4 * lh) > PAGE_H - M_B:
            w.new_page()
            head()
        y = w.y
        for k in range(n):
            if y + lh > PAGE_H - M_B:
                w.new_page()
                head()
                y = w.y
            for ci, lines in enumerate(cols):
                if k < len(lines):
                    col = color[ci] if isinstance(color, list) else color
                    doc.text(xs[ci], y + size, lines[k], size, False, col or DARK)
            y += lh
        w.y = y + 4
        doc.line(M_L, w.y - 2, PAGE_W - M_R, w.y - 2, LIGHT, 0.4)


def _topic_summary(doc, w, steps, summary, assess):
    """Cover page addition: who is responsible, and the result in plain words."""
    ppl = CH.people(steps)
    if ppl:
        w.heading("Responsibilities")
        w.y = w.pairs(M_L, w.y, CONTENT_W, [(k, v, None) for k, v in ppl], 8.3, 120) + 4
    if not assess:
        return
    w.heading("Summary")
    wg = summary["worst_grade"]
    if wg:
        worst = [s for s in steps if s["report"]["triage"].get("grade") == wg]
        label = worst[0]["report"]["triage"].get("label", "")
        _para(w, doc, f"Overall grade {wg} – {label}. The strictest step decides for the whole result: "
                      f"{CH.id_range([s['id'] for s in worst])}.", 9, GRADE_COLORS.get(wg, DARK))
        crit = [f for f in CH.merged_findings(steps) if f.get("severity") in ("critical", "high")]
        for f in crit:
            _para(w, doc, f"• [{f.get('severity')}] {f.get('text')} ({CH.id_range(f['steps'])})", 8.3,
                  SEV_COLORS.get(f.get("severity"), DARK), 8)
    ap = CH.merged_approvals(steps)
    opens = [n for n, e in ap.items() if e["open"]]
    if opens:
        _para(w, doc, "Open approvals: " + "; ".join(f"{n} ({CH.id_range(ap[n]['open'])})" for n in opens), 8.3,
              GRADE_COLORS["C"])
    miss = [c for c in CH.checklist(steps) if c[2] == "missing"]
    _para(w, doc, f"Disclosure checklist: {len(CH.CHECK_ITEMS) - len(miss)} of {len(CH.CHECK_ITEMS)} items given or "
                  "not applicable" + (" – incomplete: " + ", ".join(c[0] for c in miss) if miss else "."), 8.3, GREY)


def _topic_sections(doc, w, steps, summary, full_prompts, assess, licenses):
    """The audit gathered per subject over the whole chain."""
    sid = {s["id"]: s for s in steps}
    # short list of the steps (the graph shows them as pictures)
    w.heading("Steps")
    _table(w, doc, [8, 34, 38, 20], ["Step", "Name", "File", "Kind"],
           [([s["id"], (s["report"]["manual"].get("step") or "").strip() or "–",
              s["report"]["file"].get("name") or "",
              "camera original" if s["report"].get("kind") == CH.PHOTO else
              "declared (outside ComfyUI)" if CH.is_external(s["report"]) else "ComfyUI run"], None) for s in steps])

    if assess:
        # ---- production context: one field per line, identical contexts merged
        groups = CH.context_groups(steps)
        if groups:
            w.heading("Production context")
            known = {k for k, _ in CH.CONTEXT_LABELS} | {k for k, _ in CH.PEOPLE_LABELS} | {"context_notes"}
            for ids, ctx in groups:
                _sub(w, doc, f"{CH.id_range(ids)}")
                rows = [(label, ctx[k], None) for k, label in CH.CONTEXT_LABELS if k in ctx]
                rows += [(k.replace("_", " "), v, None) for k, v in ctx.items() if k not in known]
                w.y = w.pairs(M_L + 8, w.y, CONTENT_W - 8, rows, 8.3, 150) + 2
                if ctx.get("context_notes"):
                    doc.text(M_L + 8, w.y + 8, "Notes", 8.3, False, GREY)
                    w.y += 12
                    _para(w, doc, ctx["context_notes"], 8.3, DARK, 8)
        # ---- assessment
        w.heading("Assessment")
        ap = CH.merged_approvals(steps)
        if ap:
            _sub(w, doc, "Approvals needed")
            rows = []
            for need, e in ap.items():
                st = []
                for by, ids in e["covered"].items():
                    st.append(f"covered ({by}): {CH.id_range(ids)}")
                if e["open"]:
                    st.append(f"OPEN: {CH.id_range(e['open'])}")
                rows.append(([need, " · ".join(st)], GRADE_COLORS["C"] if e["open"] else None))
            _table(w, doc, [45, 55], ["Approval", "Status"], rows)
        fs = CH.merged_findings(steps)
        if fs:
            _sub(w, doc, "Findings")
            _table(w, doc, [10, 52, 18, 10, 10], ["Severity", "Finding", "Department", "Evidence", "Steps"],
                   [([f.get("severity"), f.get("text"), CH.ROUTES.get(str(f.get("route")), f.get("route") or ""),
                      f.get("evidence") or "", CH.id_range(f["steps"])],
                     SEV_COLORS.get(str(f.get("severity")))) for f in fs])
        ps, cols, cell = CH.principle_matrix(steps)
        if ps:
            _sub(w, doc, "Guiding principles")
            widths = [34] + [max(5, 66 / max(1, len(cols)))] * len(cols)
            rows = []
            for pid, text in ps:
                vals = [cell.get((pid, c), ("", ""))[0] for c in cols]
                rows.append(([f"{pid} {text}"] + [{"unknown": "?"}.get(v, v.upper()) if v else "–" for v in vals],
                             [None] + [STATUS_COLORS.get(v) for v in vals]))
            _table(w, doc, widths, ["Principle"] + cols, rows, 7.2)
            _para(w, doc, "YES / NO as answered in the step's audit · ? = UNKNOWN / NOT VERIFIED · – = not assessed",
                  7, GREY)
            # the reasons, once per principle and answer
            for pid, text in ps:
                seen = {}
                for c in cols:
                    st, basis = cell.get((pid, c), ("", ""))
                    if st:
                        seen.setdefault((st, basis), []).append(c)
                for (st, basis), ids in seen.items():
                    _para(w, doc, f"{pid} {st.upper()} ({CH.id_range(ids)}): {basis}", 7.4, GREY, 4)

    # ---- models and licenses, once for the whole chain
    if summary["models"]:
        w.heading("Models & licenses")
        rows = []
        for m in summary["models"]:
            role = CH.ROLE_LABELS.get(m.get("role"), m.get("role") or "model")
            name = m.get("name") or os.path.basename(str(m.get("file") or "?"))
            extra = []
            if m.get("version"):
                extra.append(f"version {m['version']}")
            if m.get("base_model"):
                extra.append(f"base {m['base_model']}")
            st = m.get("strength_model", m.get("strength"))
            if st not in (None, ""):
                extra.append(f"strength {st}")
            cu, col = _use(m)
            rows.append(([f"{name}\n{role}" + ("\n" + " · ".join(extra) if extra else ""),
                          f"{CH._val(m.get('file'))}\n{_sha(m.get('sha256'))}\n{CH._val(m.get('source'))}",
                          CH._val(m.get("license")) + ("\n" + m["license_notes"] if m.get("license_notes") else ""),
                          cu, CH.id_range(m["steps"])], [None, GREY, None, col, None]))
        _table(w, doc, [20, 40, 24, 8, 8], ["Model", "File · SHA256 · source", "License", "Commercial", "Steps"],
               rows, 7.2)

    # ---- inputs and provenance
    w.heading("Inputs & provenance")
    rows = []
    for s in steps:
        for it in s["inputs"]:
            a = it["asset"]
            fn = CH.asset_name(a)
            if it["from"] is not None:
                k = it["from"]["report"].get("kind")
                org = (f"{it['from']['id']} – " + ("camera original" if k == CH.PHOTO else
                                                   "declared step" if CH.is_external(it["from"]["report"]) else
                                                   "ComfyUI step"))
                col = None
            elif it.get("declared"):
                org, col = f"{it['declared']} (documented by the declaration of {s['id']})", GREY
            elif it["hint"]:
                org, col = f"GAP – {it['hint']}", GRADE_COLORS["C"]
            else:
                org, col = "unknown origin – document origin and rights", GRADE_COLORS["C"]
            size = f"{a.get('width')}x{a.get('height')}" if a.get("width") else ""
            add = f" · added later {str(a['added'])[:10]}" if a.get("added") else ""
            rows.append(([s["id"], f"{fn}\n{a.get('role') or ''} {size}".strip() + add, org,
                          _sha(a.get("sha256"), False)], col))
    _table(w, doc, [6, 30, 33, 31], ["Used in", "Input", "Origin", "SHA256"], rows, 7.2)
    decl = [s for s in steps if CH.is_external(s["report"])]
    if decl:
        _sub(w, doc, "Declared steps and camera originals")
        for s in decl:
            r = s["report"]
            rows = [(lab, val, None) for lab, val in CH.step_details(r, full=True)
                    if lab not in ("Resolution",)]
            doc.text(M_L, w.y + 9, f"{s['id']}  {(r['manual'].get('step') or '').strip()}  –  "
                                   f"{r['file'].get('name') or ''}", 8.3, True)
            w.y += 13
            w.y = w.pairs(M_L + 8, w.y, CONTENT_W - 8, rows, 8, 90) + 4

    # ---- environment and software
    env, pk, api = CH.merged_environment(steps)
    w.heading("Environment & software")
    if env:
        _table(w, doc, [80, 20], ["Environment", "Steps"], [([t, CH.id_range(ids)], None) for t, ids in env])
    if pk:
        _sub(w, doc, "Node packages")
        _table(w, doc, [60, 40], ["Package", "Steps"], [([t, CH.id_range(ids)], None) for t, ids in pk])
    _sub(w, doc, "APIs / external services")
    if api:
        _table(w, doc, [60, 40], ["Service", "Steps"], [([t, CH.id_range(ids)], None) for t, ids in api])
    else:
        _para(w, doc, "none – every ComfyUI step ran locally", 8.3)

    # ---- generation settings
    rows = []
    for s in steps:
        r = s["report"]
        if CH.is_external(r):
            continue
        sm = (r.get("seed_mode") or {}).get("mode")
        res = CH.step_details(r, full=False)
        resolution = next((v for lab, v in res if lab in ("Resolution", "Audio")), "")
        for x in r["sampling"]:
            p = x["params"]
            cfg = p.get("cfg", p.get("guidance"))
            extra = ", ".join(f"{k} {v}" for k, v in p.items()
                              if k not in ("seed", "noise_seed", "sampler", "scheduler", "steps", "cfg", "guidance",
                                           "denoise"))
            settings = (f"seed {CH._val(p.get('seed', p.get('noise_seed')))}" + (f" ({sm})" if sm else "")
                        + f"\nsampler {CH._val(p.get('sampler'))} · scheduler {CH._val(p.get('scheduler'))}"
                        + f"\nsteps {CH._val(p.get('steps'))} · {'cfg' if 'cfg' in p else 'guidance'} {CH._val(cfg)}"
                        + f" · denoise {CH._val(p.get('denoise'))}" + (f"\n{extra}" if extra else ""))
            rows.append(([s["id"], x["node"], settings, resolution], None))
    if rows:
        w.heading("Generation settings")
        _table(w, doc, [6, 16, 40, 38], ["Step", "Sampler node", "Settings", "Resolution"], rows, 7.4)

    # ---- prompts
    if any(s["report"]["sampling"] for s in steps):
        w.heading("Prompts")
        lim = 10 ** 9 if full_prompts else 400
        for s in steps:
            r = s["report"]
            if not r["sampling"]:
                continue
            _sub(w, doc, f"{s['id']}  {(r['manual'].get('step') or '').strip()}")
            for x in r["sampling"]:
                pos = [t for t in x["prompts"].get("positive", []) if t.strip()]
                neg = [t for t in x["prompts"].get("negative", []) if t.strip()]
                doc.text(M_L + 8, w.y + 8, f"{x['node']} – prompt", 7.6, True, GREY)
                w.y += 12
                _para(w, doc, CH._short(" | ".join(pos), lim) if pos else CH.UNK, 8.3, DARK, 8)
                if "negative" in x["prompts"]:
                    doc.text(M_L + 8, w.y + 8, "negative", 7.6, True, GREY)
                    w.y += 12
                    _para(w, doc, CH._short(" | ".join(neg), lim) if neg else "(empty)", 8.3, DARK, 8)
            for wr in r["prompt_writers"]:
                _para(w, doc, f"Prompt writer: {wr['node']} -> {', '.join(wr['feeds']) or '?'} (written by an LLM / "
                              "VLM in the run – recorded above)", 7.6, GREY, 8)
            doc.text(M_L + 8, w.y + 8, "external AI (outside this run)", 7.6, True, GREY)
            w.y += 12
            _para(w, doc, CH.ai_text(r), 8.3, DARK, 8)
            w.y += 4

    # ---- outputs
    w.heading("Outputs")
    rows = []
    for s in steps:
        r, f = s["report"], s["report"]["file"]
        size = f"{f.get('width')}x{f.get('height')}" if f.get("width") else ""
        aud = CH.audio_out(r)
        if aud and not f.get("width"):
            size = CH.audio_text(aud)
        sha = (r.get("outputs") or [""])[0] if r.get("outputs") else ""
        rng = CH.frame_range_text(r)
        tim = CH.timing_text(r).split(" (")[0]
        rows.append(([s["id"], f.get("name") or "", f"{size}\n{f.get('format') or ''}".strip() +
                      (f"\n{rng}" if rng else "") + (f"\n{tim}" if tim else ""),
                      str(f.get("saved") or r.get("generated") or "")[:16].replace("T", " "),
                      _sha(sha, False), CH.repro_text(r) +
                      (f"\n{len(s.get('alternatives') or [])} variant(s) not used" if s.get("alternatives") else "")],
                     None))
    _table(w, doc, [5, 23, 14, 11, 30, 17], ["Step", "File", "Size / format", "Saved", "SHA256", "Reproducible"],
           rows, 7.2)


def write_chain_pdf(path, steps, summary, thumbs, version="", title=None, full_prompts=False, mode="audit",
                    assess=True, licenses=True, layout="by topic"):
    """mode audit = the complete record: overview, graph, per step the full disclosure (+ the assessment with
    assess); mode breakdown = the overview: graph and per step only what / inputs / main models / grade.
    full_prompts: complete prompts in the audit. assess: triage, production context, approvals, findings,
    principles and the grade legend – off = documentation without any assessment.
    steps / summary from chain.py; thumbs: step id -> PIL image (or None)."""
    breakdown = mode in ("breakdown", "overview")
    topic = mode == "audit" and layout == "by topic"
    if mode == "overview":  # neutral listing: no assessment and no licenses at all
        assess, licenses = False, False
    final = steps[-1]["report"]
    ctx, man = final.get("context") or {}, final.get("manual") or {}
    name = final["file"].get("name") or os.path.basename(final["path"])
    footer = (f"Audit chain · {name} · Pertineo VFX {version} · " +
              ("indicative first assessment, not an approval and not legal advice" if assess else
               "documentation without assessment"))
    doc = _Doc()
    w = _Writer(doc, footer)

    # ---- page 1: header -------------------------------------------------
    kicker = "OVERVIEW" if mode == "overview" else "BREAKDOWN" if breakdown else (
        ("AUDIT BY TOPIC" if topic else "AUDIT BY STEP") + (" · FULL PROMPTS" if full_prompts else " · SHORTENED PROMPTS"))
    kicker += "" if assess or mode == "overview" else " · WITHOUT ASSESSMENT"
    doc.text(M_L, w.y + 8, f"AUDIT CHAIN · {kicker}", 8, True, GREY)
    # the delivered file as given to the node (with extension, e.g. .mp4) – the report name has none for videos
    final_file = os.path.basename(str(summary.get("final") or "")) or name
    head = (title or "").strip() or final_file
    for i, ln in enumerate(wrap(head, CONTENT_W, 16, True)[:2]):
        doc.text(M_L, w.y + 28 + i * 19, ln, 16, True)
    w.y += 30 + 19 * min(2, len(wrap(head, CONTENT_W, 16, True)))
    sub = " · ".join(x for x in ((final_file if (title or "").strip() else None), ctx.get("production_title"),
                                 ctx.get("project_episode"), man.get("author"),
                                 datetime.date.today().isoformat()) if x)
    doc.text(M_L, w.y + 4, sub, 9, False, GREY)
    w.y += 18

    # final image + summary
    top = w.y
    img_w, img_h = 230, 170
    _thumb(doc, thumbs.get(steps[-1]["id"]), M_L, top, img_w, img_h)
    x = M_L + img_w + 18
    wg = summary["worst_grade"] if assess else None
    if assess:
        _badge(doc, x, top, wg or "-", 34)
    label = next((s["report"]["triage"].get("label") for s in steps if s["report"]["triage"].get("grade") == wg), "")
    worst_steps = [s["id"] for s in steps if s["report"]["triage"].get("grade") == wg]
    if assess:
        doc.text(x + 44, top + 14, label or "no triage", 11, True, GRADE_COLORS.get(wg, GREY))
        doc.text(x + 44, top + 28, ("strictest step: " + ", ".join(worst_steps)) if wg else
                 "no Production Context in any step", 8.5, False, GREY)
    else:
        doc.text(x, top + 14, "Documentation", 11, True)
        doc.text(x, top + 28, "without assessment (no triage)", 8.5, False, GREY)
    lic = summary["licenses"]
    rows = [("Steps", f"{summary['steps']} (S1 earliest – S{summary['steps']} final)", None),
            ("Models", f"{len(summary['models'])} · commercial use yes {lic['yes']} · restricted {lic['restricted']}"
                       f" · no {lic['no']} · unknown {lic['unknown']}" if licenses else f"{len(summary['models'])}",
             USE_COLORS["no"] if lic["no"] and licenses else None),
            ("Undocumented", f"{len(summary['external_inputs'])} input(s) of unknown origin – not made in an "
                             "audited or declared step" if summary["external_inputs"] else
             "none – every input comes from an audited ComfyUI step or a declared step", None),
            ("Gaps", f"{len(summary['gaps'])} input(s) changed outside ComfyUI without a declaration (link not "
                     "verified)" if summary["gaps"] else "none – every link verified by SHA256", None),
            ("Variants", f"{summary['alternatives']} other output(s) of these steps in total (alternatives, "
                         "iterations) – not used for this result" if summary["alternatives"] else "none", None)]
    if summary.get("external_steps"):
        rows.insert(1, ("External", ", ".join(summary["external_steps"]) + " – work outside ComfyUI, declared", None))
    opens = sorted({o["need"] for o in summary["open_points"]}) if assess else []
    if opens:
        rows.append(("Open", "; ".join(opens), GRADE_COLORS["C"]))
    y2 = w.pairs(x, top + 46, PAGE_W - M_R - x, rows, 8.5, 72)
    w.y = max(top + img_h, y2) + 14

    if topic:
        _topic_summary(doc, w, steps, summary, assess)
    # chain overview: a node graph, sources at the top, the final step at the bottom
    _draw_graph(doc, w, steps, thumbs, assess)

    if not topic:
        # ---- steps --------------------------------------------------------------
        w.heading("Steps")
        tw, th = (96, 66) if breakdown else (150, 110)
        for s in steps:
            r = s["report"]
            man_s, f = r["manual"], r["file"]
            rows = []
            if man_s.get("step_description"):
                rows.append(("What", man_s["step_description"], None))
            if r["workflow_type"]:
                rows.append(("Workflow", ", ".join(map(str, r["workflow_type"])), None))
            size = f"{f.get('width')}x{f.get('height')} · " if f.get("width") and f.get("height") else ""
            rng = CH.frame_range_text(r)
            rows.append(("Output", f"{f.get('name') or ''} · {size}{f.get('format') or ''} · "
                                   + (f"{rng} · " if rng else "")
                                   + f"{str(f.get('saved') or r['generated'])[:16].replace('T', ' ')}", None))
            for it in s["inputs"]:
                a = it["asset"]
                fn = CH.asset_name(a) + (
                    f" (added later, {str(a['added'])[:10]})" if a.get("added") else "")
                if it["from"] is not None:
                    rows.append(("Input", f"{fn}  <- {it['from']['id']}", None))
                elif it.get("declared"):
                    rows.append(("Input", f"{fn}  {it['declared']} (documented by this declaration)", GREY))
                elif it["hint"]:
                    rows.append(("Input", f"{fn}  GAP – {it['hint']}", GRADE_COLORS["C"]))
                else:
                    rows.append(("Input", f"{fn}  outside source – document origin and rights", GRADE_COLORS["C"]))
            for label, value in (CH.breakdown_details(r, licenses) if breakdown else
                                 CH.step_details(r, full=full_prompts)):
                col = None
                if label == "Model":
                    cu = value.rsplit("commercial use ", 1)[-1].strip().lower()
                    col = USE_COLORS.get(cu) if cu in ("no", "restricted") else (GREY if CH.UNK in value else None)
                    col = USE_COLORS["no"] if cu == "no" else col
                if CH.UNK in value and col is None:
                    col = GRADE_COLORS["C"]
                elif col is None and ("not applicable" in value or "added later" in value):
                    col = GREY
                rows.append((label, value, col))
            t = r["triage"] if assess else {}
            if t.get("grade"):
                rows.append(("Triage", f"{t['grade']} {t.get('label', '')} – {t.get('reason', '')}",
                             GRADE_COLORS.get(t["grade"])))
            if assess and not breakdown:
                lv = {"bad": GRADE_COLORS["X"], "warn": GRADE_COLORS["C"]}
                for label, value, level in CH.audit_details(r, full=full_prompts):
                    rows.append((label, value, lv.get(level)))
            if r["reproducibility"] and not breakdown:
                rows.append(("Reproducible", r["reproducibility"], None))
            if s.get("alternatives") and not breakdown:
                rows.append(("Same step", f"{len(s['alternatives'])} other output(s), not in this chain", GREY))

            # start on a new page when not even the picture and the first rows fit
            text_w = CONTENT_W - tw - 14
            w.need(th + (30 if breakdown else 40))
            y0 = w.y
            if assess:
                _badge(doc, PAGE_W - M_R - 18, y0, t.get("grade") or "-", 18)
            doc.text(M_L, y0 + 12, s["id"], 12, True, GRADE_COLORS.get(t.get("grade"), GREY))
            head = ((man_s.get("step") or "").strip() or "(no step name)") + \
                ("" if r.get("kind") == CH.PHOTO else "  (external step)" if CH.is_external(r) else "")
            doc.text(M_L + 30, y0 + 12, wrap(head, CONTENT_W - 60, 11, True)[0], 11, True)
            _thumb(doc, thumbs.get(s["id"]), M_L, y0 + 22, tw, th)
            if thumbs.get(s["id"]) is None and r.get("kind") == CH.PHOTO:
                doc.rect(M_L, y0 + 22, tw, th, fill=(0.96, 0.96, 0.96))
                _camera(doc, M_L + tw / 2, y0 + 22 + th / 2, tw * 0.4)
            page0 = len(doc.pages)
            y_end = w.pairs(M_L + tw + 14, y0 + 22, text_w, rows, 8, 70)
            w.y = (max(y_end, y0 + 22 + th) if len(doc.pages) == page0 else y_end) + 10
            if w.y > PAGE_H - M_B - 10:
                w.new_page()
            doc.line(M_L, w.y, PAGE_W - M_R, w.y, LIGHT, 0.6)
            w.y += 10

        # ---- models -------------------------------------------------------------
        if summary["models"] and not licenses:
            w.heading("Models in the chain")
            for m in summary["models"]:
                role = CH.ROLE_LABELS.get(m.get("role"), m.get("role") or "model")
                w.need(12)
                w.y = w.pairs(M_L, w.y, CONTENT_W, [(", ".join(m["steps"]), f"{m.get('name') or os.path.basename(str(m.get('file') or '?'))} ({role})", None)], 8.3, 110)
        if summary["models"] and licenses:
            w.heading("Models in the chain")
            cols = (M_L, M_L + 170, M_L + 330, M_L + 420)
            w.need(20)
            for cx, lab in zip(cols, ("Model", "License", "Commercial use", "Steps")):
                doc.text(cx, w.y + 8, lab, 7.5, True, GREY)
            w.y += 14
            for m in summary["models"]:
                name_l = wrap(m.get("name") or os.path.basename(str(m.get("file") or "?")), 160, 8.3)
                lic_l = wrap(CH._val(m.get("license")), 150, 8.3)
                notes = wrap(m.get("license_notes") or "", CONTENT_W - 170, 7.2) if m.get("license_notes") else []
                h = max(len(name_l), len(lic_l)) * 11 + len(notes) * 9.2 + 6
                w.need(h)
                cu, col = _use(m)
                for k, ln in enumerate(name_l):
                    doc.text(cols[0], w.y + 8 + k * 11, ln, 8.3, True)
                for k, ln in enumerate(lic_l):
                    doc.text(cols[1], w.y + 8 + k * 11, ln, 8.3)
                doc.dot(cols[2] + 3, w.y + 5, 3, col)
                for k, ln in enumerate(wrap(cu, cols[3] - cols[2] - 14, 8.3)):
                    doc.text(cols[2] + 10, w.y + 8 + k * 11, ln, 8.3, False, col)
                doc.text(cols[3], w.y + 8, ", ".join(m["steps"]), 8.3, False, GREY)
                yy = w.y + max(len(name_l), len(lic_l)) * 11
                for k, ln in enumerate(notes):
                    doc.text(cols[1], yy + 6 + k * 9.2, ln, 7.2, False, GREY)
                w.y += h + 2
                doc.line(M_L, w.y, PAGE_W - M_R, w.y, LIGHT, 0.4)
                w.y += 4

        # ---- outside sources, gaps, open points, note ---------------------------
        if summary["external_inputs"] or summary["gaps"]:
            w.heading("Outside sources and gaps")
            rows = [(e["step"], f"{e.get('file')} – outside source: document origin and rights", None)
                    for e in summary["external_inputs"]]
            rows += [(e["step"], f"{e.get('file')} – {e['hint']}", GRADE_COLORS["C"]) for e in summary["gaps"]]
            for row in rows:
                w.need(30)
                w.y = w.pairs(M_L, w.y, CONTENT_W, [row], 8.3, 40)
        if summary["open_points"] and assess:
            w.heading("Open points")
            grouped = {}
            for o in summary["open_points"]:
                grouped.setdefault(o["need"], []).append(o["step"])
            col_w = 130
            for need, sts in grouped.items():
                left = wrap(", ".join(sts), col_w - 8, 8.3)
                right = wrap(need, CONTENT_W - col_w, 8.3)
                h = max(len(left), len(right)) * 11 + 3
                w.need(h)
                for k, ln in enumerate(left):
                    doc.text(M_L, w.y + 8 + k * 11, ln, 8.3, False, GREY)
                for k, ln in enumerate(right):
                    doc.text(M_L + col_w, w.y + 8 + k * 11, ln, 8.3)
                w.y += h
    else:
        _topic_sections(doc, w, steps, summary, full_prompts, assess, licenses)
    for g, label, text in (CH.GRADE_LEGEND + [("-", "NO GRADE", CH.NO_GRADE_TEXT)]) if assess else []:
        if g == "A":
            w.heading("Triage grades")
        lines = wrap(text, CONTENT_W - 150, 8)
        h = 11 * len(lines) + 4
        w.need(h)
        if g == summary["worst_grade"]:
            doc.rect(M_L - 4, w.y - 1, CONTENT_W + 8, h - 1, fill=(0.97, 0.97, 0.97))
        _badge(doc, M_L, w.y + 1, g, 11)
        doc.text(M_L + 16, w.y + 9, label, 7.6, True, GRADE_COLORS.get(g, (0.6, 0.6, 0.6)))
        for k, ln in enumerate(lines):
            doc.text(M_L + 150, w.y + 8 + k * 11, ln, 8, False, DARK)
        w.y += h
    for ln in wrap(CH.GRADE_NOTE, CONTENT_W, 8) if assess else []:
        w.need(11)
        doc.text(M_L, w.y + 9, ln, 8, False, GREY)
        w.y += 10.5
    w.heading("Note")
    note = ("Links between steps are made by SHA256: an input of one step is exactly the file another step wrote. "
            "An edit outside ComfyUI (Krita, Nuke, Photoshop) changes the file – the link breaks there and shows as a "
            "gap or an outside source, unless the edit was declared with Pertineo Audit External Step; a close name "
            "is shown as a hint, never as a verified link. Declared steps and fields added later are statements of "
            "the user, not recorded by ComfyUI. " + (
                "Triage and licenses are copied from the single reports: an indicative first assessment, not an "
                "approval and not legal advice. " if assess or licenses else
                "This overview contains no assessment and no license information. ") +
            "The complete data is in the .chain.txt and .chain.json next to this PDF.")
    for ln in wrap(note, CONTENT_W, 8.3):
        w.need(12)
        doc.text(M_L, w.y + 8, ln, 8.3, False, GREY)
        w.y += 11
    w.finish()
    doc.save(path, ((title or "").strip() + " – " if (title or "").strip() else "") + f"Audit chain – {name}",
             man.get("author"))


def write_simple_pdf(path, kicker, title, subtitle, picture, rows, note, footer, author=None):
    """A one-subject PDF (used by Audit External Step): picture on the left, label / value rows, a note.
    rows: (label, value, color or None)."""
    doc = _Doc()
    w = _Writer(doc, footer)
    doc.text(M_L, w.y + 8, kicker, 8, True, GREY)
    lines = wrap(title, CONTENT_W, 16, True)[:2]
    for i, ln in enumerate(lines):
        doc.text(M_L, w.y + 28 + i * 19, ln, 16, True)
    w.y += 30 + 19 * len(lines)
    if subtitle:
        doc.text(M_L, w.y + 4, subtitle, 9, False, GREY)
        w.y += 18
    top = w.y
    if picture is not None:
        img_w, img_h = 200, 150
        _thumb(doc, picture, M_L, top, img_w, img_h)
        x = M_L + img_w + 16
        page0 = len(doc.pages)
        y_end = w.pairs(x, top, PAGE_W - M_R - x, rows, 8.5, 80)
        w.y = (max(y_end, top + img_h) if len(doc.pages) == page0 else y_end) + 16
    else:
        w.y = w.pairs(M_L, top, CONTENT_W, rows, 8.5, 90) + 16
    if note:
        w.heading("Note")
        for ln in wrap(note, CONTENT_W, 8.3):
            w.need(12)
            doc.text(M_L, w.y + 8, ln, 8.3, False, GREY)
            w.y += 11
    w.finish()
    doc.save(path, title, author)


def write_checklist_pdf(path, steps, final_name, version=""):
    """The disclosure checklist as its own PDF – incomplete items first, then all items with coloured dots."""
    items = CH.checklist(steps)
    miss = [c for c in items if c[2] == "missing"]
    doc = _Doc()
    w = _Writer(doc, f"Disclosure checklist · {final_name} · Pertineo VFX {version} · for checking the audit chain")
    doc.text(M_L, w.y + 8, "DISCLOSURE CHECKLIST", 8, True, GREY)
    lines = wrap(final_name, CONTENT_W, 16, True)[:2]
    for i, ln in enumerate(lines):
        doc.text(M_L, w.y + 28 + i * 19, ln, 16, True)
    w.y += 30 + 19 * len(lines)
    doc.text(M_L, w.y + 4, datetime.date.today().isoformat(), 9, False, GREY)
    w.y += 18
    res = (f"{len(miss)} of {len(items)} items incomplete" if miss else
           f"complete – all {len(items)} items given or not applicable")
    doc.text(M_L, w.y + 10, res, 11, True, GRADE_COLORS["C"] if miss else GRADE_COLORS["A"])
    w.y += 20
    steps_txt = ", ".join(f"{s['id']} {(s['report']['manual'].get('step') or '').strip()}".strip() for s in steps)
    w.y = w.pairs(M_L, w.y, CONTENT_W, [("Steps", steps_txt, GREY)], 8.3, 50) + 4
    lv_col = {"ok": GRADE_COLORS["A"], "na": (0.70, 0.70, 0.70), "missing": GRADE_COLORS["C"]}

    def block(rows):
        for item, status, level in rows:
            ls = wrap(status, CONTENT_W - 190, 8.3)
            w.need(11 * len(ls) + 4)
            doc.dot(M_L + 4, w.y + 5, 3.2, lv_col[level])
            doc.text(M_L + 12, w.y + 8, item, 8.3, level == "missing")
            for k, ln in enumerate(ls):
                doc.text(M_L + 190, w.y + 8 + k * 11, ln, 8.3, False,
                         lv_col[level] if level == "missing" else (GREY if level == "na" else DARK))
            w.y += 11 * len(ls) + 3

    if miss:
        w.heading("Incomplete")
        block(miss)
        w.y += 4
        for ln in wrap("How to close a gap: purpose, origin sources, external software and changes of an existing "
                       "report – Pertineo Audit External Step, mode 'amend report'. A file edited or made outside "
                       "ComfyUI – mode 'external step' next to that file. Values the workflow did not record "
                       "(resolution, CFG …) cannot be added later; they stay UNKNOWN / NOT VERIFIED.", CONTENT_W, 8):
            w.need(11)
            doc.text(M_L, w.y + 8, ln, 8, False, GREY)
            w.y += 10.5
    w.heading("All items")
    block(items)
    w.finish()
    doc.save(path, f"Disclosure checklist – {final_name}", None)
