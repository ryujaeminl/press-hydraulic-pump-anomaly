"""Minimal HWPX writer that fills the official contest-report template (paragraph styles are reused, not invented)."""
import math
import re
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

TEXT_W = 48188            # usable line width [HWPUNIT]
CP_MARK2, CP_MARK3, CP_MARK4 = 14, 13, 16          # template char styles of the "◦", "-", "*" markers
CP_BODY, CP_NOTE = 8, 15                            # 휴먼명조 14 / 10
CP_TBL, CP_TBL_B, CP_BODY_B, CP_NOTE_B = 37, 38, 39, 40   # added by _patch_header
PP_CENTER, PP_LEFT = 42, 43                         # added by _patch_header (table / caption paragraphs)
PP_B2, PP_B3, PP_B4 = 44, 45, 46                    # left-aligned copies of the template ◦ / - / * paragraph styles
BF_TBL, BF_HEAD = 8, 6                              # template border fills: solid box / solid box + grey


def _split_top(body: str) -> list:
    depth, start, tops = 0, 0, []
    for m in re.finditer(r"<(/?)hp:p\b[^>]*?(/?)>", body):
        if m.group(1) == "":
            if depth == 0:
                start = m.start()
            if m.group(2) != "/":
                depth += 1
            elif depth == 0:
                tops.append(body[start:m.end()])
        else:
            depth -= 1
            if depth == 0:
                tops.append(body[start:m.end()])
    assert "".join(tops) == body, "unexpected content between top-level paragraphs"
    return tops


def _runs(text: str, cp: int, cp_bold: int) -> str:
    """'a **b** c' -> three runs, the middle one bold."""
    out = []
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            out.append(f'<hp:run charPrIDRef="{cp_bold if i % 2 else cp}"><hp:t>{escape(part)}</hp:t></hp:run>')
    return "".join(out) or f'<hp:run charPrIDRef="{cp}"/>'


def _p(pp: int, runs: str, page_break: int = 0) -> str:
    return (f'<hp:p id="0" paraPrIDRef="{pp}" styleIDRef="0" pageBreak="{page_break}" columnBreak="0" merged="0">'
            f"{runs}</hp:p>")


class Template:
    def __init__(self, hwpx: Path):
        with zipfile.ZipFile(hwpx) as zf:
            self.z = {i.filename: zf.read(i.filename) for i in zf.infolist()}
        self.header = self.z["Contents/header.xml"].decode("utf8")
        sec = self.z["Contents/section0.xml"].decode("utf8")
        i = sec.index(">", sec.index("<hs:sec")) + 1
        j = sec.rindex("</hs:sec>")
        self.sec_head, self.sec_tail = sec[:i], sec[j:]
        self.paras = _split_top(sec[i:j])
        self.images: list = []           # (published name, bytes)
        self._uid = 1_900_000_000
        self._patch_header()

    # ------------------------------------------------------------------ header (extra styles)
    def _patch_header(self):
        h = self.header

        def char_copy(base, new_id, height=None, bold=False):
            x = re.search(rf'<hh:charPr id="{base}".*?</hh:charPr>', h, flags=re.S).group(0)
            x = x.replace(f'id="{base}"', f'id="{new_id}"', 1)
            if height:
                x = re.sub(r'height="\d+"', f'height="{height}"', x, count=1)
            if bold and "<hh:bold/>" not in x:
                x = x.replace("</hh:charPr>", "<hh:bold/></hh:charPr>")
            return x

        def para_copy(base, new_id, align, spacing=None):
            x = re.search(rf'<hh:paraPr id="{base}".*?</hh:paraPr>', h, flags=re.S).group(0)
            x = x.replace(f'id="{base}"', f'id="{new_id}"', 1)
            x = re.sub(r'horizontal="\w+"', f'horizontal="{align}"', x, count=1)
            if spacing is None:
                return x
            return re.sub(r'(<hh:lineSpacing type="PERCENT" value=")\d+', rf"\g<1>{spacing}", x)

        chars = [char_copy(15, CP_TBL, 900), char_copy(15, CP_TBL_B, 900, True),
                 char_copy(8, CP_BODY_B, None, True), char_copy(15, CP_NOTE_B, None, True)]
        paras = [para_copy(40, PP_CENTER, "CENTER", 130), para_copy(40, PP_LEFT, "LEFT", 130),
                 para_copy(25, PP_B2, "LEFT"), para_copy(26, PP_B3, "LEFT"), para_copy(27, PP_B4, "LEFT")]
        n_c = int(re.search(r'<hh:charProperties itemCnt="(\d+)"', h).group(1))
        n_p = int(re.search(r'<hh:paraProperties itemCnt="(\d+)"', h).group(1))
        assert n_c == CP_TBL and n_p == PP_CENTER, "template style counts changed"
        h = h.replace(f'<hh:charProperties itemCnt="{n_c}">', f'<hh:charProperties itemCnt="{n_c + len(chars)}">')
        h = h.replace("</hh:charProperties>", "".join(chars) + "</hh:charProperties>")
        h = h.replace(f'<hh:paraProperties itemCnt="{n_p}">', f'<hh:paraProperties itemCnt="{n_p + len(paras)}">')
        h = h.replace("</hh:paraProperties>", "".join(paras) + "</hh:paraProperties>")
        self.header = h

    def uid(self) -> int:
        self._uid += 1
        return self._uid

    # ------------------------------------------------------------------ block builders
    @staticmethod
    def b2(text):
        return _p(PP_B2, f'<hp:run charPrIDRef="{CP_MARK2}"><hp:t> ◦ </hp:t></hp:run>' + _runs(text, CP_BODY, CP_BODY_B))

    @staticmethod
    def b3(text):
        return _p(PP_B3, f'<hp:run charPrIDRef="{CP_MARK3}"><hp:t>   - </hp:t></hp:run>' + _runs(text, CP_BODY, CP_BODY_B))

    @staticmethod
    def b4(text):
        return _p(PP_B4, f'<hp:run charPrIDRef="{CP_MARK4}"><hp:t>       * </hp:t></hp:run>' + _runs(text, CP_NOTE, CP_NOTE_B))

    @staticmethod
    def gap():
        return _p(PP_B2, '<hp:run charPrIDRef="28"/>')

    @staticmethod
    def caption(text):
        return _p(PP_CENTER, _runs(text, CP_NOTE, CP_NOTE_B))

    @staticmethod
    def heading(text, page_break=0):
        return _p(20, f'<hp:run charPrIDRef="11"><hp:t>{escape(text)}</hp:t></hp:run>', page_break)

    def table(self, rows, widths, header_rows=1, left_cols=()):
        """rows: list of lists of str ('\\n' = new paragraph in the cell). widths: relative column widths."""
        tot = sum(widths)
        w = [int(TEXT_W * 0.98 * x / tot) for x in widths]
        trs, heights = [], []
        for r, row in enumerate(rows):
            head = r < header_rows
            lines = 1
            for c, txt in enumerate(row):
                cost = sum(900 if ord(ch) > 0x2E7F else 520 for ch in txt.replace("\n", ""))
                lines = max(lines, math.ceil(cost / max(w[c] - 600, 800)) + txt.count("\n"))
            h = lines * 1250 + 360
            heights.append(h)
            tcs = []
            for c, txt in enumerate(row):
                pp = PP_LEFT if (c in left_cols and not head) else PP_CENTER
                ps = "".join(_p(pp, _runs(t, CP_TBL, CP_TBL_B) if not head else
                                f'<hp:run charPrIDRef="{CP_TBL_B}"><hp:t>{escape(t)}</hp:t></hp:run>')
                             for t in txt.split("\n"))
                tcs.append(
                    f'<hp:tc name="" header="{int(head)}" hasMargin="0" protect="0" editable="0" dirty="0" '
                    f'borderFillIDRef="{BF_HEAD if head else BF_TBL}"><hp:subList id="" textDirection="HORIZONTAL" '
                    'lineWrap="BREAK" vertAlign="CENTER" linkListIDRef="0" linkListNextIDRef="0" textWidth="0" '
                    f'textHeight="0" hasTextRef="0" hasNumRef="0">{ps}</hp:subList><hp:cellAddr colAddr="{c}" '
                    f'rowAddr="{r}"/><hp:cellSpan colSpan="1" rowSpan="1"/><hp:cellSz width="{w[c]}" height="{h}"/>'
                    '<hp:cellMargin left="141" right="141" top="100" bottom="100"/></hp:tc>')
            trs.append("<hp:tr>" + "".join(tcs) + "</hp:tr>")
        tbl = (f'<hp:tbl id="{self.uid()}" zOrder="0" numberingType="TABLE" textWrap="TOP_AND_BOTTOM" '
               f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" pageBreak="CELL" repeatHeader="1" '
               f'rowCnt="{len(rows)}" colCnt="{len(widths)}" cellSpacing="0" borderFillIDRef="{BF_TBL}" noAdjust="0">'
               f'<hp:sz width="{sum(w)}" widthRelTo="ABSOLUTE" height="{sum(heights)}" heightRelTo="ABSOLUTE" protect="0"/>'
               '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
               'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               '<hp:outMargin left="140" right="140" top="140" bottom="140"/>'
               '<hp:inMargin left="140" right="140" top="140" bottom="140"/>' + "".join(trs) + "</hp:tbl>")
        return _p(PP_CENTER, f'<hp:run charPrIDRef="{CP_TBL}">{tbl}<hp:t/></hp:run>')

    def image(self, png: Path, width_ratio: float = 0.92) -> str:
        data = Path(png).read_bytes()
        pw, ph = struct.unpack(">II", data[16:24])
        name = f"image{len(self.images) + 2}.png"            # image1.bmp belongs to the template
        self.images.append((name, data))
        w = int(TEXT_W * width_ratio)
        h = int(w * ph / pw)
        pic = (f'<hp:pic id="{self.uid()}" zOrder="0" numberingType="PICTURE" textWrap="TOP_AND_BOTTOM" '
               f'textFlow="BOTH_SIDES" lock="0" dropcapstyle="None" href="" groupLevel="0" instid="{self.uid()}" reverse="0">'
               f'<hp:offset x="0" y="0"/><hp:orgSz width="{w}" height="{h}"/><hp:curSz width="{w}" height="{h}"/>'
               '<hp:flip horizontal="0" vertical="0"/>'
               f'<hp:rotationInfo angle="0" centerX="{w // 2}" centerY="{h // 2}" rotateimage="1"/>'
               '<hp:renderingInfo><hc:transMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
               '<hc:scaMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/>'
               '<hc:rotMatrix e1="1" e2="0" e3="0" e4="0" e5="1" e6="0"/></hp:renderingInfo>'
               f'<hp:imgRect><hc:pt0 x="0" y="0"/><hc:pt1 x="{w}" y="0"/><hc:pt2 x="{w}" y="{h}"/><hc:pt3 x="0" y="{h}"/></hp:imgRect>'
               f'<hp:imgClip left="0" right="{w}" top="0" bottom="{h}"/><hp:inMargin left="0" right="0" top="0" bottom="0"/>'
               f'<hp:imgDim dimwidth="{w}" dimheight="{h}"/>'
               f'<hc:img binaryItemIDRef="{name[:-4]}" bright="0" contrast="0" effect="REAL_PIC" alpha="0"/><hp:effects/>'
               f'<hp:sz width="{w}" widthRelTo="ABSOLUTE" height="{h}" heightRelTo="ABSOLUTE" protect="0"/>'
               '<hp:pos treatAsChar="1" affectLSpacing="0" flowWithText="1" allowOverlap="0" holdAnchorAndSO="0" '
               'vertRelTo="PARA" horzRelTo="COLUMN" vertAlign="TOP" horzAlign="LEFT" vertOffset="0" horzOffset="0"/>'
               '<hp:outMargin left="0" right="0" top="0" bottom="0"/></hp:pic>')
        return _p(PP_CENTER, f'<hp:run charPrIDRef="7">{pic}<hp:t/></hp:run>')

    # ------------------------------------------------------------------ cover page
    def fill_cover(self, project: str, team: str, summary_lines: list):
        p = self.paras[0]
        empties = list(re.finditer(r'<hp:run charPrIDRef="19"/>', p))
        assert len(empties) == 2, "cover layout changed"
        for m, text in reversed(list(zip(empties, [project, team]))):
            p = p[:m.start()] + f'<hp:run charPrIDRef="24"><hp:t>{escape(text)}</hp:t></hp:run>' + p[m.end():]
        cell = re.search(r'<hp:p id="2147483648" paraPrIDRef="31"[^>]*><hp:run charPrIDRef="22"/>.*?</hp:p>', p, flags=re.S)
        paras = "".join(_p(31, f'<hp:run charPrIDRef="25"><hp:t>{escape(t)}</hp:t></hp:run>') for t in summary_lines)
        self.paras[0] = p[:cell.start()] + paras + p[cell.end():]

    # ------------------------------------------------------------------ assemble
    def set_page_break(self, idx: int):
        self.paras[idx] = self.paras[idx].replace('pageBreak="0"', 'pageBreak="1"', 1)

    def replace_paragraphs(self, first: int, last: int, blocks: list):
        """Replace template paragraphs first..last (inclusive) by generated blocks."""
        self.paras[first:last + 1] = blocks

    def save(self, out: Path, preview_text: str = ""):
        z = dict(self.z)
        z["Contents/header.xml"] = self.header.encode("utf8")
        z["Contents/section0.xml"] = (self.sec_head + "".join(self.paras) + self.sec_tail).encode("utf8")
        hpf = z["Contents/content.hpf"].decode("utf8")
        items = "".join(f'<opf:item id="{n[:-4]}" href="BinData/{n}" media-type="image/png" isEmbeded="1"/>'
                        for n, _ in self.images)
        z["Contents/content.hpf"] = hpf.replace("</opf:manifest>", items + "</opf:manifest>").encode("utf8")
        for n, data in self.images:
            z[f"BinData/{n}"] = data
        if preview_text:
            z["Preview/PrvText.txt"] = preview_text.encode("utf8")
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w") as zf:
            zf.writestr("mimetype", z.pop("mimetype"), compress_type=zipfile.ZIP_STORED)
            for name, data in z.items():
                zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
