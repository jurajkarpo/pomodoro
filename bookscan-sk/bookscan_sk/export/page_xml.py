"""PAGE XML export.

PAGE (Page Analysis and Ground-truth Elements) is a
standard XML format for page-level OCR and layout
analysis results, used by e.g. OCR4all and eScriptorium.

Reference: https://github.com/PRImA-Research-Lab/PAGE-XML
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..core.types import OCRPage, PageResult

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/0.9"


def export_page_xml(
    pages: list[PageResult],
    output_path: str,
    first_page_number: int = 1,
) -> str:
    """Export OCR results to a PAGE XML file."""
    pc_gts = ET.Element("PcGts", {
        "xmlns": PAGE_NS,
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": (
            f"{PAGE_NS} http://schema.primaresearch.org/PAGE/gts/pagecontent/0.9/PAGE-0.9.xsd"
        ),
    })
    metadata = ET.SubElement(pc_gts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "BookScan SK"
    ET.SubElement(metadata, "Created").text = "2026-01-01T00:00:00"

    for i, page in enumerate(pages):
        if page.ocr is None:
            continue
        page_no = first_page_number + i
        ocr = page.ocr
        page_el = ET.SubElement(pc_gts, "Page", {
            "imageFilename": page.source.filename if page.source else f"page_{page_no}",
            "imageWidth": str(ocr.width),
            "imageHeight": str(ocr.height),
        })
        for block in ocr.blocks:
            region_el = ET.SubElement(page_el, "TextRegion", {
                "id": f"region_{page_no}_{block.reading_order}",
            })
            if block.box:
                ET.SubElement(region_el, "Coords", {
                    "points": (
                        f"{int(block.box.x0)},{int(block.box.y0)} "
                        f"{int(block.box.x1)},{int(block.box.y0)} "
                        f"{int(block.box.x1)},{int(block.box.y1)} "
                        f"{int(block.box.x0)},{int(block.box.y1)}"
                    )
                })
            for line in block.lines:
                line_el = ET.SubElement(region_el, "TextLine", {
                    "id": f"line_{page_no}_{block.reading_order}_{id(line)}",
                })
                if line.box:
                    ET.SubElement(line_el, "Coords", {
                        "points": (
                            f"{int(line.box.x0)},{int(line.box.y0)} "
                            f"{int(line.box.x1)},{int(line.box.y0)} "
                            f"{int(line.box.x1)},{int(line.box.y1)} "
                            f"{int(line.box.x0)},{int(line.box.y1)}"
                        )
                    })
                for word in line.words:
                    word_el = ET.SubElement(line_el, "Word", {
                        "id": f"word_{page_no}_{id(word)}",
                    })
                    bx = word.box
                    ET.SubElement(word_el, "Coords", {
                        "points": (
                            f"{int(bx.x0)},{int(bx.y0)} "
                            f"{int(bx.x1)},{int(bx.y0)} "
                            f"{int(bx.x1)},{int(bx.y1)} "
                            f"{int(bx.x0)},{int(bx.y1)}"
                        )
                    })
                    te = ET.SubElement(word_el, "TextEquiv")
                    ET.SubElement(te, "Unicode").text = word.display_text
                    ET.SubElement(te, "Confidence").text = f"{word.confidence:.3f}"

    xml_str = ET.tostring(pc_gts, encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    return output_path
