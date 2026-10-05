"""ALTO XML export.

ALTO (Analyzed Layout and Text Object) is a standard
XML schema for OCR results, used by libraries and
archives. This exporter writes ALTO v4.

Reference: http://www.loc.gov/standards/alto/
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..core.types import OCRPage, PageResult

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
ALTO_XSI = "http://www.w3.org/2001/XMLSchema-instance"


def export_alto(
    pages: list[PageResult],
    output_path: str,
    first_page_number: int = 1,
) -> str:
    """Export OCR results to an ALTO XML file."""
    alto = ET.Element("alto", {
        "xmlns": ALTO_NS,
        "xmlns:xsi": ALTO_XSI,
        "xsi:schemaLocation": (
            "http://www.loc.gov/standards/alto/ns-v4# "
            "http://www.loc.gov/standards/alto/v4/alto-4-0.xsd"
        ),
    })
    description = ET.SubElement(alto, "Description")
    ET.SubElement(description, "MeasurementUnit").text = "pixel"
    ET.SubElement(
        ET.SubElement(description, "sourceImageInformation"), "fileName"
    )

    layout = ET.SubElement(alto, "Layout")

    for i, page in enumerate(pages):
        if page.ocr is None:
            continue
        page_no = first_page_number + i
        ocr = page.ocr
        page_el = ET.SubElement(layout, "Page", {
            "ID": f"page_{page_no}",
            "HEIGHT": str(ocr.height),
            "WIDTH": str(ocr.width),
            "PHYSICAL_IMG_NR": str(page_no),
        })
        print_space = ET.SubElement(page_el, "PrintSpace", {
            "ID": f"ps_{page_no}",
            "HPOS": "0",
            "VPOS": "0",
            "WIDTH": str(ocr.width),
            "HEIGHT": str(ocr.height),
        })
        for block in ocr.blocks:
            block_el = ET.SubElement(print_space, "TextBlock", {
                "ID": f"block_{page_no}_{block.reading_order}",
                "HPOS": str(int(block.box.x0)) if block.box else "0",
                "VPOS": str(int(block.box.y0)) if block.box else "0",
                "WIDTH": str(int(block.box.width)) if block.box else "0",
                "HEIGHT": str(int(block.box.height)) if block.box else "0",
            })
            for line in block.lines:
                line_el = ET.SubElement(block_el, "TextLine", {
                    "ID": f"line_{page_no}_{block.reading_order}_{id(line)}",
                    "HPOS": str(int(line.box.x0)) if line.box else "0",
                    "VPOS": str(int(line.box.y0)) if line.box else "0",
                    "WIDTH": str(int(line.box.width)) if line.box else "0",
                    "HEIGHT": str(int(line.box.height)) if line.box else "0",
                })
                for word in line.words:
                    ET.SubElement(line_el, "String", {
                        "ID": f"str_{page_no}_{id(word)}",
                        "CONTENT": word.display_text,
                        "HPOS": str(int(word.box.x0)),
                        "VPOS": str(int(word.box.y0)),
                        "WIDTH": str(int(word.box.width)),
                        "HEIGHT": str(int(word.box.height)),
                        "WC": f"{word.confidence:.3f}",
                    })

    xml_str = ET.tostring(alto, encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    return output_path
