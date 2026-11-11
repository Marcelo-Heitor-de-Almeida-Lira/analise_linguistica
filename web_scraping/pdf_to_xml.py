#!/usr/bin/env python3
"""Send PDFs to GROBID and save the raw TEI XML for debugging."""

import os
import requests
from pathlib import Path

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def main():
    ROOT = Path(__file__).resolve().parent.parent
    pdf_dir = ROOT / "web_scraping" / "articles" / "pdf"
    xml_dir = ROOT / "web_scraping" / "articles" / "converted_xml"
    
    # Create the output directory if it doesn't exist
    xml_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} does not exist.")
        return
    arquivo = "eniac_2024_33589.pdf"
    for pdf_file in [arquivo]:
        if not pdf_file.lower().endswith(".pdf"):
            continue

        full_pdf_path = pdf_dir / pdf_file
        out_path = xml_dir / f"{full_pdf_path.stem}.xml"

        print(f"Sending {pdf_file} to GROBID...")
        
        try:
            with open(full_pdf_path, "rb") as f:
                files = {"input": (pdf_file, f, "application/pdf")}
                data = {
                    "generateIDs": "1",
                    "consolidateHeader": "0",
                    "consolidateCitations": "0",
                }
                response = requests.post(GROBID_URL, files=files, data=data, timeout=120)

            if response.status_code == 200:
                # Force UTF-8 to preserve the raw diacritics exactly as GROBID saw them
                response.encoding = "utf-8"
                
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(response.text)
                    
                print(f"  Saved raw XML -> {out_path}")
            else:
                print(f"  GROBID Error {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            print(f"  Failed to process {pdf_file}: {e}")

if __name__ == "__main__":
    main()