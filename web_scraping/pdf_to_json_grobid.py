#!/usr/bin/env python3
"""
GROBID + Poppler Hybrid TEI XML Processor
Features:
- Poppler Alignment Engine (Wide-Window In-Block Splice & Ghost Duplicate Fix)
- Canonical Academic Whitelist
- Footer, Table, Pseudocode, & Chart Leak Traps
- Hierarchical Sanity Pass (Blocks out-of-order numerical jumps)
- Bilingual Abstract Separation (pt/en)
- Nested JSON Structure (Main Sections + Subsections)
- Global Main Section Accumulator
- Footnote Triage & Un-smashing
- Front-Matter Dam (Introduction-based cutoff)
- Runtime OCR Fallback (for Vectorized PDFs) with Original Filename Preservation
"""

import json
import os
import re
import unicodedata
import subprocess
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup
import requests

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

SECTION_ACCUMULATOR = set()

CANONICAL_WHITELIST = [
    "agradecimentos", "acknowledgements", "referências", "references", 
    "transparência da ia", "ethical considerations", "trabalhos relacionados", 
    "related work", "revisão da", "revisão de", "literature review", 
    "estado da arte", "state of the art", "referencial teórico", 
    "theoretical background", "metodologia", "methodology", "materiais e métodos", 
    "materials and methods", "proposed method", "proposed approach", "avaliação", 
    "evaluation", "experimentos", "experiments", "validação", "validation",
    "resultados", "results", "discussão", "discussões", "discussion",
    "conclusão", "conclusões", "conclusion", "considerações finais", 
    "trabalhos futuros", "future work", "limitações", "limitations", 
    "ameaças à validade", "threats to validity"
]


def run_poppler(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def extract_poppler_anchors(pdf_path: str) -> list:
    sections = []
    try:
        out_file = tempfile.mktemp(suffix=".txt")
        subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", pdf_path, out_file], 
                       check=True, stderr=subprocess.DEVNULL)
        with open(out_file, encoding="utf-8") as f:
            lines = f.readlines()
        os.remove(out_file)
    except Exception as e:
        print(f"Poppler warning (skipping structural audit): {e}")
        return sections

    whitelist_pattern = r'^(' + '|'.join(CANONICAL_WHITELIST) + r')$'

    for i, line in enumerate(lines):
        line_str = line.strip()
        
        # Broadened regex to catch en-dashes, colons, and accents in the remainder of the string
        match_num = re.match(r'^(?:((?:\d+\.)+\d*|\d+)|([IVXLCDM]+))\.?\s+([A-Z].*)$', line_str)
        match_white = re.match(whitelist_pattern, line_str, re.IGNORECASE)
        
        if (match_num or match_white) and len(line_str) < 150:
            header = line_str
            anchor = ""
            for j in range(i + 1, min(i + 15, len(lines))):
                candidate = lines[j].strip()
                if candidate and not candidate.startswith(('\x0c', 'Figure', 'Table', 'Figura', 'Tabela', 'Fig.', 'Tab.')):
                    anchor = candidate
                    break
            
            if anchor:
                sections.append({"header": header, "anchor": anchor})
                
    return sections


def normalize_for_match(text: str, truncate: int = 0) -> str:
    """Normalizes strings for matching. Removes accents, spaces, punctuation."""
    if not text: return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9]', '', text.lower())
    return text[:truncate] if truncate else text


def call_grobid(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
        data = {"generateIDs": "1", "consolidateHeader": "0", "consolidateCitations": "0"}
        response = requests.post(GROBID_URL, files=files, data=data, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"GROBID failed: {response.text[:200]}")
    response.encoding = "utf-8"
    return response.text


def fix_detached_diacritics(text: str) -> str:
    if not text: return text
    replacements = {
        r"c\s*¸": "ç", r"C\s*¸": "Ç", r"a\s*˜": "ã", r"A\s*˜": "Ã",
        r"o\s*˜": "õ", r"O\s*˜": "Õ", r"a\s*´": "á", r"A\s*´": "Á",
        r"e\s*´": "é", r"E\s*´": "É", r"i\s*´": "í", r"I\s*´": "Í",
        r"o\s*´": "ó", r"O\s*´": "Ó", r"u\s*´": "ú", r"U\s*´": "Ú",
        r"a\s*ˆ": "â", r"A\s*ˆ": "Â", r"e\s*ˆ": "ê", r"E\s*ˆ": "Ê",
        r"o\s*ˆ": "ô", r"O\s*ˆ": "Ô",
    }
    for pattern, fixed in replacements.items():
        text = re.sub(pattern, fixed, text)
    return unicodedata.normalize("NFC", text)


def process_references_and_brackets(soup: BeautifulSoup) -> int:
    ref_count = 0
    list_bibl = soup.find("listBibl")
    if list_bibl:
        ref_count = len(list_bibl.find_all("biblStruct"))
        list_bibl.decompose() 

    for ref in soup.find_all("ref"):
        ref.unwrap() 
        
    return ref_count


def clean_bracket_noise(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\s*\[[^\]]{0,150}\]\s*', ' ', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    return " ".join(text.split())


def strip_abstract_prefix(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.,]+\s*,?\s*', '', text)
    text = re.sub(r'^\s*(Resumo|Abstract|Palavras[- ]chave|Keywords)[\.\:\-]?\s+', '', text, flags=re.IGNORECASE)
    return text.strip()


def detect_language(text: str) -> str:
    text_lower = text.lower()
    if "resumo" in text_lower or "palavras-chave" in text_lower: return "pt"
    if "abstract" in text_lower or "keywords" in text_lower: return "en"
    
    pt_score = len(re.findall(r'\b(de|que|em|para|com|os|as|uma|um|não|são)\b', text_lower))
    en_score = len(re.findall(r'\b(the|and|of|in|to|is|for|on|with|this|are)\b', text_lower))
    return "en" if en_score > pt_score else "pt"


def extract_and_route_abstracts(soup: BeautifulSoup, abstract_json: dict):
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        for p in abstract_tag.find_all("p"):
            text = " ".join(fix_detached_diacritics(p.get_text(separator=" ")).split())
            text = clean_bracket_noise(text)
            text = strip_abstract_prefix(text)
            if not text: continue
            lang = detect_language(text)
            abstract_json[lang] += text + "\n\n"
        abstract_tag.decompose()

    for note in soup.find_all("note", place="foot"):
        raw_text = note.get_text(separator=" ")
        text = fix_detached_diacritics(raw_text)
        text = clean_bracket_noise(text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > 250 or "Resumo." in text or "Introduc" in text:
            text = re.sub(r'Proceedings of.*?Brazil', '', text, flags=re.IGNORECASE)
            text = re.sub(r'([a-z\u00E0-\u00FF\)\.])([A-Z1-9])', r'\1 \2', text)

            resumo_match = re.search(r'(Resumo\..*?)(1\.\s*[A-Z]|$)', text)
            if resumo_match:
                extracted = strip_abstract_prefix(resumo_match.group(1))
                abstract_json["pt"] += extracted + "\n\n"
                text = text.replace(resumo_match.group(1), '')

            abs_match = re.search(r'(Abstract\..*?)(1\.\s*[A-Z]|$)', text)
            if abs_match:
                extracted = strip_abstract_prefix(abs_match.group(1))
                abstract_json["en"] += extracted + "\n\n"
                text = text.replace(abs_match.group(1), '')

            if text.strip():
                new_p = soup.new_tag("p")
                new_p.string = text.strip()
                body = soup.find("body")
                if body: body.insert(0, new_p)
            note.decompose()
            
    abstract_json["pt"] = abstract_json["pt"].strip()
    abstract_json["en"] = abstract_json["en"].strip()


def is_valid_section(head_tag, metadata=None) -> bool:
    if not head_tag: return False
    text = head_tag.get_text(strip=True)
    text_lower = text.lower()
    
    # 1. Footer & Meta Traps
    if "et al." in text_lower: return False
    if metadata and metadata.get("authors"):
        for author in metadata["authors"]:
            if len(author) > 5 and author.lower() in text_lower: return False

    # 2. Block Float/Percentage table leaks
    if not re.search(r'[a-zA-Z]', text): return False
    
    # 3. Code, Math, and Ugly Punctuation Traps (Pseudocode destruction)
    if re.search(r'[\|\=\{\}\*\[\]\+\<\>\_\\]', text): return False
    if re.search(r'\b(foreach|while|else|return|break|emit|true|false)\b', text_lower): return False
    if re.search(r'[a-zA-Z]\.[a-zA-Z]', text): return False 

    # 4. Wild Number Trap (Blocks artifacts like "314 WB 4 sec.")
    num_match = re.match(r'^(\d+)\s+', text)
    if num_match and int(num_match.group(1)) > 20: return False
        
    # 5. Valid Numbered Header
    if head_tag.has_attr("n") and head_tag["n"].strip(): return True
        
    # 6. Canonical Whitelist
    if any(w in text_lower for w in CANONICAL_WHITELIST): return True
        
    # 7. Strict Fallback Regex
    if re.match(r'^(?:(?:\d+\.)+\d*|\d+|[IVXLCDM]+)\.?\s+[A-Za-z]', text): return True
        
    return False


def parse_body_to_blocks(soup: BeautifulSoup, metadata: dict) -> list:
    blocks = []
    body = soup.find("body")
    if not body: return blocks

    for div in body.find_all("div"):
        head = div.find("head", recursive=False)
        if head and is_valid_section(head, metadata):
            head_num = head.get("n", "").strip()
            head_title = fix_detached_diacritics(head.get_text(separator=" "))
            section_title = f"{head_num} {head_title}".strip() if head_num and not head_title.startswith(head_num) else head_title
            blocks.append({"type": "header", "text": section_title})

        for tag in div.find_all(["figure", "table", "formula"]):
            tag.decompose()

        for item in div.find_all(["p", "item"], recursive=False):
            normalized = " ".join(fix_detached_diacritics(item.get_text(separator=" ")).split())
            normalized = clean_bracket_noise(normalized)
            if not normalized: continue
            
            if re.match(r'^(Figura|Tabela|Quadro|Figure|Table)\s*\d+[\.\-:]', normalized, re.IGNORECASE) and len(normalized) < 200:
                continue
            if re.match(r'^\s*\([a-z]\)\s*[A-Z]', normalized):
                continue
            alpha_chars = sum(c.isalpha() for c in normalized)
            if len(normalized) > 0 and (alpha_chars / len(normalized)) < 0.4:
                continue
                
            blocks.append({"type": "p", "text": normalized})
            
    return blocks


def process_pdf_to_json(pdf_path: str, out_path: str, original_filename: str = None) -> dict:
    source_name = original_filename if original_filename else os.path.basename(pdf_path)
    poppler_sections = extract_poppler_anchors(pdf_path)

    xml_content = call_grobid(pdf_path)
    soup = BeautifulSoup(xml_content, "xml")

    ref_count = process_references_and_brackets(soup)

    abstract_data = {"pt": "", "en": ""}
    extract_and_route_abstracts(soup, abstract_data)

    metadata = {"title": None, "authors": [], "reference_count": ref_count}
    title_stmt = soup.find("titleStmt")
    if title_stmt and title_stmt.find("title"):
        raw_title = title_stmt.find("title").get_text(strip=True)
        metadata["title"] = fix_detached_diacritics(raw_title)

    for author in soup.find_all("author"):
        pers_name = author.find("persName")
        if pers_name:
            forenames = [fn.get_text(strip=True) for fn in pers_name.find_all("forename")]
            surname = pers_name.find("surname")
            surname_text = surname.get_text(strip=True) if surname else ""
            raw_name = " ".join(forenames + ([surname_text] if surname_text else [])).strip()
            fixed_name = fix_detached_diacritics(raw_name)
            if fixed_name and fixed_name not in metadata["authors"]:
                metadata["authors"].append(fixed_name)

    grobid_blocks = parse_body_to_blocks(soup, metadata)

    # --- 4. POPPLER ALIGNMENT ENGINE (Rescue & Ghost Fix) ---
    for p_sec in poppler_sections:
        p_header = p_sec["header"]
        p_anchor = p_sec["anchor"]
        
        # Check against full normalized strings to avoid ghost duplicates
        header_norm = normalize_for_match(p_header)
        anchor_norm = normalize_for_match(p_anchor, truncate=40)
        
        found_in_grobid = False
        for block in grobid_blocks:
            if block["type"] == "header" and normalize_for_match(block["text"]) == header_norm:
                found_in_grobid = True
                break
                
        if found_in_grobid:
            continue
            
        # WIDE-WINDOW IN-BLOCK SPLICE (Untouched)
        for i, block in enumerate(grobid_blocks):
            if block["type"] == "p":
                block_norm = normalize_for_match(block["text"])
                
                # Check the entire normalized block to locate swallowed headers
                if header_norm and header_norm in block_norm:
                    escaped_words = [re.escape(w) for w in p_header.split()]
                    pattern = r'\s+'.join(escaped_words)
                    match = re.search(pattern, block["text"], re.IGNORECASE)

                    if match:
                        before_text = block["text"][:match.start()].strip()
                        after_text = block["text"][match.end():].strip()

                        new_blocks = []
                        if before_text: new_blocks.append({"type": "p", "text": before_text})
                        new_blocks.append({"type": "header", "text": p_header})
                        if after_text: new_blocks.append({"type": "p", "text": after_text})

                        # Splice it in directly, replacing the mangled block
                        grobid_blocks = grobid_blocks[:i] + new_blocks + grobid_blocks[i+1:]
                        break

                # Backup: If header text was totally garbled but anchor matched
                elif len(anchor_norm) >= 15 and block_norm.startswith(anchor_norm):
                    grobid_blocks.insert(i, {"type": "header", "text": p_header})
                    break
    # -------------------------------------------------------------

    # --- 5. THE FRONT-MATTER DAM ---
    intro_index = -1
    for i, block in enumerate(grobid_blocks):
        if block["type"] == "header":
            text_lower = block["text"].lower()
            if re.search(r'\b(introdu[cç][aã]o|introduction)\b', text_lower) or re.match(r'^1\.\s+', text_lower):
                intro_index = i
                break

    if intro_index > 0:
        rogue_blocks = grobid_blocks[:intro_index]
        for block in rogue_blocks:
            text = block["text"]
            if len(text) < 15:  
                continue
            text = strip_abstract_prefix(text)
            if not text: continue
            lang = detect_language(text)
            abstract_data[lang] += text + "\n\n"
        
        abstract_data["pt"] = abstract_data["pt"].strip()
        abstract_data["en"] = abstract_data["en"].strip()
        
        grobid_blocks = grobid_blocks[intro_index:]
    # -------------------------------------------------------------

    # --- HIERARCHICAL SANITY PASS ---
    current_main_num = 0
    for block in grobid_blocks:
        if block["type"] == "header":
            title = block["text"]
            match = re.match(r'^(\d+)', title)
            if match:
                num = int(match.group(1))
                if current_main_num > 0:
                    is_whitelisted = any(w in title.lower() for w in CANONICAL_WHITELIST)
                    # Block sequential jumps if it's not a known back-matter whitelist
                    if not is_whitelisted and (num > current_main_num + 2 or num < current_main_num):
                        block["type"] = "p"
                        continue
                current_main_num = max(current_main_num, num)

    # 6. Build Nested JSON from Corrected Blocks
    sections = []
    current_main = None
    active_container = None

    for block in grobid_blocks:
        if block["type"] == "header":
            title = block["text"]
            is_sub = bool(re.match(r'^(\d+(?:\.\d+)+)\.?\s+', title))
                
            if is_sub:
                if current_main is None:
                    current_main = {"sectionTitle": "Orphaned Subsections", "sectionContent": "", "subsections": []}
                    sections.append(current_main)
                
                current_sub = {"sectionTitle": title, "sectionContent": ""}
                current_main["subsections"].append(current_sub)
                active_container = current_sub
            else:
                current_main = {"sectionTitle": title, "sectionContent": "", "subsections": []}
                sections.append(current_main)
                active_container = current_main
                
                clean_title = re.sub(r'^(\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+', '', title).strip().title()
                SECTION_ACCUMULATOR.add(clean_title)
                
        elif block["type"] == "p":
            if active_container is not None:
                active_container["sectionContent"] += block["text"] + "\n\n"
            else:
                current_main = {"sectionTitle": "Preamble", "sectionContent": block["text"] + "\n\n", "subsections": []}
                sections.append(current_main)
                active_container = current_main

    # 7. Cleanup content strings & construct full text
    full_prose_blocks = []
    if abstract_data["pt"]: full_prose_blocks.append(abstract_data["pt"])
    if abstract_data["en"]: full_prose_blocks.append(abstract_data["en"])

    for s in sections:
        s["sectionContent"] = s["sectionContent"].strip()
        if s["sectionContent"]: full_prose_blocks.append(s["sectionContent"])
        for sub in s["subsections"]:
            sub["sectionContent"] = sub["sectionContent"].strip()
            if sub["sectionContent"]: full_prose_blocks.append(sub["sectionContent"])
            
    full_prose = "\n\n".join(full_prose_blocks)

    document = {
        "source_file": source_name,
        "metadata": metadata,
        "abstract": abstract_data,
        "sections": sections,
        "full_text": full_prose,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    return document


def run_ocr_fallback(input_pdf: str, output_pdf: str) -> bool:
    try:
        print(f"  -> Vectorized PDF/Crash detected. Running OCR on {os.path.basename(input_pdf)}...")
        subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "por+eng", input_pdf, output_pdf],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"  -> OCR failed: {e}")
        return False


def main():
    ROOT = Path(__file__).resolve().parent.parent
    pdf_dir = ROOT / "web_scraping" / "articles" / "pdf"
    out_dir = ROOT / "dataset" / "converted_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} does not exist.")
        return

    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        file_name = Path(pdf_file).stem
        input_path = str(pdf_dir / pdf_file)
        out_path = str(out_dir / f"{file_name}.json")
        
        try:
            document = process_pdf_to_json(input_path, out_path, original_filename=pdf_file)
            
            # Health Check 
            word_count = len(document["full_text"].split())
            if word_count < 50 or not document["sections"]:
                raise ValueError("Word count too low or sections missing.")
                
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            # Trigger OCR if GROBID crashes with NO_BLOCKS or word count is low
            if "NO_BLOCKS" in error_msg or "empty content" in error_msg or "Word count too low" in error_msg:
                temp_ocr_pdf = tempfile.mktemp(suffix=".pdf")
                
                ocr_success = run_ocr_fallback(input_path, temp_ocr_pdf)
                
                if ocr_success:
                    print(f"  -> OCR successful. Reprocessing {file_name}...")
                    try:
                        document = process_pdf_to_json(temp_ocr_pdf, out_path, original_filename=pdf_file)
                    except Exception as fallback_e:
                        print(f"  -> Failed to parse {file_name} even after OCR: {fallback_e}")
                        
                if os.path.exists(temp_ocr_pdf):
                    os.remove(temp_ocr_pdf)
            else:
                print(f"Failed to process {pdf_file}: {e}")
                
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")
        else:
            print(f"Processed: {file_name}")

    print("\n" + "="*50)
    print("UNIQUE MAIN SECTIONS FOUND ACROSS DATASET:")
    print("="*50)
    for section in sorted(SECTION_ACCUMULATOR):
        print(f"- {section}")


if __name__ == "__main__":
    main()