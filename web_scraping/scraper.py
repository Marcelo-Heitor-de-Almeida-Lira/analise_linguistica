from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "articles" / "pdf"
JSON_DIR = BASE_DIR / "articles" / "json_update"

PDF_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Due to some problems with bot detection, WEBMEDIA and SBES were excluded from the search

EVENTS = {
    "2020": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/596",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/616",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/619",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/682",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/581",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/575"
    },
    "2021": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/760",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/851",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/790",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/843",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/772",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/765"
    },
    "2022": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/954",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/1027",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/963",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/987",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/937",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/949"
    },
    "2023": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/1129",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/1150",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/1102",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/1145",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/1117",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/1120"
    },
    "2024": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/1307",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/1460",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/1340",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/1371",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/1306",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/1327"
    },
    "2025": {
        "semish": "https://sol.sbc.org.br/index.php/semish/issue/view/1558",
        "eniac": "https://sol.sbc.org.br/index.php/eniac/issue/view/1601",
        "sbrc": "https://sol.sbc.org.br/index.php/sbrc/issue/view/1500",
        "sbbd": "https://sol.sbc.org.br/index.php/sbbd/issue/view/1577",
        "brasnam": "https://sol.sbc.org.br/index.php/brasnam/issue/view/1543",
        "wics": "https://sol.sbc.org.br/index.php/wics/issue/view/1531"
    },
}

DEFAULT_USER_AGENT = (
    "marcelo-estagio2026 academic scraper "
    "(contact: add-institutional-email-in-readme)"
)
BLOCK_STATUSES = {403, 429, 503}
MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\ufffd",
    "\u00cb\u2020",
)


class BlockedRequestError(RuntimeError):
    pass


def make_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def polite_sleep(min_delay: float, jitter: float) -> None:
    time.sleep(min_delay + random.uniform(0, jitter))


def response_text(response: requests.Response) -> str:
    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    return response.text


def fetch(
    session: requests.Session,
    url: str,
    timeout: tuple[int, int],
    min_delay: float,
    jitter: float,
) -> requests.Response:
    polite_sleep(min_delay, jitter)
    response = session.get(url, timeout=timeout)

    if response.status_code in BLOCK_STATUSES or looks_like_cloudflare_challenge(response):
        raise BlockedRequestError(f"Bloqueio/rate limit detectado em {url}: {response.status_code}")

    response.raise_for_status()
    return response


def looks_like_cloudflare_challenge(response: requests.Response) -> bool:
    body = response.text[:4000].lower()
    signals = (
        "cloudflare",
        "checking your browser",
        "cf-chl",
        "just a moment",
        "attention required",
    )
    return any(signal in body for signal in signals)


def looks_mojibaked(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def clean_text(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\t", " ")
    cleaned = " ".join(cleaned.split())

    if looks_mojibaked(cleaned):
        try:
            cleaned = cleaned.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass

    return unicodedata.normalize("NFC", cleaned)


def soup_from_response(response: requests.Response) -> BeautifulSoup:
    return BeautifulSoup(response_text(response), "lxml")


def article_id_from_url(url: str) -> str:
    parts = [part for part in url.rstrip("/").split("/") if part]
    return parts[-1] if parts else "unknown"


def find_article_titles(issue_soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
    return issue_soup.find_all("div", class_="title")


def extract_abstract(article_soup: BeautifulSoup) -> str | None:
    abstract_tag = article_soup.find("div", class_="item abstract")
    if abstract_tag is None:
        return None

    heading = abstract_tag.find("h3")
    if heading is not None:
        heading.decompose()

    return clean_text(abstract_tag.get_text(" "))


def find_pdf_entry(article_soup: BeautifulSoup) -> tuple[str, str] | None:
    for css_class in ("obj_galley_link pdf", "obj_galley_link file"):
        pdf_tag = article_soup.find("a", class_=css_class)
        if pdf_tag is not None and pdf_tag.get("href"):
            lang = pdf_tag.text.strip().upper()

            if lang == "PDF":
                lang = "portuguese"
            elif lang == "PDF (ENGLISH)":
                lang = "english"
            return [css_class, pdf_tag.get("href")], lang
    return None


def resolve_pdf_download(
    session: requests.Session,
    pdf_class: str,
    pdf_url: str,
    timeout: tuple[int, int],
    min_delay: float,
    jitter: float,
) -> tuple[str, str]:
    pdf_response = fetch(session, pdf_url, timeout, min_delay, jitter)
    pdf_soup = soup_from_response(pdf_response)

    if pdf_class == "obj_galley_link pdf":
        download_tag = pdf_soup.find("a", class_="download")
        if download_tag is None or not download_tag.get("href"):
            raise ValueError(f"Link de download nao encontrado: {pdf_url}")

        download_url = urljoin(pdf_url, download_tag.get("href"))
        pdf_name = article_id_from_url(download_url)
        return download_url, pdf_name

    acm_format_panel = pdf_soup.find("div", class_="info-panel__formats info-panel__item")
    if acm_format_panel is None:
        raise ValueError(f"Painel de formato ACM nao encontrado: {pdf_url}")

    acm_page_tag = acm_format_panel.find("a")
    if acm_page_tag is None or not acm_page_tag.get("href"):
        raise ValueError(f"Pagina ACM nao encontrada: {pdf_url}")

    acm_page_url = urljoin(pdf_url, acm_page_tag.get("href"))
    acm_response = fetch(session, acm_page_url, timeout, min_delay, jitter)
    acm_soup = soup_from_response(acm_response)
    download_tag = acm_soup.find("div", class_="grouped_right")
    if download_tag is None or download_tag.find("a") is None:
        raise ValueError(f"Download ACM nao encontrado: {acm_page_url}")

    download_url = urljoin(acm_page_url, download_tag.find("a").get("href"))
    pdf_name = article_id_from_url(pdf_url)
    return download_url, pdf_name


def download_pdf(
    session: requests.Session,
    download_url: str,
    pdf_path: Path,
    overwrite: bool,
    timeout: tuple[int, int],
    min_delay: float,
    jitter: float,
) -> None:
    if pdf_path.exists() and pdf_path.stat().st_size > 0 and not overwrite:
        return

    response = fetch(session, download_url, timeout, min_delay, jitter)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(response.content)


def scrape_event(
    year: str,
    event_name: str,
    url: str,
    args: argparse.Namespace,
) -> tuple[str, str, int]:
    session = make_session(args.user_agent)
    json_path = args.json_dir / f"{event_name}_{year}.json"

    if json_path.exists() and not args.overwrite:
        print(f"[SKIP] {event_name} - {year}: JSON ja existe")
        return event_name, year, 0

    issue_response = fetch(session, url, args.timeout, args.min_delay, args.jitter)
    issue_soup = soup_from_response(issue_response)

    articles: OrderedDict[str, OrderedDict[str, object]] = OrderedDict()

    for title_tag in find_article_titles(issue_soup):
        try:
            link_tag = title_tag.find("a")
            if link_tag is None or not link_tag.get("href"):
                raise ValueError("Titulo sem link")

            title = clean_text(link_tag.get_text(" "))
            article_url = urljoin(url, link_tag.get("href"))
            article_response = fetch(session, article_url, args.timeout, args.min_delay, args.jitter)
            article_soup = soup_from_response(article_response)

            pdf_entry, lang = find_pdf_entry(article_soup)
            pdf_url = None
            download_url = None
            pdf_path = None
            article_key = article_id_from_url(article_url)

            if pdf_entry is not None:
                pdf_class, pdf_url = pdf_entry
                pdf_url = urljoin(article_url, pdf_url)
                download_url, pdf_name = resolve_pdf_download(
                    session, pdf_class, pdf_url, args.timeout, args.min_delay, args.jitter
                )
                article_key = pdf_name
                pdf_path = args.pdf_dir / f"{event_name}_{year}_{pdf_name}.pdf"

                print(f"[INFO]: {event_name} - {year} - {title} - {lang}")

                if not Path(pdf_path).is_file():
                    download_pdf(
                        session,
                        download_url,
                        pdf_path,
                        args.overwrite,
                        args.timeout,
                        args.min_delay,
                        args.jitter,
                    )
                else:
                    print(f"[SKIP PDF] {pdf_name} já existe localmente.")

            article_id = len(articles) + 1
            articles[article_key] = OrderedDict(
                [
                    ("ID", article_id),
                    ("event", event_name),
                    ("year", year),
                    ("title", title),
                    ("abstract", extract_abstract(article_soup)),
                    ("article_url", article_url),
                    ("pdf_url", pdf_url),
                    ("pdf_download_url", download_url),
                    ("pdf_path", str(pdf_path.relative_to(BASE_DIR)) if pdf_path else None),
                    ("language", lang),
                ]
            )
        except BlockedRequestError:
            raise
        except Exception as exc:
            print(f"[ERRO] {event_name} - {year}: {exc}")

    args.json_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=4)
        file.write("\n")

    print(f"[OK] {event_name} - {year}: {len(articles)} artigos")
    return event_name, year, len(articles)


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    years = args.years or sorted(EVENTS)
    events = args.events
    tasks = []

    for year in years:
        if year not in EVENTS:
            raise ValueError(f"Ano nao configurado: {year}")

        event_map = EVENTS[year]
        selected_events = events or sorted(event_map)
        for event_name in selected_events:
            if event_name not in event_map:
                raise ValueError(f"Evento nao configurado para {year}: {event_name}")
            tasks.append((year, event_name, event_map[event_name]))

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scraper responsavel para artigos da SBC OpenLib.")
    parser.add_argument("--years", nargs="+", help="Anos a coletar. Exemplo: --years 2024 2025")
    parser.add_argument("--events", nargs="+", help="Eventos a coletar. Exemplo: --events semish eniac")
    parser.add_argument("--max-workers", type=int, default=2, help="Concorrencia maxima.")
    parser.add_argument("--min-delay", type=float, default=1.5, help="Pausa minima entre requisicoes.")
    parser.add_argument("--jitter", type=float, default=1.0, help="Variacao aleatoria adicional.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescreve JSONs/PDFs existentes.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent das requisicoes.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help="Diretorio dos PDFs.")
    parser.add_argument("--json-dir", type=Path, default=JSON_DIR, help="Diretorio dos JSONs.")
    parser.add_argument("--connect-timeout", type=int, default=10, help="Timeout de conexao.")
    parser.add_argument("--read-timeout", type=int, default=30, help="Timeout de leitura.")

    args = parser.parse_args()
    args.timeout = (args.connect_timeout, args.read_timeout)
    args.pdf_dir = args.pdf_dir.resolve()
    args.json_dir = args.json_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    tasks = selected_tasks(args)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(scrape_event, year, event_name, url, args): (year, event_name)
            for year, event_name, url in tasks
        }

        for future in as_completed(futures):
            year, event_name = futures[future]
            try:
                future.result()
            except BlockedRequestError as exc:
                print(f"[BLOQUEIO] {event_name} - {year}: {exc}")
            except Exception as exc:
                print(f"[FALHA] {event_name} - {year}: {exc}")


if __name__ == "__main__":
    main()
