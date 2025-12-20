# Law Crawler

Crawler for vbpl.vn that downloads Vietnamese legal documents as JSON files grouped by document type.

## How it works
- Entry point loops over configured document type IDs and kicks off per-type crawling in [law-crawler/main.py](law-crawler/main.py#L105-L113).
- Each list page is paginated; page counting and URL building live in [law-crawler/main.py](law-crawler/main.py#L53-L104).
- Individual document pages are fetched, validity-checked, and written to disk as JSON in [law-crawler/main.py](law-crawler/main.py#L19-L52).
- Base URLs, ID ranges, and output folder are set in [law-crawler/config.py](law-crawler/config.py#L1-L4).

## Prerequisites
- Python 3.8+
- Dependencies: requests, beautifulsoup4

Install deps:
```bash
pip install requests beautifulsoup4
```

## Running the crawler
From the law-crawler folder:
```bash
python main.py
```
The script will iterate through the IDs in `ID_LISTS`, crawl each document type, and save results under `vbpl_documents/<document_type>/`.

## Configuration
- BASE_URL / START_URL: Target site and base list page.
- ID_LISTS: Document type IDs to crawl; adjust to expand or narrow coverage.
- DOWNLOAD_FOLDER: Root folder for saved JSON files.

Edit these in [law-crawler/config.py](law-crawler/config.py#L1-L4).

## Output
- Files: `vbpl_documents/<document_type>/<document_id>.json`
- JSON shape:
```json
{
	"Id": "170620",
	"Content": "Full document text..."
}
```

## Notes
- The crawler skips documents that are expired or partially expired, and it will not re-download files already on disk.
- A short delay between requests is included to reduce load on the source site; increase the delay if you crawl larger ID ranges.
