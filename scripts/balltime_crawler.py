#!/usr/bin/env python3
import argparse
import csv
import random
import re
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

DEFAULT_OUTPUT_DIR = Path("/Users/arashoutadi/volleyball-videos/balltime_versions")


def sanitize_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[\[\](){}]", "", name)
    name = re.sub(r"[,\'\"]", "", name)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def get_video_ids(page) -> list[str]:
    content = page.content()
    pattern = r"/video/([a-f0-9-]{36})"
    matches = re.findall(pattern, content)
    return list(set(matches))


def load_processed(processed_file: Path) -> set[str]:
    if not processed_file.exists():
        return set()
    lines = processed_file.read_text().strip().split("\n")
    return set(line for line in lines if line)


def save_processed(processed_file: Path, video_ids: set[str]):
    processed_file.write_text("\n".join(sorted(video_ids)))


def extract_rally_times(page) -> list[dict]:
    content = page.content()
    pattern = r"(\d+:\d{2})\s*-\s*(\d+:\d{2})"
    matches = re.findall(pattern, content)
    return [{"start": start, "end": end} for start, end in matches]


def get_video_title(page) -> str:
    title_el = page.query_selector("input[name='name']")
    if title_el:
        return title_el.get_attribute("value") or ""
    return ""


def jitter(min_s=1, max_s=3):
    time.sleep(random.uniform(min_s, max_s))


def download_video(page, download_dir: Path) -> Path | None:
    # Set download path for this page
    cdp = page.context.new_cdp_session(page)
    cdp.send(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": str(download_dir.resolve())},
    )

    jitter(0.5, 1.5)
    page.click("button:has-text('Export')")
    page.wait_for_selector("text=Export Settings", timeout=60000)
    jitter(0.5, 1)

    title = get_video_title(page)
    sanitized_name = sanitize_filename(title) if title else "video"

    jitter(0.5, 1)
    with page.expect_download(timeout=600000) as download_info:
        page.click("button:has-text('Export Video')")

    download = download_info.value
    ext = Path(download.suggested_filename).suffix
    dest_path = download_dir / f"{sanitized_name}{ext}"
    download.save_as(dest_path)
    return dest_path


def process_video(page, video_id: str, output_dir: Path) -> bool:
    url = f"https://app.balltime.com/video/{video_id}"
    page.goto(url)
    page.wait_for_load_state("networkidle", timeout=30000)
    jitter(1, 2)

    # Dismiss "Recording Setup Issues Detected" popup
    try:
        page.click("button:has-text('Got it')", timeout=2000)
        jitter(0.5, 1)
    except Exception:
        pass

    try:
        rallies = extract_rally_times(page)
        jitter(0.5, 1.5)
        video_path = download_video(page, output_dir)

        if video_path and rallies:
            csv_path = video_path.with_suffix(".csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["start", "end"])
                writer.writeheader()
                writer.writerows(rallies)
            print(f"  -> {video_path.name} ({len(rallies)} rallies)")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Crawl and process Balltime videos")
    parser.add_argument("--folder-id", required=True, help="Balltime folder ID")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--processed-file", type=Path, default=Path("processed_videos.txt")
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Download and extract all available videos",
    )
    parser.add_argument(
        "--mark-processed", nargs="+", help="Mark video IDs as processed"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Take screenshot for debugging"
    )
    args = parser.parse_args()

    processed = load_processed(args.processed_file)

    if args.mark_processed:
        processed.update(args.mark_processed)
        save_processed(args.processed_file, processed)
        print(f"Marked {len(args.mark_processed)} videos as processed")
        return

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.new_page()

        folder_url = f"https://app.balltime.com/videos?folderId={args.folder_id}"
        page.goto(folder_url)
        page.wait_for_load_state("networkidle", timeout=30000)

        # Dismiss "Invite teammates" popup if present
        try:
            close_btn = (
                page.locator("text=Invite teammates").locator("..").locator("svg").first
            )
            close_btn.click(timeout=2000)
        except Exception:
            pass

        if args.debug:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.output_dir / "debug.png"), full_page=True)
            print(f"Screenshot saved to {args.output_dir / 'debug.png'}")
            page.close()
            return

        all_videos = get_video_ids(page)
        available = [v for v in all_videos if v not in processed]

        print(
            f"Total: {len(all_videos)} | Processed: {len(all_videos) - len(available)} | Available: {len(available)}"
        )

        if not args.process:
            if available:
                print("\nAvailable:")
                for vid in available:
                    print(f"  {vid}")
            page.close()
            return

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for i, vid in enumerate(available, 1):
            print(f"\n[{i}/{len(available)}] Processing {vid}")
            if process_video(page, vid, args.output_dir):
                processed.add(vid)
                save_processed(args.processed_file, processed)
            jitter(2, 5)

        page.close()

    print(f"\nDone. Processed {len(processed)} total videos.")


if __name__ == "__main__":
    main()
