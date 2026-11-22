"""Load the original chapter labels from the episode pages for a private run."""

import hashlib
import json
import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen


CHAPTER_LINE = re.compile(
    r"(.+?)\s*[-–—]\s*(\d+):([0-5]\d):([0-5]\d)\s*\.?\s*$",
    flags=re.DOTALL,
)


class _EpisodePage(HTMLParser):
    """Collect timestamped list items inside the episode description."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.content_depth = 0
        self.item_parts = None
        self.chapters = []

    def handle_starttag(self, tag, attrs):
        if tag == "div":
            classes = (dict(attrs).get("class") or "").split()
            if self.content_depth or "post-podcast-content" in classes:
                self.content_depth += 1
        if tag == "li" and self.content_depth:
            self.item_parts = []

    def handle_data(self, data):
        if self.item_parts is not None:
            self.item_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "div" and self.content_depth:
            self.content_depth -= 1
        if tag == "li" and self.item_parts is not None:
            label = "".join(self.item_parts).strip()
            self.item_parts = None
            match = CHAPTER_LINE.fullmatch(label)
            if match:
                title, hours, minutes, seconds = match.groups()
                start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                self.chapters.append({"start": float(start), "title": title.strip()})


def chapter_digest(chapters):
    """Hash the ordered (time, label) records using a stable JSON encoding."""
    data = json.dumps(chapters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_manifest(episodes):
    """Restore the private input structure and reject changes to the reference labels."""
    restored = []
    for episode in episodes:
        page = episode["page"]
        request = Request(page, headers={"User-Agent": "PodcastChapterResearch/1.0"})
        with urlopen(request, timeout=30) as response:
            markup = response.read().decode("utf-8")

        parser = _EpisodePage()
        parser.feed(markup)
        parser.close()
        chapters = parser.chapters
        metadata = dict(episode)
        expected_hash = metadata.pop("chapter_labels_sha256")
        if chapter_digest(chapters) != expected_hash:
            raise ValueError(f"Авторские названия, порядок или таймкоды глав изменились на странице {page}")
        metadata["chapters"] = chapters
        restored.append(metadata)
    return restored
