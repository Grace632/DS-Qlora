"""Fetch CBETA primary texts once and make source-traceable token chunks."""
import argparse
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import requests
from transformers import AutoTokenizer

WORKS = {
    'T0262': '妙法蓮華經', 'T0277': '佛說觀普賢菩薩行法經',
    'T0276': '無量義經', 'T1509': '大智度論',
    'T1915': '修習止觀坐禪法要', 'T1929': '四教義',
    'T1911': '摩訶止觀', 'T0374': '大般涅槃經',
}
BASE = 'https://cbdata.dila.edu.tw/stable/'


def fetch(root):
    raw = root / 'data/raw/cbeta'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = raw / 'sources.json'
    previous = json.loads(manifest.read_text('utf-8')) if manifest.exists() else {}
    session = requests.Session()
    records = {}
    for work, title in WORKS.items():
        old = previous.get(work)
        if old and all((raw / f).exists() and hashlib.sha256((raw / f).read_bytes()).hexdigest() == h
                       for f, h in old['files'].items()):
            records[work] = old
            print('reuse', work, len(old['files']), 'volumes', flush=True)
            continue
        url = f'{BASE}download/text/{work}.txt.zip'
        response = session.get(url, timeout=(15, 90))
        response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        files = {}
        headers = []
        for name in archive.namelist():
            if not re.fullmatch(work + r'_\d+\.txt', name):
                continue
            data = archive.read(name)
            text = data.decode('utf-8-sig', errors='strict')
            if '\ufffd' in text or '#【經文資訊】' not in text:
                raise ValueError(f'Unexpected text encoding or missing CBETA header: {name}')
            (raw / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
            headers.extend(line for line in text.splitlines() if line.startswith('#【'))
        if not files:
            raise ValueError(f'No primary text in {url}')
        records[work] = dict(title=title, url=url, edition='CBETA / 大正新脩大藏經',
                             headers=list(dict.fromkeys(headers)), files=files)
        manifest.write_text(json.dumps(records | {k: v for k, v in previous.items() if k not in records},
                                       ensure_ascii=False, indent=2), 'utf-8')
        print('download', work, title, len(files), 'volumes', flush=True)
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), 'utf-8')
    return records


def clean(text):
    text = unicodedata.normalize('NFC', text.lstrip('\ufeff'))
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b]', '', text)
    # Keep original wording, punctuation, headings and notes. Only metadata/page-only lines go.
    lines = [line.rstrip() for line in text.splitlines()
             if not line.startswith('#') and not re.fullmatch(r'\s*(?:第\s*)?\d+\s*頁?\s*', line)]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def preprocess(root, model, maximum=768, overlap=80):
    records = fetch(root)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    def count(text): return len(tokenizer.encode(text, add_special_tokens=False))
    def split_long(text):
        while count(text) > maximum:
            low, high = 1, min(len(text), maximum)
            while low < high:
                mid = (low + high + 1) // 2
                if count(text[:mid]) <= maximum: low = mid
                else: high = mid - 1
            end = low
            boundary = max(text.rfind(c, max(1, end // 2), end) for c in '。！？；\n')
            if boundary > 0: end = boundary + 1
            yield text[:end]
            text = text[end:]
        if text: yield text
    output = root / 'data/processed/chunks.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for work, meta in records.items():
            seen = set()
            stats = dict(raw_chars=0, clean_chars=0, chunks=0, tokens=0, duplicate_chunks=0, short_filtered=0)
            for filename, sha in sorted(meta['files'].items()):
                raw = (root / 'data/raw/cbeta' / filename).read_text('utf-8-sig')
                text = clean(raw)
                stats['raw_chars'] += len(raw); stats['clean_chars'] += len(text)
                paragraphs = re.split(r'\n\s*\n', text)
                buffer, start, end, serial = [], 0, 0, 0
                def emit():
                    nonlocal serial
                    content = '\n\n'.join(buffer).strip()
                    if not content: return
                    if len(re.findall(r'[\u3400-\u9fff]', content)) < 20:
                        stats['short_filtered'] += 1; return
                    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    if digest in seen:
                        stats['duplicate_chunks'] += 1; return
                    seen.add(digest); serial += 1
                    n = count(content)
                    assert n <= maximum, (filename, n)
                    row = dict(id=f'{Path(filename).stem}-c{serial:04}', work=work, title=meta['title'],
                               source_file=filename, source_sha256=sha, url=meta['url'],
                               location=f'{Path(filename).stem}: paragraphs {start}-{end}',
                               paragraph_start=start, paragraph_end=end, text=content, tokens=n,
                               sha256=digest)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stats['chunks'] += 1; stats['tokens'] += n
                for index, paragraph in enumerate(paragraphs, 1):
                    for piece in split_long(paragraph):
                        if buffer and count('\n\n'.join(buffer + [piece])) > maximum:
                            previous = buffer[-1]
                            previous_end = end
                            emit()
                            buffer = [previous] if count(previous) <= overlap and count(previous + '\n\n' + piece) <= maximum else []
                            start = previous_end if buffer else index
                        if not buffer: start = index
                        buffer.append(piece); end = index
                emit()
            stats['average_tokens'] = round(stats['tokens'] / max(stats['chunks'], 1), 1)
            print(work, meta['title'], json.dumps(stats, ensure_ascii=False), flush=True)
    print('saved', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', default='models/DeepSeek-V2-Lite-Chat')
    parser.add_argument('--max-tokens', type=int, default=768)
    args = parser.parse_args()
    preprocess(args.root, args.model, args.max_tokens)
