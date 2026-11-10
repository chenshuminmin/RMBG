"""多线程分段下载 HF 模型文件（hf-mirror 单线程限速明显，分段可提速 3 倍以上）。

用法: python _dl/fetch_model.py <repo> <filename> <out_path> [threads]
支持断点续传：每个分片写在 <out>.partN，全部完成后合并。
"""
import concurrent.futures
import os
import sys
import time
import urllib.request

ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(url: str, idx: int, start: int, end: int, out: str) -> int:
    part = f"{out}.part{idx}"
    if os.path.exists(part) and os.path.getsize(part) == end - start + 1:
        return end - start + 1
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "curl/8"}
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    return os.path.getsize(part)


def main() -> None:
    repo, filename, out = sys.argv[1], sys.argv[2], sys.argv[3]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    url = f"{ENDPOINT}/{repo}/resolve/main/{filename}"
    total = head_size(url)
    print(f"总大小: {total / 1048576:.1f} MB, 分片: {threads}", flush=True)

    chunk = (total + threads - 1) // threads
    jobs = []
    for i in range(threads):
        start = i * chunk
        end = min(start + chunk - 1, total - 1)
        if start <= end:
            jobs.append((i, start, end))

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        futures = [ex.submit(fetch_chunk, url, i, s, e, out) for i, s, e in jobs]
        done = 0
        while done < len(futures):
            time.sleep(5)
            done = sum(1 for f in futures if f.done())
            got = sum(os.path.getsize(f"{out}.part{i}") for i, _, _ in jobs
                      if os.path.exists(f"{out}.part{i}"))
            speed = got / max(0.1, time.time() - t0) / 1048576
            print(f"进度 {got / 1048576:.1f}/{total / 1048576:.1f} MB "
                  f"({got / total * 100:.1f}%) {speed:.2f} MB/s", flush=True)
        for f in futures:
            f.result()

    with open(out, "wb") as w:
        for i, _, _ in jobs:
            with open(f"{out}.part{i}", "rb") as r:
                while True:
                    buf = r.read(1 << 23)
                    if not buf:
                        break
                    w.write(buf)
    for i, _, _ in jobs:
        os.remove(f"{out}.part{i}")

    size = os.path.getsize(out)
    print(f"完成: {out} {size / 1048576:.1f} MB, 耗时 {time.time() - t0:.0f}s", flush=True)
    if size != total:
        print(f"⚠️ 大小不一致: 期望 {total} 实际 {size}")
        sys.exit(2)


if __name__ == "__main__":
    main()
