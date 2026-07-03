#!/usr/bin/env python3
"""
全量统计所有目录下每个说话人的音频总时长和缺口。多进程并行。

--source-dirs: 原始音频目录。speaker_path 由此决定。
--clone-dirs:  复刻音频目录 (任意文件名格式), 时长计入但标记为 clone。

Usage:
  python 00_prepare_stats.py \
      --source-dirs /data/audio \
      --clone-dirs /data/clone1 /data/clone2 \
      --output-dir ./stats_output \
      --workers 32
"""

import argparse, csv, json, os, struct, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

AUDIO_EXTS = {".wav", ".flac", ".mp3"}
TARGET_SEC = 3600.0
SKIP_DIRS = frozenset({"logs", "__pycache__", "ref", "eval_sim_embedding_cache"})


def _wav_dur(fpath: str, fsize: int) -> float:
    try:
        with open(fpath, "rb") as f:
            h = f.read(44)
        if len(h) >= 44 and h[:4] == b"RIFF" and h[8:12] == b"WAVE":
            ch = struct.unpack_from("<H", h, 22)[0]
            sr = struct.unpack_from("<I", h, 24)[0]
            bps = struct.unpack_from("<H", h, 34)[0]
            if sr > 0 and bps > 0 and ch > 0:
                return (fsize - 44) / (sr * ch * (bps / 8))
    except OSError:
        pass
    return 0.0


def _dur(path: str) -> float:
    try:
        fsize = os.path.getsize(path)
    except OSError:
        return 0.0
    if fsize <= 1000:
        return 0.0
    if os.path.splitext(path)[1].lower() == ".wav":
        return _wav_dur(path, fsize)
    try:
        import soundfile as sf

        return sf.info(path).duration
    except Exception:
        return 0.0


def _list_files(spk_path: str) -> list:
    """列出 spk_path 下所有音频文件路径"""
    files = []
    has_sub = False
    try:
        for e in os.scandir(spk_path):
            if e.is_file() and os.path.splitext(e.name)[1].lower() in AUDIO_EXTS:
                files.append(e.path)
            elif e.is_dir() and e.name not in SKIP_DIRS:
                has_sub = True
    except OSError:
        return files
    if has_sub:
        for dp, dns, fns in os.walk(spk_path):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(dp, fn))
    return files


def _scan_speakers(args: tuple) -> dict:
    """
    扫描一批说话人.
    args = (task_label, [(dataset, spk_id, source_paths, clone_paths)])
    source_paths: 原始音频目录列表
    clone_paths:  复刻音频目录列表
    返回 {(dataset,spk_id): {source_sec, clone_sec, num_files, has_7to20s, spk_path}}
    """
    label, items = args
    out: dict = {}
    for dataset, spk_id, src_paths, cln_paths in items:
        # 源音频
        src_files = []
        for p in src_paths:
            src_files.extend(_list_files(p))
        src_durs = [d for fp in src_files if (d := _dur(fp)) > 0]
        source_sec = sum(src_durs)

        # clone 音频
        cln_files = []
        for p in cln_paths:
            cln_files.extend(_list_files(p))
        cln_durs = [d for fp in cln_files if (d := _dur(fp)) > 0]
        clone_sec = sum(cln_durs)

        all_durs = src_durs + cln_durs
        if not all_durs:
            continue

        key = (dataset, spk_id)
        # spk_path 取自源目录 (第一个有音频的)
        spk_path = src_paths[0] if src_paths else (cln_paths[0] if cln_paths else "")
        out[key] = {
            "source_sec": source_sec,
            "clone_sec": clone_sec,
            "total_sec": source_sec + clone_sec,
            "num_files": len(all_durs),
            "has_7to20s": any(7 <= x <= 20 for x in all_durs),
            "has_source": len(src_durs) > 0,
            "spk_path": spk_path,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dirs", nargs="+", default=[], help="原始音频目录")
    p.add_argument("--clone-dirs", nargs="+", default=[], help="复刻音频目录 (时长计入, speaker_path 不取自此)")
    p.add_argument("--output-dir", default=None, help="输出目录")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--target-sec", type=float, default=TARGET_SEC, help="每个说话人目标时长(秒), 默认 3600")
    a = p.parse_args()
    workers = a.workers or min(cpu_count(), 32)
    target = a.target_sec
    t0 = time.time()

    # ---- 收集说话人 ----
    # spk_map: (dataset, spk_id) → {"source_paths": [...], "clone_paths": [...]}
    spk_map: dict[tuple[str, str], dict] = defaultdict(lambda: {"source_paths": [], "clone_paths": []})

    for root in a.source_dirs:
        if not os.path.isdir(root):
            continue
        for ds in os.listdir(root):
            ds_path = os.path.join(root, ds)
            if not os.path.isdir(ds_path):
                continue
            for spk_id in os.listdir(ds_path):
                spk_path = os.path.join(ds_path, spk_id)
                if os.path.isdir(spk_path):
                    spk_map[(ds, spk_id)]["source_paths"].append(spk_path)

    for root in a.clone_dirs:
        if not os.path.isdir(root):
            continue
        for ds in os.listdir(root):
            ds_path = os.path.join(root, ds)
            if not os.path.isdir(ds_path):
                continue
            for spk_id in os.listdir(ds_path):
                spk_path = os.path.join(ds_path, spk_id)
                if os.path.isdir(spk_path):
                    spk_map[(ds, spk_id)]["clone_paths"].append(spk_path)

    all_speakers = sorted(spk_map.items())
    n_total = len(all_speakers)
    n_has_source = sum(1 for _, v in spk_map.items() if v["source_paths"])
    n_clone_only = n_total - n_has_source
    print(f"共 {n_total:,} 个说话人 (有源音频: {n_has_source:,}, 仅有复刻: {n_clone_only:,})")
    print(f"源目录: {len(a.source_dirs)} 个, 复刻目录: {len(a.clone_dirs)} 个")

    # ---- 分块并行 ----
    CHUNK = 20
    tasks = []
    for i in range(0, n_total, CHUNK):
        chunk = all_speakers[i : i + CHUNK]
        items = []
        for ds_spk, paths in chunk:
            items.append((ds_spk[0], ds_spk[1], paths["source_paths"], paths["clone_paths"]))
        tasks.append((f"{i // CHUNK + 1}/{n_total // CHUNK + 1}", items))

    print(f"共 {len(tasks)} 个任务, {workers} workers", flush=True)

    # ---- 并行扫描 ----
    stats: dict = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_speakers, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            partial = fut.result()
            for key, info in partial.items():
                if key not in stats:
                    stats[key] = info
                else:
                    stats[key]["source_sec"] += info["source_sec"]
                    stats[key]["clone_sec"] += info["clone_sec"]
                    stats[key]["total_sec"] += info["total_sec"]
                    stats[key]["num_files"] += info["num_files"]
                    stats[key]["has_7to20s"] = stats[key]["has_7to20s"] or info["has_7to20s"]
                    stats[key]["has_source"] = stats[key]["has_source"] or info["has_source"]
            done += 1
            print(f"  [{done:>4}/{len(tasks)}] 累计 {len(stats):,} speakers", flush=True)

    scan_t = time.time() - t0

    # ---- 统计 ----
    rows = []
    ok = need = insuf = clone_only_need = 0
    gap_total = 0.0
    for (dataset, spk_id), info in sorted(stats.items()):
        dur = info["total_sec"]
        gap = max(0.0, target - dur)
        if dur >= target:
            ok += 1
        elif info["num_files"] >= 20:
            need += 1
            gap_total += gap
            if not info["has_source"]:
                clone_only_need += 1
        else:
            insuf += 1
            gap_total += gap
        rows.append(
            {
                "dataset": dataset,
                "speaker_id": spk_id,
                "num_files": info["num_files"],
                "has_7to20s": info["has_7to20s"],
                "has_source_audio": info["has_source"],
                "source_duration_sec": round(info["source_sec"], 2),
                "clone_duration_sec": round(info["clone_sec"], 2),
                "total_duration_sec": round(info["total_sec"], 2),
                "gap_sec": round(gap, 2),
                "status": "OK" if dur >= target else ("NEED" if info["num_files"] >= 20 else "LOW"),
                "speaker_path": info["spk_path"],
            }
        )

    # ---- 输出 ----
    print(f"\n扫描: {scan_t:.0f}s", flush=True)
    print("=" * 72)
    print(f"总说话人:           {len(stats):,}")
    print(f"已达标:             {ok:,}")
    print(f"需克隆:             {need:,}  (其中仅有复刻音频: {clone_only_need:,})")
    print(f"数据不足:           {insuf:,}")
    print(f"累计缺口:           {gap_total / 3600:,.1f} 小时")
    print()

    for lo, hi, lb in [
        (0, 60, "<1min"),
        (60, 300, "1-5m"),
        (300, 600, "5-10m"),
        (600, 1200, "10-20m"),
        (1200, 2400, "20-40m"),
        (2400, 3600, "40-60m"),
        (3600, 7200, "1-2h"),
        (7200, 999999, ">2h"),
    ]:
        n = sum(1 for r in rows if lo <= r["total_duration_sec"] < hi)
        if n:
            bar = "#" * max(1, n // max(1, len(rows) // 80))
            print(f"  {lb:>8s}: {n:>7,}  {bar}")

    need_list = sorted([r for r in rows if r["status"] == "NEED"], key=lambda x: -x["gap_sec"])
    print(f"\n--- 缺口最大 TOP 20 ---")
    print(f"  {'Dataset/Speaker':<50s} {'源音频':>8s} {'复刻':>8s} {'缺口':>8s} {'文件':>6s} {'有源':>4s}")
    for r in need_list[:20]:
        has_src = "Y" if r["has_source_audio"] else "N"
        print(
            f"  {r['dataset']}/{r['speaker_id']:<45s} "
            f"{r['source_duration_sec']:>7.0f}s {r['clone_duration_sec']:>7.0f}s "
            f"{r['gap_sec']:>7.0f}s {r['num_files']:>5}  {has_src:>4}"
        )
    if len(need_list) > 20:
        print(f"  ... 共 {len(need_list):,} 人")

    print(f"\n--- 按 dataset ---")
    print(f"  {'Dataset':<30s} {'总数':>7s} {'已达标':>7s} {'需克隆':>7s} {'不足':>7s} {'仅复刻':>7s}")
    ds_grp = defaultdict(list)
    for r in rows:
        ds_grp[r["dataset"]].append(r)
    for ds in sorted(ds_grp):
        rr = ds_grp[ds]
        t = len(rr)
        ok_n = sum(1 for r in rr if r["status"] == "OK")
        need_n = sum(1 for r in rr if r["status"] == "NEED")
        low_n = sum(1 for r in rr if r["status"] == "LOW")
        clone_n = sum(1 for r in rr if r["status"] == "NEED" and not r["has_source_audio"])
        print(f"  {ds:<30s} {t:>7,} {ok_n:>7,} {need_n:>7,} {low_n:>7,} {clone_n:>7,}")

    if a.output_dir:
        os.makedirs(a.output_dir, exist_ok=True)
        csv_p = os.path.join(a.output_dir, "all_speakers.csv")
        fields = [
            "dataset",
            "speaker_id",
            "num_files",
            "has_7to20s",
            "has_source_audio",
            "source_duration_sec",
            "clone_duration_sec",
            "total_duration_sec",
            "gap_sec",
            "status",
            "speaker_path",
        ]
        with open(csv_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {csv_p}")
        s = {
            "total": len(stats),
            "ok": ok,
            "need_clone": need,
            "insufficient": insuf,
            "clone_only_need": clone_only_need,
            "gap_hours": round(gap_total / 3600, 2),
            "scan_sec": round(scan_t, 1),
            "source_dirs": a.source_dirs,
            "clone_dirs": a.clone_dirs,
        }
        jp = os.path.join(a.output_dir, "summary.json")
        with open(jp, "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        print(f"JSON: {jp}")

    print(f"\n总耗时: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
