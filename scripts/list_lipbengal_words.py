from pathlib import Path

root = Path("datasets/LipBengal/s1")
if not root.exists():
    print("Error: datasets/LipBengal/s1 not found")
    raise SystemExit(1)

dirs = sorted(p.name for p in root.iterdir() if p.is_dir())

out = Path("datasets/LipBengal/word_list.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(dirs) + ("\n" if dirs else ""), encoding="utf-8")

print(f"Collected {len(dirs)} folder names -> {out}")
for w in dirs[:10]:
    print(" ", w)