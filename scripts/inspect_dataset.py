from pathlib import Path
import json

ROOT = Path("data/raw")

def main():
    files = sorted(ROOT.rglob("data_info.json"))

    if not files:
        print("未找到 data_info.json，请检查解压目录")
        return

    for path in files:
        print(f"\n{'=' * 60}")
        print(f"文件: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"JSON 类型: {type(data).__name__}")

        if isinstance(data, list):
            print(f"记录数: {len(data)}")
            if not data:
                continue
            record = data[0]
        elif isinstance(data, dict):
            print(f"顶层字段: {list(data.keys())}")
            record = data
        else:
            continue

        print("\n第一条记录:")
        print(json.dumps(
            record,
            ensure_ascii=False,
            indent=2
        )[:6000])

if __name__ == "__main__":
    main()
