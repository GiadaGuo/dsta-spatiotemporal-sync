'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-22 15:11:20
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-22 15:11:32
FilePath: /dsta-spatiotemporal-sync/scripts/inspect_spatial.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from pathlib import Path
import json

ROOT = Path("data/raw/V2X-Seq-SPD-Example")


def show_json(path):
    print("\n" + "=" * 70)
    print("FILE:", path)

    if not path.exists():
        print("NOT FOUND")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("Type:", type(data).__name__)

    if isinstance(data, list):
        print("Records:", len(data))
        if data:
            print(json.dumps(
                data[0], ensure_ascii=False, indent=2
            )[:4000])

    elif isinstance(data, dict):
        print(json.dumps(
            data, ensure_ascii=False, indent=2
        )[:4000])


def main():
    for side in ["vehicle-side", "infrastructure-side"]:
        print(f"\n\n######## {side} ########")

        root = ROOT / side

        for category in ["label", "calib"]:
            folder = root / category
            files = sorted(folder.rglob("*.json"))

            print(f"\n{category}: {len(files)} JSON files")

            # 每类文件按子目录分别抽样，避免漏掉标定类型
            groups = {}
            for path in files:
                relative_parent = str(path.parent.relative_to(folder))
                groups.setdefault(relative_parent, path)

            for path in groups.values():
                show_json(path)


if __name__ == "__main__":
    main()
