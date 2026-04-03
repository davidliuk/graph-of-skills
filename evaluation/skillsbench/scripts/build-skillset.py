#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SKILLSBENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILLSBENCH_ROOT.parents[1]
SKILLSETS_ROOT = REPO_ROOT / "data" / "skillsets"
BASELINE_ROOT = SKILLSBENCH_ROOT / "all_skills"
DEFAULT_LARGE_ROOT = SKILLSBENCH_ROOT.parent.parent / "skills_1000"


def copy_skill(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def list_skills(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").exists()])


def build_skillset(target_name: str, target_size: int, large_root: Path) -> Path:
    baseline = list_skills(BASELINE_ROOT)
    if target_size < len(baseline):
        raise ValueError(f"target_size={target_size} is smaller than baseline size={len(baseline)}")

    large_skills = {p.name: p for p in list_skills(large_root)}
    extras = [name for name in sorted(large_skills) if name not in {p.name for p in baseline}]
    needed = target_size - len(baseline)
    selected_extra_names = extras[:needed]
    if len(selected_extra_names) < needed:
        raise ValueError(f"only found {len(selected_extra_names)} extra skills, need {needed}")

    out = SKILLSETS_ROOT / target_name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for skill in baseline:
        copy_skill(skill, out / skill.name)
    for name in selected_extra_names:
        copy_skill(large_skills[name], out / name)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a named skillsbench skillset")
    parser.add_argument("target_name", help="Output skillset directory name, e.g. skills_200")
    parser.add_argument("target_size", type=int, help="Desired number of skills")
    parser.add_argument(
        "--large-root",
        type=Path,
        default=DEFAULT_LARGE_ROOT,
        help="Large source library used to fill beyond baseline all_skills",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_skillset(args.target_name, args.target_size, args.large_root.resolve())
    count = sum(1 for _ in out.rglob("SKILL.md"))
    print(f"built {out} with {count} skills")


if __name__ == "__main__":
    main()
