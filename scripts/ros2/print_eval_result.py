#!/usr/bin/env python3
"""Pretty-print a pick-task evaluation result read from stdin.

    cat eval_ep02.json | scripts/ros2/print_eval_result.py ep02
"""
import json
import sys


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "eval"
    try:
        d = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[ERROR] could not parse the result json: {e}", file=sys.stderr)
        return 1
    episodes = d.get("episodes", 0)
    success = d.get("success", 0)
    rate = 100.0 * d.get("success_rate", 0.0)
    print(f"[RESULT] {tag}: {success}/{episodes} = {rate:.1f}%")
    for name, v in sorted(d.get("per_object", {}).items()):
        print(f"   {name:16s} {v['ok']:3d}/{v['n']:3d}")
    failures = [r for r in d.get("results", []) if not r.get("success")]
    if failures:
        print(f"   {len(failures)} failure(s); grasp condition at the end of each:")
        for r in failures[:8]:
            g = r.get("grasp_condition", {})
            print(
                f"     ep{r['episode']:<3d} {r['object']:16s} dxy={g.get('dxy')} "
                f"vertical_ok={g.get('vertical_ok')} hand_motor={g.get('hand_motor')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
