# -*- coding: utf-8 -*-
"""v43 future-function audit wrapper (other versions callable via args).

Reuses the proven v40 truncation/perturbation/09:00 audit, but substitutes the
selected config (default v43) for the v40 slot. v41, v40, v37 and v30 remain the
historical controls.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lookahead_v40 as audit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V43 = os.path.join(ROOT, "references", "final_cfg_v43.json")
CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else V43
TAG = sys.argv[2] if len(sys.argv) > 2 else "v43"
_load_cfg = audit.load_cfg


def load_cfg(ver):
    return json.load(open(CFG_PATH)) if ver == "v40" else _load_cfg(ver)


audit.load_cfg = load_cfg
audit.OUT = os.path.join(ROOT, "out", f"audit_lookahead_{TAG}")
os.makedirs(audit.OUT, exist_ok=True)


if __name__ == "__main__":
    print(f"[note] 继承审计中的 v40 标签在本轮代表 {TAG} 候选；JSON 键已由包装层改写为 {TAG}。", flush=True)
    extra = sys.argv[3:]
    rc = audit.main(n_random=10, n_friday=5) if "--quick" in extra else audit.main()
    src = os.path.join(audit.OUT, "audit_lookahead_v40.json")
    dst = os.path.join(audit.OUT, f"audit_lookahead_{TAG}.json")
    if os.path.exists(src):
        import json as _json

        def _rename(obj):
            if isinstance(obj, dict):
                return {(TAG + k[3:] if k == "v40" or k.startswith("v40_") else k): _rename(v)
                        for k, v in obj.items()}
            if isinstance(obj, list):
                return [_rename(v) for v in obj]
            return obj

        with open(src, encoding="utf-8") as fh:
            report = _json.load(fh)
        with open(dst, "w", encoding="utf-8") as fh:
            _json.dump(_rename(report), fh, ensure_ascii=False, indent=1)
        os.remove(src)
        print(f"[ok] {dst}", flush=True)
    raise SystemExit(rc)
