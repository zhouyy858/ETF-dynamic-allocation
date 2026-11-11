# -*- coding: utf-8 -*-
"""v41 future-function audit wrapper.

Reuses the proven v40 truncation/perturbation/09:00 audit, but substitutes the
v41 config for the v40 slot. v40, v37 and v30 remain the historical controls.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lookahead_v40 as audit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V41 = os.path.join(ROOT, "references", "final_cfg_v41.json")
_load_cfg = audit.load_cfg


def load_cfg(ver):
    return json.load(open(V41)) if ver == "v40" else _load_cfg(ver)


audit.load_cfg = load_cfg
audit.OUT = os.path.join(ROOT, "out", "audit_lookahead_v41")
os.makedirs(audit.OUT, exist_ok=True)


if __name__ == "__main__":
    print("[note] 继承审计中的 v40 标签在本轮代表 v41 候选；JSON 键已由包装层改写为 v41。", flush=True)
    rc = audit.main()
    src = os.path.join(audit.OUT, "audit_lookahead_v40.json")
    dst = os.path.join(audit.OUT, "audit_lookahead_v41.json")
    if os.path.exists(src):
        import json as _json

        def _rename(obj):
            if isinstance(obj, dict):
                return {("v41" + k[3:] if k == "v40" or k.startswith("v40_") else k): _rename(v)
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
