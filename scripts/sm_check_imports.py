"""Import every external module used by envs/, lib/, main/ and scripts/sm_*.py.
Usage: python scripts/sm_check_imports.py [repo_root]"""
import ast, importlib, pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
files = [p for d in ("envs", "lib", "main") for p in (root / d).rglob("*.py")] + sorted((root / "scripts").glob("sm_*.py"))
local = {p.name for p in root.iterdir()} | {p.stem for p in root.rglob("*.py") if "third_party" not in p.parts}
mods = {}
for f in files:
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError as e:
        print(f"PARSE {f.relative_to(root)}: {e}"); continue
    for n in ast.walk(tree):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
        for m in names:
            top = m.split(".")[0]
            if top not in local and top not in sys.stdlib_module_names:
                mods.setdefault(m, set()).add(str(f.relative_to(root)))
bad = 0
for m in sorted(mods):
    try:
        importlib.import_module(m); print(f"ok    {m}")
    except Exception as e:
        bad += 1; print(f"FAIL  {m}  ({type(e).__name__}: {str(e)[:80]})  used in {sorted(mods[m])[:3]}")
print(f"\n{len(files)} files, {len(mods)} external modules, {bad} failed")
