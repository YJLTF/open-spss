"""AST 检查:所有 _run_* runner 中 res 赋值先于使用。"""
import ast, pathlib, sys

bad = []
for f in pathlib.Path("open_spss/models").glob("*.py"):
    tree = ast.parse(f.read_text(encoding="utf-8"))
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("_run")):
            continue
        assigned = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "res" for t in node.targets):
                assigned = True
                break
        uses = any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "res"
                   for n in ast.walk(fn))
        if uses and not assigned:
            bad.append(f"{f.name}:{fn.lineno} {fn.name}")
print("\n".join(bad) if bad else "all runners OK")
sys.exit(1 if bad else 0)
