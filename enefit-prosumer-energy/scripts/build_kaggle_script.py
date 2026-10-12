"""Build kaggle_submit/enefit_blend.py: one self-contained script from the tested modules, with experiment-only code pruned (AST-level)."""
import ast, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = {  # module -> top-level names to delete ; ("Class", "method") pairs for methods
    "gbdt": {"add_profile", "v2_features", "PARAMS_TW", "HistoryCallback", "fit_group", "SAME_SEASON", "tune_rounds", "run_variant"},
    "nn": {("Runner", m) for m in ("_take", "_predict_raw", "predict_resident", "evaluate", "_mae_resident", "train")},
    "data": {"roles"},
}

def prune(name, src):
    t = ast.parse(src)
    drop = DROP.get(name, set())
    body = []
    for node in t.body:
        nm = getattr(node, "name", None)
        if nm in drop:
            continue
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id in drop for x in node.targets):
            continue
        if isinstance(node, ast.ClassDef):
            node.body = [m for m in node.body if not (isinstance(m, ast.FunctionDef) and (node.name, m.name) in drop)]
        body.append(node)
    t.body = body
    for node in ast.walk(t):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return ast.unparse(t)

parts = []
for m in ("data", "gbdt", "nn", "nn_ens", "serve"):
    code = prune(m, open(f"{ROOT}/src/enefit13/{m}.py").read())
    assert "'''" not in code, m
    parts.append(f"open(os.path.join(PKG, '{m}.py'), 'w').write(r'''{code}\n''')")
script = "\n".join([
    "# Enefit - blend of V1 LightGBM-L1 (capacity-relative, lags>=48h) + V3 TiDE (capacity-relative) + V4 TiDE (window-normalised).",
    "# Self-contained: writes its package, fits on all labelled data, then serves the time-series API one day at a time.",
    "import os, sys, time", "PKG = '/kaggle/working/pkg/enefit13'", "os.makedirs(PKG, exist_ok=True)",
    "open(os.path.join(PKG, '__init__.py'), 'w').write('')", *parts,
    "sys.path.insert(0, '/kaggle/working/pkg')", "import torch",
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)",
    "from enefit13 import data as D, serve as S", "import enefit",
    "ROOT = '/kaggle/input/predict-energy-behavior-of-prosumers'", "t0 = time.time()",
    "server = S.train_final(D.CsvStore(ROOT), quick=False, log=lambda m: print(m, flush=True))",
    "print('fit done in', round(time.time() - t0), 's', flush=True)",
    "env = enefit.make_env()", "n = S.run_api(env, server, log=lambda m: print(m, flush=True))",
    "print('served days:', n, 'total', round(time.time() - t0), 's', flush=True)", ""])
os.makedirs(f"{ROOT}/kaggle_submit", exist_ok=True)
open(f"{ROOT}/kaggle_submit/enefit_blend.py", "w").write(script)
bad = [i + 1 for i, l in enumerate(script.split("\n")) if re.search(r"""f'[^'\n]*\{[^}'\n]*'[^'\n]*'""", l)]
compile(script, "enefit_blend.py", "exec")
print("bytes", len(script), "lines", script.count("\n"), "nested-quote f-strings:", bad)
