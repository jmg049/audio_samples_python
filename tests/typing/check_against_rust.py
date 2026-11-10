"""Check the hand-written stubs against the stubs generated from the Rust bindings.

``scripts/stubs.sh`` writes ``stubs/generated/``: the stubs pyo3 derives from the Rust code (names,
parameters, defaults, and the Python types of the Rust argument and return types or of the
``#[pyo3(signature = (x: "T") -> "R")]`` annotations). The hand-written stubs in
``audio_samples/`` add what Rust cannot say: overloads, the generic ``AudioSamples``, dtype rules.
This script fails when the two disagree:

* an item of the bindings (function, class, method, property, constant) is missing from the stubs,
  or the stubs have an item the bindings do not;
* a parameter differs in name, kind (positional, keyword-only, ...) or in having a default;
* a stub type is not compatible with the Rust type: every overload's parameter types must be
  accepted by the Rust type and every overload's return type must be a subtype of it. mypy and
  pyright prove this on a generated file of assignments (``target/stubcheck/agreement.py``);
* the Rust side gives no type (``Any``, or ``Incomplete`` for a Rust type without a Python type)
  where the stub states one, and the item is not listed in ``tests/typing/untyped_bindings.txt``.
  That list may only shrink: new bindings must carry their types in Rust.

Modules the bindings do not declare are skipped: ``audio_samples.spectrograms`` (compiled from the
spectrograms crate) and ``audio_samples.exceptions`` (Python source run at import); pure Python
modules have no generated stub. mypy.stubtest checks those against the runtime.

    python tests/typing/check_against_rust.py
    python tests/typing/check_against_rust.py --update-untyped   # rewrite the list (only shrinks)
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "stubs" / "generated"
HAND = ROOT / "audio_samples"
UNTYPED = ROOT / "tests" / "typing" / "untyped_bindings.txt"
WORK = ROOT / "target" / "stubcheck"

#: Modules that exist at run time but are not declared by the bindings crate.
NOT_GENERATED = {"audio_samples.spectrograms", "audio_samples.exceptions"}
#: Generic classes of the hand stubs and their class-scoped type variables.
CLASS_TYPEVARS = {"audio_samples.AudioSamples": ("_S_co",)}
UNTYPED_NAMES = {"typing.Any", "_typeshed.Incomplete", "typing_extensions.Any"}
CLASS_NAMES: dict[str, str] = {}


# --------------------------------------------------------------------------------------------
# Reading stubs
# --------------------------------------------------------------------------------------------


@dataclass
class Param:
    name: str
    kind: str  # posonly, pos, vararg, kwonly, kwarg
    default: bool
    ann: str | None  # fully qualified annotation source, None if absent


@dataclass
class Sig:
    params: list[Param]
    ret: str | None
    decorators: set[str]
    line: int


@dataclass
class Item:
    kind: str  # function, class, property, attribute
    sigs: list[Sig] = field(default_factory=list)  # one per overload
    members: dict[str, Item] = field(default_factory=dict)  # for classes
    ann: str | None = None  # attributes
    reexport: str | None = None  # "from x import y" in a stub: the qualified target
    settable: bool = False


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class Qualifier(ast.NodeTransformer):
    """Rewrite names in an annotation to fully qualified ones; strings in Literal[...] stay."""

    def __init__(self, scope: dict[str, str], extra: dict[str, str] | None = None):
        self.scope = scope
        self.extra = extra or {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        target = self.extra.get(node.id) or self.scope.get(node.id)
        if target is None:
            return node
        return ast.parse(target, mode="eval").body

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.value = self.visit(node.value)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node.value = self.visit(node.value)
        head = node.value
        name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", "")
        if name != "Literal":
            node.slice = self.visit(node.slice)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # A forward reference written as a string.
        if isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return node
            return self.visit(inner)
        return node


def qualify(node: ast.expr | None, scope: dict[str, str], extra: dict[str, str] | None = None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        copy = ast.parse(node.value, mode="eval").body
    else:
        copy = ast.parse(ast.unparse(node), mode="eval").body
    return ast.unparse(Qualifier(scope, extra).visit(copy))


def scope_of(tree: ast.Module, mod: str, is_pkg: bool) -> dict[str, str]:
    scope: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    scope[a.asname] = a.name
                else:
                    scope[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            base = absolute(node, mod, is_pkg)
            for a in node.names:
                scope[a.asname or a.name] = f"{base}.{a.name}"
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            scope[node.name] = f"{mod}.{node.name}"
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    scope[t.id] = f"{mod}.{t.id}"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            scope[node.target.id] = f"{mod}.{node.target.id}"
    return scope


def absolute(node: ast.ImportFrom, mod: str, is_pkg: bool) -> str:
    if not node.level:
        return node.module or ""
    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def decorator_names(fn: ast.FunctionDef) -> set[str]:
    out = set()
    for d in fn.decorator_list:
        if isinstance(d, ast.Name):
            out.add(d.id)
        elif isinstance(d, ast.Attribute):
            out.add(d.attr)
    return out


def signature(fn: ast.FunctionDef, scope, extra) -> Sig:
    a = fn.args
    params: list[Param] = []
    pos = a.posonlyargs + a.args
    defaults = [False] * (len(pos) - len(a.defaults)) + [True] * len(a.defaults)
    for i, p in enumerate(pos):
        kind = "posonly" if i < len(a.posonlyargs) else "pos"
        params.append(Param(p.arg, kind, defaults[i], qualify(p.annotation, scope, extra)))
    if a.vararg:
        params.append(Param(a.vararg.arg, "vararg", False, qualify(a.vararg.annotation, scope, extra)))
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append(Param(p.arg, "kwonly", d is not None, qualify(p.annotation, scope, extra)))
    if a.kwarg:
        params.append(Param(a.kwarg.arg, "kwarg", False, qualify(a.kwarg.annotation, scope, extra)))
    return Sig(params, qualify(fn.returns, scope, extra), decorator_names(fn), fn.lineno)


def read_body(body, scope, extra, owner: str | None) -> dict[str, Item]:
    items: dict[str, Item] = {}
    for node in body:
        if isinstance(node, ast.FunctionDef):
            decs = decorator_names(node)
            if any(d in ("setter", "deleter") for d in decs):
                if node.name in items:
                    items[node.name].settable = True
                continue
            sig = signature(node, scope, extra)
            if "property" in decs:
                items[node.name] = Item("property", [sig])
                continue
            item = items.setdefault(node.name, Item("function"))
            if "overload" in decs or not item.sigs:
                item.sigs.append(sig)
            else:  # an implementation after overloads
                pass
        elif isinstance(node, ast.ClassDef) and owner is None:
            cls_q = scope[node.name]
            ex = dict(extra)
            for tv in CLASS_TYPEVARS.get(cls_q, ()):
                ex[tv] = "typing.Any"
            ex["Self"] = cls_q
            items[node.name] = Item("class", members=read_body(node.body, scope, ex, node.name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            items[node.target.id] = Item("attribute", ann=qualify(node.annotation, scope, extra))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    items[t.id] = Item("attribute")
    return items


#: Names the ``#[pyo3(signature = (x: "T"))]`` annotations may use; generated stubs import nothing for them.
ANNOTATION_NAMES = {
    "np": "numpy", "npt": "numpy.typing", "NDArray": "numpy.typing.NDArray", "ArrayLike": "numpy.typing.ArrayLike",
    "DTypeLike": "numpy.typing.DTypeLike", "Any": "typing.Any", "Optional": "typing.Optional",
    "Union": "typing.Union", "Literal": "typing.Literal", "Callable": "typing.Callable",
    "Sequence": "typing.Sequence", "Iterator": "typing.Iterator", "Mapping": "typing.Mapping",
    "Iterable": "typing.Iterable", "Tuple": "typing.Tuple", "List": "typing.List", "Dict": "typing.Dict",
    "Path": "pathlib.Path", "BinaryIO": "typing.BinaryIO",
}


def read_tree(root: Path, base: str = "audio_samples", reexports: bool = True) -> dict[str, tuple[dict[str, Item], dict[str, str], bool]]:
    """{module: (items, scope, incomplete)} for every .pyi under root/base.

    With ``reexports``, ``from x import y as y`` (and any import in the package root) counts as an
    item of the module. Generated stubs import only what their annotations use.
    """
    out = {}
    for path in sorted((root / base).rglob("*.pyi")):
        mod = module_name(path, root)
        is_pkg = path.name == "__init__.pyi"
        tree = ast.parse(path.read_text())
        scope = scope_of(tree, mod, is_pkg)
        if not reexports:
            scope = {**ANNOTATION_NAMES, **CLASS_NAMES, **scope}
        items = read_body(tree.body, scope, {}, None)
        for node in tree.body if reexports else ():
            if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                src = absolute(node, mod, is_pkg)
                for a in node.names:
                    if a.asname == a.name or (a.asname is None and mod == base):
                        items.setdefault(a.asname or a.name, Item("reexport", reexport=f"{src}.{a.name}"))
        incomplete = "__getattr__" in items
        items.pop("__getattr__", None)
        out[mod] = (items, scope, incomplete)
    return out


# --------------------------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------------------------


def private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def untyped(ann: str | None) -> bool:
    return ann is None or ann in UNTYPED_NAMES


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.untyped: set[str] = set()
        self.checks: list[tuple[str, str, str, str]] = []  # (label, from-type, to-type, why)

    def error(self, msg: str):
        self.errors.append(msg)


INPLACE = {f"__i{op}__" for op in ("add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "pow",
                                     "lshift", "rshift", "and", "xor", "or")}
SKIP_DUNDERS = {"__new__", "__init__"}


def compare_callable(rep: Report, path: str, g: Item, h: Item, method: bool):
    gsig = g.sigs[0]
    if h.kind == "reexport":
        return
    if g.kind == "property" and h.kind == "attribute":
        # `x: T` in a stub declares a readable and writable attribute.
        if not g.settable:
            rep.error(f"{path}: read-only in the bindings, a writable attribute in the stubs (use @property)")
        if untyped(gsig.ret):
            if not untyped(h.ann):
                rep.untyped.add(f"{path} -> return")
        else:
            rep.checks.append((f"{path} (attribute type)", h.ann or "typing.Any", gsig.ret, "return"))
        return
    if h.kind != g.kind and not (g.kind == "function" and h.kind == "function"):
        rep.error(f"{path}: a {g.kind} in the bindings but a {h.kind} in the stubs")
        return
    if g.kind == "property":
        hsig = h.sigs[0]
        if untyped(gsig.ret):
            if not untyped(hsig.ret):
                rep.untyped.add(f"{path} -> return")
        else:
            rep.checks.append((f"{path} (property type)", hsig.ret or "typing.Any", gsig.ret, "return"))
        if g.settable and not h.settable:
            rep.error(f"{path}: settable in the bindings, read-only in the stubs")
        if h.settable and not g.settable:
            rep.error(f"{path}: read-only in the bindings, settable in the stubs")
        return
    gkind = {"staticmethod", "classmethod"} & gsig.decorators
    implicit = {"classmethod"} if path.rsplit(".", 1)[-1] in ("__class_getitem__", "__init_subclass__") else set()
    gp = gsig.params[1:] if method and not ("staticmethod" in gkind) else gsig.params
    for n, hsig in enumerate(h.sigs, 1):
        tag = f"{path}" + (f" (overload {n})" if len(h.sigs) > 1 else "")
        hkind = ({"staticmethod", "classmethod"} & hsig.decorators) | implicit
        if gkind != hkind:
            rep.error(f"{tag}: {'/'.join(sorted(gkind)) or 'instance method'} in the bindings, "
                      f"{'/'.join(sorted(hkind)) or 'instance method'} in the stubs (line {hsig.line})")
            continue
        hp = hsig.params[1:] if method and "staticmethod" not in hkind else hsig.params
        if not same_shape(gp, hp, overloaded=len(h.sigs) > 1):
            rep.error(f"{tag}: parameters differ (stub line {hsig.line})\n"
                      f"      bindings: {fmt_params(gp)}\n      stubs:    {fmt_params(hp)}")
            continue
        hby = {p.name: p for p in hp if p.kind == "kwonly"}
        pairs = [(g, h) for g, h in zip([p for p in gp if p.kind != "kwonly" and p.kind != "kwarg"],
                                         [p for p in hp if p.kind != "kwonly" and p.kind != "kwarg"])]
        pairs += [(g, hby[g.name]) for g in gp if g.kind == "kwonly"]
        for gpar, hpar in pairs:
            if untyped(gpar.ann):
                if not untyped(hpar.ann):
                    rep.untyped.add(f"{path}({gpar.name})")
                continue
            rep.checks.append((f"{tag}: parameter {gpar.name}", hpar.ann or "typing.Any", gpar.ann, "param"))
        gret = gsig.ret
        if gret == "None" and path.rsplit(".", 1)[-1] in INPLACE:
            # An in-place operator returning nothing in Rust makes Python return the object.
            gret = path.rsplit(".", 1)[0]
        if untyped(gret):
            if not untyped(hsig.ret):
                rep.untyped.add(f"{path} -> return")
        else:
            rep.checks.append((f"{tag}: return", hsig.ret or "typing.Any", gret, "return"))


def same_shape(gp: list[Param], hp: list[Param], overloaded: bool) -> bool:
    """Positional parameters in order, keyword-only ones by name.

    Positional-only names do not matter. An overload may require a parameter the binding gives a
    default (overloads select on such parameters), not the reverse. Where the binding takes
    ``**kwargs``, the stub may name keyword-only parameters instead.
    """
    def ok(g: Param, h: Param) -> bool:
        if h.default and not g.default:
            return False
        return not (g.default and not h.default and not overloaded)

    gpos = [p for p in gp if p.kind in ("posonly", "pos", "vararg")]
    hpos = [p for p in hp if p.kind in ("posonly", "pos", "vararg")]
    if len(gpos) != len(hpos):
        return False
    for g, h in zip(gpos, hpos):
        if g.kind != h.kind or (g.kind != "posonly" and g.name != h.name) or not ok(g, h):
            return False
    gkw = {p.name: p for p in gp if p.kind == "kwonly"}
    hkw = {p.name: p for p in hp if p.kind == "kwonly"}
    gkwargs = any(p.kind == "kwarg" for p in gp)
    hkwargs = any(p.kind == "kwarg" for p in hp)
    if hkwargs and not gkwargs:
        return False
    for name, g in gkw.items():
        if name not in hkw or not ok(g, hkw[name]):
            return False
    return not (set(hkw) - set(gkw)) or gkwargs


def fmt_params(ps: list[Param]) -> str:
    out = []
    for p in ps:
        s = {"vararg": "*", "kwarg": "**"}.get(p.kind, "") + p.name
        if p.default:
            s += "=..."
        out.append(f"{s}[{p.kind}]" if p.kind in ("posonly", "kwonly") else s)
    return "(" + ", ".join(out) + ")"


def compare(gen, hand) -> Report:
    rep = Report()
    for mod, (gitems, _gscope, gincomplete) in gen.items():
        if mod not in hand:
            rep.error(f"{mod}: module in the bindings has no stub")
            continue
        hitems, _hs, _hi = hand[mod]
        for name, g in gitems.items():
            if private(name):
                continue
            path = f"{mod}.{name}"
            h = hitems.get(name)
            if h is None:
                rep.error(f"{path}: in the bindings, missing from the stubs")
                continue
            if g.kind == "class":
                if h.kind == "reexport":
                    continue
                if h.kind != "class":
                    rep.error(f"{path}: a class in the bindings, a {h.kind} in the stubs")
                    continue
                for mname, gm in g.members.items():
                    if private(mname):
                        continue
                    mpath = f"{path}.{mname}"
                    hm = h.members.get(mname)
                    if hm is None:
                        if mname.startswith("__") and mname.endswith("__"):
                            continue  # pyo3 adds slots such as __eq__/__ne__/__hash__ freely
                        rep.error(f"{mpath}: in the bindings, missing from the stubs")
                        continue
                    if mname in SKIP_DUNDERS:
                        continue
                    if gm.kind == "attribute":
                        continue
                    compare_callable(rep, mpath, gm, hm, method=True)
                for mname, hm in h.members.items():
                    if mname not in g.members and not private(mname) and not (mname.startswith("__") and mname.endswith("__")):
                        rep.error(f"{path}.{mname}: in the stubs, not in the bindings")
            elif g.kind == "attribute":
                continue
            else:
                compare_callable(rep, path, g, h, method=False)
        if not gincomplete:
            for name, h in hitems.items():
                if name.startswith("_") or name in gitems or h.kind in ("reexport", "attribute"):
                    continue
                rep.error(f"{mod}.{name}: in the stubs, not in the bindings")
    return rep


# --------------------------------------------------------------------------------------------
# Type compatibility, proven by mypy and pyright
# --------------------------------------------------------------------------------------------


def modules_in(expr: str) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(expr, mode="eval")):
        if isinstance(node, ast.Attribute):
            parts = []
            n: ast.AST = node
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
                parts.reverse()
                for i in range(1, len(parts)):
                    out.add(".".join(parts[:i]))
    return out


def write_agreement(rep: Report, known_modules: set[str]) -> Path:
    mods = set()
    for _label, a, b, _why in rep.checks:
        mods |= modules_in(a) | modules_in(b)
    lines = ["# Generated by tests/typing/check_against_rust.py. Each function states that a stub type",
             "# is compatible with the type the Rust bindings give; mypy and pyright must accept all.",
             "from __future__ import annotations", "import typing", ""]
    for m in sorted(mods & known_modules | {"typing", "numpy", "numpy.typing"}):
        lines.append(f"import {m}")
    lines.append("")
    index = {}
    for i, (label, frm, to, why) in enumerate(rep.checks):
        lines.append(f"# {label}")
        # A stub parameter type must be accepted by Rust; a stub return type must be one Rust returns.
        lines.append(f"def c{i}(x: {frm}) -> {to}:")
        lines.append("    return x")
        index[len(lines)] = (label, frm, to, why)  # 1-based line of the return, where errors point
        lines.append("")
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "agreement.py"
    out.write_text("\n".join(lines) + "\n")
    (WORK / "agreement_index.json").write_text(json.dumps({str(k): v for k, v in index.items()}))
    return out


def tool(name: str) -> str:
    exe = Path(sys.executable).parent / name
    return str(exe) if exe.exists() else (shutil.which(name) or name)


def run_checkers(path: Path) -> list[str]:
    index = {int(k): v for k, v in json.loads((WORK / "agreement_index.json").read_text()).items()}
    failures: dict[int, list[str]] = {}
    proc = subprocess.run([sys.executable, "-m", "mypy", "--config-file", str(ROOT / "pyproject.toml"),
                           "--no-incremental", "--no-error-summary", "--hide-error-context", str(path)],
                          cwd=ROOT, capture_output=True, text=True)
    other = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 3)
        if len(parts) >= 4 and parts[0].endswith("agreement.py") and " error" in parts[2]:
            ln = int(parts[1])
            if ln in index:
                failures.setdefault(ln, []).append("mypy: " + parts[3].strip())
            else:
                other.append("mypy: " + line)
    proc = subprocess.run([tool("pyright"), "--outputjson", "--pythonpath", sys.executable, str(path)],
                          cwd=ROOT, capture_output=True, text=True)
    for d in json.loads(proc.stdout).get("generalDiagnostics", []):
        if d["severity"] != "error":
            continue
        ln = d["range"]["start"]["line"] + 1
        if ln in index:
            failures.setdefault(ln, []).append("pyright: " + d["message"].splitlines()[0])
        else:
            other.append(f"pyright: line {ln}: {d['message']}")
    msgs = []
    for ln, why in sorted(failures.items()):
        label, frm, to, kind = index[ln]
        what = ("the stub accepts a type the Rust binding rejects" if kind == "param"
                else "the stub promises a type the Rust binding does not return")
        msgs.append(f"{label}: {what}\n      stub: {frm}\n      Rust: {to}\n      " + "\n      ".join(why))
    if other:
        msgs.append("errors outside the comparisons (a type that cannot be resolved?):\n  " + "\n  ".join(other))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-untyped", action="store_true")
    opts = ap.parse_args()
    if not GEN.exists():
        print("stubs/generated/ is missing. Run: scripts/stubs.sh")
        return 1
    hand = read_tree(ROOT)
    # A pyclass declared without `module = "..."` appears in generated stubs by bare name.
    seen: dict[str, list[str]] = {}
    for mod, (items, _s, _i) in hand.items():
        for name, it in items.items():
            if it.kind == "class":
                seen.setdefault(name, []).append(f"{mod}.{name}")
    CLASS_NAMES.update({n: q[0] for n, q in seen.items() if len(q) == 1})
    gen = {m: v for m, v in read_tree(GEN, reexports=False).items() if m not in NOT_GENERATED}
    rep = compare(gen, hand)
    known = set(hand) | set(gen) | {"numpy", "numpy.typing", "typing", "typing_extensions", "collections.abc",
                                    "builtins", "pathlib", "os", "datetime", "_typeshed"}
    agreement = write_agreement(rep, known)
    rep.errors += run_checkers(agreement)

    listed = set()
    if UNTYPED.exists():
        listed = {l.strip() for l in UNTYPED.read_text().splitlines() if l.strip() and not l.startswith("#")}
    if opts.update_untyped:
        new = sorted(rep.untyped - listed)
        if new and UNTYPED.exists():
            print("Not written: these bindings give no type in Rust. Add the types in Rust "
                  "(#[pyo3(signature = (x: \"T\") -> \"R\")] or INPUT_TYPE/OUTPUT_TYPE) instead:")
            print("".join(f"  {n}\n" for n in new))
            return 1
        UNTYPED.write_text("# Bindings whose Rust side gives no Python type for a parameter or the return\n"
                           "# (Any or Incomplete in stubs/generated), so their stub types cannot be checked\n"
                           "# against Rust. This list may only shrink; see tests/typing/check_against_rust.py.\n"
                           + "".join(f"{n}\n" for n in sorted(rep.untyped)))
        print(f"{UNTYPED.relative_to(ROOT)}: {len(rep.untyped)} entries")
    else:
        for n in sorted(rep.untyped - listed):
            rep.error(f"{n}: the Rust binding gives no type here, so the stub's type is unchecked. "
                      "Annotate it in Rust: #[pyo3(signature = (x: \"T\") -> \"R\")].")
        gone = sorted(listed - rep.untyped)
        if gone:
            rep.error("typed in Rust now (or removed); delete from tests/typing/untyped_bindings.txt "
                      "(python tests/typing/check_against_rust.py --update-untyped):\n  " + "\n  ".join(gone))
    if rep.errors:
        print("The stubs disagree with the Rust bindings (stubs/generated):\n")
        print("\n".join(f"- {e}" for e in rep.errors))
        print(f"\n{len(rep.errors)} problems. Fix the stubs in audio_samples/ (or the Rust types), "
              "then run scripts/stubs.sh.")
        return 1
    print(f"stubs agree with the bindings: {len(rep.checks)} type comparisons, "
          f"{len(rep.untyped)} untyped in Rust (listed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
