"""Find librosa usage in Python files and notebooks and say what replaces it.

    python -m audio_samples.migrate PATH [PATH ...]
    python -m audio_samples.migrate --format json src notebooks
    python -m audio_samples.migrate --fail-on blocking src

Each path is a ``.py`` file, an ``.ipynb`` notebook or a directory searched
recursively. For every librosa call the checker prints where it is, the call,
the ``audio_samples`` equivalent, how closely it matches, and what to watch for
in that call: keywords that change, options that are not supported, defaults
that differ and that the call relies on, and shapes that differ.

The files are parsed with :mod:`ast` and never changed or executed; librosa does
not have to be installed.

Exit status:

- 0: nothing to report under ``--fail-on``.
- 1: findings under ``--fail-on``. With the default ``any``, any librosa usage.
  With ``blocking``, usage with no equivalent or with an option the equivalent
  rejects. ``never`` never exits 1.
- 2: a path could not be read or parsed. This takes priority over 1.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from audio_samples._migrate_table import (
    ALIASES,
    EQUIVALENT,
    NOT_PROVIDED,
    REMOVED,
    TABLE,
    WITH_DIFFERENCES,
    Absorbed,
    Default,
    Entry,
    Form,
    Only,
    Renamed,
    Unsupported,
)

__all__ = ["Finding", "Watch", "check_paths", "check_source", "main"]

UNKNOWN = "unknown"

# Directories that hold other people's installed code or build output.
_SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "node_modules",
        "site-packages",
        "target",
    }
)


@dataclass(frozen=True)
class Watch:
    """One thing to watch for in a call."""

    kind: str
    message: str
    keyword: str | None = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "keyword": self.keyword, "message": self.message}


# Watch kinds.
KEYWORD_CHANGED = "keyword changed"
UNSUPPORTED = "unsupported option"
DEFAULT_DIFFERS = "default differs"
SHAPE_DIFFERS = "shape differs"
BEHAVIOUR = "behaviour"
NOT_CHECKED = "not checked"


@dataclass(frozen=True)
class Finding:
    """One use of librosa in a source file."""

    path: str
    line: int
    column: int
    cell: int | None
    librosa: str
    call: str
    is_call: bool
    equivalent: tuple[str, ...]
    match: str
    note: str
    watch: tuple[Watch, ...] = field(default=())

    @property
    def location(self) -> str:
        if self.cell is None:
            return f"{self.path}:{self.line}"
        return f"{self.path}:cell{self.cell}:{self.line}"

    @property
    def blocking(self) -> bool:
        """No equivalent exists, or the equivalent rejects something this call does."""
        if self.match in (NOT_PROVIDED, UNKNOWN):
            return True
        return any(w.kind == UNSUPPORTED for w in self.watch)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "cell": self.cell,
            "librosa": self.librosa,
            "call": self.call,
            "is_call": self.is_call,
            "equivalent": list(self.equivalent),
            "match": self.match,
            "note": self.note,
            "blocking": self.blocking,
            "watch": [w.as_dict() for w in self.watch],
        }


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------


def _is_librosa(dotted: str) -> bool:
    return dotted == "librosa" or dotted.startswith("librosa.")


def _target_names(target: ast.expr) -> Iterator[str]:
    """Names bound by an assignment target."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _argument_names(args: ast.arguments) -> Iterator[str]:
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        yield arg.arg
    if args.vararg:
        yield args.vararg.arg
    if args.kwarg:
        yield args.kwarg.arg


class _Scope:
    """Names bound in one scope, each to a librosa path or to something else.

    Binding is flow-insensitive. A name that any import in the scope binds to
    librosa counts as librosa, so ``try: import librosa`` with ``except
    ImportError: librosa = None`` still resolves. A name only ever bound to
    something else hides a librosa name of an enclosing scope.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.librosa: dict[str, str] = {}
        self.other: set[str] = set()
        self.star = False

    def bind_other(self, name: str) -> None:
        self.other.add(name)

    def bind_librosa(self, name: str, dotted: str) -> None:
        self.librosa[name] = dotted


class _Collector(ast.NodeVisitor):
    """Records what the statements of one scope bind, without entering nested scopes."""

    def __init__(self, scope: _Scope, resolve) -> None:
        self.scope = scope
        self.resolve = resolve
        # Aliases such as ``s = librosa.stft`` wait until the imports are known.
        self.assignments: list[tuple[str, ast.expr]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _is_librosa(alias.name):
                self.scope.bind_other(alias.asname or alias.name.split(".")[0])
            elif alias.asname:
                self.scope.bind_librosa(alias.asname, alias.name)
            else:
                # ``import librosa.feature`` binds ``librosa``.
                self.scope.bind_librosa("librosa", "librosa")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level == 0 and _is_librosa(module):
            for alias in node.names:
                if alias.name == "*":
                    self.scope.star = True
                else:
                    self.scope.bind_librosa(alias.asname or alias.name, f"{module}.{alias.name}")
        else:
            for alias in node.names:
                if alias.name != "*":
                    self.scope.bind_other(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.bind_other(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.bind_other(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.bind_other(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def _visit_comprehension(self, node) -> None:
        pass

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments.append((target.id, node.value))
            else:
                for name in _target_names(target):
                    self.scope.bind_other(name)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in _target_names(node.target):
            if node.value is not None:
                self.assignments.append((name, node.value))
            else:
                self.scope.bind_other(name)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _target_names(node.target):
            self.scope.bind_other(name)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.scope.bind_other(node.target.id)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        for name in _target_names(node.target):
            self.scope.bind_other(name)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                for name in _target_names(item.optional_vars):
                    self.scope.bind_other(name)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.scope.bind_other(node.name)
        self.generic_visit(node)


class _Analyser(ast.NodeVisitor):
    """Walks a module (or a notebook's cells) and records each use of librosa."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.scopes: list[_Scope] = []
        self.findings: list[Finding] = []
        self.cell: int | None = None
        self.source_lines: list[str] = []

    # Scope handling ------------------------------------------------------

    def _enter(self, kind: str, body: Iterable[ast.AST], bound: Iterable[str] = ()) -> _Scope:
        scope = self.scopes[0] if kind == "module" and self.scopes else _Scope(kind)
        for name in bound:
            scope.bind_other(name)
        collector = _Collector(scope, self._resolve)
        for statement in body:
            collector.visit(statement)
        if scope not in self.scopes:
            self.scopes.append(scope)
        # Aliases resolve once the scope's imports are known, in source order.
        for name, value in collector.assignments:
            dotted = self._resolve(value)
            if dotted is not None:
                scope.bind_librosa(name, dotted)
            else:
                scope.bind_other(name)
        return scope

    def _lookup(self, name: str) -> str | None:
        innermost = len(self.scopes) - 1
        for depth in range(innermost, -1, -1):
            scope = self.scopes[depth]
            # A class body's names are not visible from the functions inside it.
            if scope.kind == "class" and depth != innermost:
                continue
            if name in scope.librosa:
                return scope.librosa[name]
            if name in scope.other:
                return None
        if self.scopes and self.scopes[0].star and _star_resolves(name):
            return f"librosa.{name}"
        return None

    def _resolve(self, node: ast.expr) -> str | None:
        """The librosa path a Name or dotted Attribute refers to, if it refers to one."""
        if isinstance(node, ast.Name):
            return self._lookup(node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            if base is not None:
                return f"{base}.{node.attr}"
        return None

    def visit_FunctionDef(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self._enter("function", node.body, _argument_names(node.args))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self._enter("function", [], _argument_names(node.args))
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expr in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
            self.visit(expr)
        self._enter("class", node.body)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def _visit_comprehension(self, node) -> None:
        bound = [name for gen in node.generators for name in _target_names(gen.target)]
        self._enter("function", [], bound)
        self.generic_visit(node)
        self.scopes.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    # Uses ----------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        # ``s = librosa.stft`` names an alias; the calls through it are what is reported.
        if all(isinstance(t, ast.Name) for t in node.targets) and self._resolve(node.value) is not None:
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        pass

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        pass

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._resolve(node.func)
        if dotted is None:
            self.generic_visit(node)
            return
        if not _is_namespace(dotted):
            self._report(_canonical(dotted), node, node)
        for argument in [*node.args, *node.keywords]:
            self.visit(argument)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = self._resolve(node)
        if dotted is None:
            self.generic_visit(node)
        else:
            self._report_reference(dotted, node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            dotted = self._lookup(node.id)
            if dotted is not None:
                self._report_reference(dotted, node)

    def _report_reference(self, dotted: str, node: ast.expr) -> None:
        if not _is_namespace(dotted):
            self._report(_canonical(dotted), node, None)

    def _report(self, canonical: str, node: ast.expr, call: ast.Call | None) -> None:
        text = _source_text(self.source_lines, node, canonical)
        entry = TABLE.get(canonical)
        location = {
            "path": self.path,
            "line": node.lineno,
            "column": node.col_offset,
            "cell": self.cell,
            "librosa": canonical,
            "call": text,
            "is_call": call is not None,
        }
        if canonical in REMOVED:
            replacement = REMOVED[canonical]
            target = TABLE.get(replacement) if replacement in TABLE else None
            if target is None:
                self.findings.append(
                    Finding(
                        equivalent=(),
                        match=NOT_PROVIDED,
                        note=f"Removed from librosa: {replacement}",
                        **location,
                    )
                )
                return
            entry = target
            location["librosa"] = replacement
        if entry is None:
            self.findings.append(
                Finding(
                    equivalent=(),
                    match=UNKNOWN,
                    note="Not a public function of librosa 1.0 in the migration table.",
                    **location,
                )
            )
            return
        self.findings.append(_finding_for(entry, call, location))


def _star_resolves(name: str) -> bool:
    return f"librosa.{name}" in TABLE


# Namespaces rather than functions. ``librosa.feature`` on its own, as an argument for
# instance, is not something to migrate call by call.
_NAMESPACES = frozenset(
    {
        "librosa",
        "librosa.core",
        "librosa.feature",
        "librosa.feature.inverse",
        "librosa.effects",
        "librosa.onset",
        "librosa.beat",
        "librosa.decompose",
        "librosa.filters",
        "librosa.util",
        "librosa.util.exceptions",
        "librosa.segment",
        "librosa.sequence",
        "librosa.display",
    }
)


def _is_namespace(dotted: str) -> bool:
    return dotted in _NAMESPACES or dotted in ALIASES


def _canonical(dotted: str) -> str:
    """Rewrite implementation paths such as ``librosa.core.spectrum.stft`` to the public name."""
    parts = dotted.split(".")
    # The longest module prefix that is an implementation path.
    for end in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:end])
        if module in ALIASES:
            dotted = ".".join([ALIASES[module], *parts[end:]])
            break
    # An attribute of a function, such as ``librosa.cache.clear``, belongs to the function.
    parts = dotted.split(".")
    for end in range(len(parts) - 1, 1, -1):
        prefix = ".".join(parts[:end])
        if prefix in TABLE or prefix in REMOVED:
            return prefix
    return dotted


def _source_text(lines: Sequence[str], node: ast.expr, fallback: str) -> str:
    """The source of a node on one line, shortened when long."""
    try:
        start, end = node.lineno - 1, node.end_lineno  # type: ignore[attr-defined]
        if start == end - 1:
            text = lines[start].encode()[node.col_offset : node.end_col_offset].decode()
        else:
            first = lines[start].encode()[node.col_offset :].decode()
            middle = lines[start + 1 : end - 1]
            last = lines[end - 1].encode()[: node.end_col_offset].decode()
            text = " ".join(part.strip() for part in [first, *middle, last])
    except (IndexError, TypeError, UnicodeDecodeError):
        return fallback
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= 100 else text[:97] + "..."


# --------------------------------------------------------------------------
# Advice for one call
# --------------------------------------------------------------------------


def _literal(node: ast.expr) -> tuple[bool, object]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _used_keywords(call: ast.Call, entry: Entry) -> tuple[dict[str, ast.expr], list[Watch]]:
    """Map each argument of the call to a librosa parameter name."""
    used: dict[str, ast.expr] = {}
    unknown: list[Watch] = []
    for name, arg in zip(entry.positional, call.args):
        if isinstance(arg, ast.Starred):
            unknown.append(Watch(NOT_CHECKED, "Positional arguments unpacked with * were not checked."))
            break
        used[name] = arg
    else:
        extra = call.args[len(entry.positional) :]
        if any(isinstance(arg, ast.Starred) for arg in extra):
            unknown.append(Watch(NOT_CHECKED, "Positional arguments unpacked with * were not checked."))
        elif extra and not entry.var_positional:
            unknown.append(
                Watch(
                    NOT_CHECKED,
                    f"{len(call.args)} positional arguments, but librosa 1.0 takes at most "
                    f"{len(entry.positional)} here: this call fails on librosa 1.0. Older releases took more.",
                )
            )
    for keyword in call.keywords:
        if keyword.arg is not None:
            used[keyword.arg] = keyword.value
        elif isinstance(keyword.value, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keyword.value.keys
        ):
            for k, v in zip(keyword.value.keys, keyword.value.values):
                used[k.value] = v  # type: ignore[union-attr]
        else:
            source = ast.unparse(keyword.value)
            unknown.append(
                Watch(
                    NOT_CHECKED,
                    f"Keywords unpacked from **{source} were not checked; defaults were not "
                    "checked either, as that mapping may set them.",
                )
            )
    return used, unknown


def _select_form(entry: Entry, used: dict[str, ast.expr]) -> Form | None:
    for form in entry.forms:
        if any(name in used and _not_none(used[name]) for name in form.when):
            return form
    return None


def _not_none(node: ast.expr) -> bool:  # ``S=None`` selects nothing
    return not (isinstance(node, ast.Constant) and node.value is None)


def _finding_for(entry: Entry, call: ast.Call | None, location: dict) -> Finding:
    if call is None:
        return Finding(
            equivalent=entry.ours,
            match=entry.match,
            note=entry.note,
            watch=(
                Watch(
                    NOT_CHECKED,
                    "Referenced but not called here, so its arguments were not checked.",
                ),
            ),
            **location,
        )

    used, watch = _used_keywords(call, entry)
    form = _select_form(entry, used)
    ours = form.ours if form else entry.ours
    match = form.match if form and form.match else entry.match
    note = form.note if form and form.note is not None else entry.note
    shape = form.shape if form else entry.shape
    options = form.options if form else entry.options

    seen_mapping = any(keyword.arg is None and not isinstance(keyword.value, ast.Dict) for keyword in call.keywords)
    seen_starred = any(isinstance(arg, ast.Starred) for arg in call.args)

    # Arguments that go away for the same reason are reported together.
    absorbed: dict[str, list[str]] = {}
    for name in used:
        option = options.get(name)
        if isinstance(option, Absorbed):
            absorbed.setdefault(option.reason, []).append(name)
    for reason, names in absorbed.items():
        verb = "goes" if len(names) == 1 else "go"
        watch.append(Watch(KEYWORD_CHANGED, f"{', '.join(names)} {verb} away: {reason}", names[0]))

    for name, node in used.items():
        option = options.get(name)
        if option is None:
            if not entry.takes_keywords and name not in entry.parameters:
                watch.append(
                    Watch(
                        NOT_CHECKED,
                        f"{name}= is not a parameter of {entry.name} in librosa 1.0.",
                        name,
                    )
                )
            continue
        watch.extend(_option_watch(name, node, option))

    if not (seen_mapping or seen_starred):
        for name, option in options.items():
            if isinstance(option, Default) and name not in used:
                watch.append(
                    Watch(
                        DEFAULT_DIFFERS,
                        f"{name} is not passed: librosa uses {option.librosa!r}, "
                        f"audio_samples uses {option.ours}. {option.advice}".rstrip(),
                        name,
                    )
                )

    if shape:
        watch.append(Watch(SHAPE_DIFFERS, shape))
    behaviours = form.behaviour if form and form.behaviour else entry.behaviour
    for text in behaviours:
        watch.append(Watch(BEHAVIOUR, text))

    return Finding(equivalent=ours, match=match, note=note, watch=tuple(watch), **location)


def _option_watch(name: str, node: ast.expr, option: object) -> Iterator[Watch]:
    if isinstance(option, Renamed):
        yield Watch(KEYWORD_CHANGED, f"{name}= becomes {option.to}=. {option.note}".rstrip(), name)
    elif isinstance(option, Unsupported):
        yield Watch(UNSUPPORTED, f"{name}= is not supported: {option.reason}", name)
    elif isinstance(option, Only):
        known, value = _literal(node)
        if not known:
            source = ast.unparse(node)
            if source in option.names:
                return
            yield Watch(
                NOT_CHECKED,
                f"{name}={source} is not a constant; only {option.describe()} is supported. "
                f"{option.reason}".rstrip(),
                name,
            )
        elif not option.allows(value):
            yield Watch(
                UNSUPPORTED,
                f"{name}={value!r} is not supported; only {option.describe()} is. {option.reason}".rstrip(),
                name,
            )
    elif isinstance(option, Default):
        if option.value_note:
            yield Watch(KEYWORD_CHANGED, f"{name}=: {option.value_note}", name)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


class SourceError(Exception):
    """A file could not be read or parsed."""


_MAGIC = re.compile(r"^(\s*)(%|!|\?)")
_ASSIGNED_SHELL = re.compile(r"^(\s*)[\w, ]+=\s*(!|%)")
_HELP_SUFFIX = re.compile(r"^(\s*)[\w.]+\?{1,2}\s*$")
# Cell magics whose body is Python.
_PYTHON_CELL_MAGICS = frozenset({"time", "timeit", "capture", "prun", "memit", "memray_flamegraph"})


def _strip_ipython(source: str) -> str:
    """Turn IPython syntax into blank statements, keeping the line numbers.

    A cell magic in a language other than Python empties the cell.
    """
    lines = source.split("\n")
    if lines and lines[0].lstrip().startswith("%%"):
        magic = lines[0].lstrip()[2:].split()
        name = magic[0] if magic else ""
        lines[0] = ""
        if name not in _PYTHON_CELL_MAGICS:
            return "\n" * (len(lines) - 1)
    out = []
    for line in lines:
        match = _MAGIC.match(line) or _ASSIGNED_SHELL.match(line) or _HELP_SUFFIX.match(line)
        out.append(f"{match.group(1)}pass" if match else line)
    return "\n".join(out)


def _analyse_module(analyser: _Analyser, tree: ast.Module, source: str) -> None:
    analyser.source_lines = source.split("\n")
    module = analyser._enter("module", tree.body)
    # ``global librosa`` followed by an import, the usual way to import lazily.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declared = {name for child in ast.walk(node) if isinstance(child, ast.Global) for name in child.names}
            if declared:
                inner = _Scope("function")
                collector = _Collector(inner, analyser._resolve)
                for statement in node.body:
                    collector.visit(statement)
                for name in declared & inner.librosa.keys():
                    module.bind_librosa(name, inner.librosa[name])
    for statement in tree.body:
        analyser.visit(statement)


def check_source(source: str, path: str = "<source>") -> list[Finding]:
    """Check Python source text."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:
        raise SourceError(f"{path}: cannot parse: {error}") from error
    analyser = _Analyser(path)
    _analyse_module(analyser, tree, source)
    return sorted(analyser.findings, key=lambda f: (f.line, f.column))


def _check_python_file(path: Path) -> list[Finding]:
    try:
        with tokenize.open(path) as stream:
            source = stream.read()
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise SourceError(f"{path}: cannot read: {error}") from error
    return check_source(source, str(path))


def _check_notebook(path: Path) -> list[Finding]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"{path}: cannot read: {error}") from error
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise SourceError(f"{path}: cannot read: not a version 4 notebook")

    # Cells share one namespace, so an import in one cell applies to later cells.
    analyser = _Analyser(str(path))
    number = 0
    unparsed: list[int] = []
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        number += 1
        raw = cell.get("source", "")
        source = _strip_ipython("".join(raw) if isinstance(raw, list) else str(raw))
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            unparsed.append(number)
            continue
        analyser.cell = number
        _analyse_module(analyser, tree, source)
    if unparsed:
        cells = ", ".join(map(str, unparsed))
        raise SourceError(
            f"{path}: cannot parse code cells {cells}",
            sorted(analyser.findings, key=lambda f: (f.cell or 0, f.line, f.column)),
        )
    return sorted(analyser.findings, key=lambda f: (f.cell or 0, f.line, f.column))


def _iter_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for root, directories, files in os.walk(path):
        directories[:] = sorted(d for d in directories if d not in _SKIPPED_DIRECTORIES)
        for name in sorted(files):
            if name.endswith((".py", ".ipynb")):
                yield Path(root) / name


def check_paths(paths: Iterable[str | os.PathLike]) -> tuple[list[Finding], list[str], int]:
    """Check files and directories.

    Returns the findings, the errors, and the number of files read.
    """
    findings: list[Finding] = []
    errors: list[str] = []
    files = 0
    for given in paths:
        path = Path(given)
        if not path.exists():
            errors.append(f"{path}: no such file or directory")
            continue
        if path.is_file() and path.suffix not in (".py", ".ipynb"):
            errors.append(f"{path}: not a .py file or a .ipynb notebook")
            continue
        for file in _iter_files(path):
            files += 1
            check = _check_notebook if file.suffix == ".ipynb" else _check_python_file
            try:
                findings.extend(check(file))
            except SourceError as error:
                errors.append(str(error.args[0]))
                if len(error.args) > 1:
                    findings.extend(error.args[1])
    return findings, errors, files


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _summary(findings: Sequence[Finding], errors: Sequence[str], files: int) -> dict:
    by_match = {EQUIVALENT: 0, WITH_DIFFERENCES: 0, NOT_PROVIDED: 0, UNKNOWN: 0}
    by_watch: dict[str, int] = {}
    for finding in findings:
        by_match[finding.match] += 1
        for watch in finding.watch:
            by_watch[watch.kind] = by_watch.get(watch.kind, 0) + 1
    return {
        "files": files,
        "files_with_librosa": len({f.path for f in findings}),
        "uses": len(findings),
        "functions": len({f.librosa for f in findings}),
        "by_match": by_match,
        "by_watch": dict(sorted(by_watch.items())),
        "blocking": sum(f.blocking for f in findings),
        "errors": len(errors),
    }


def _render_equivalent(names: Sequence[str]) -> str:
    return ", ".join(names) if names else "none"


def _print_text(findings: Sequence[Finding], errors: Sequence[str], summary: dict, stream) -> None:
    for finding in findings:
        print(f"{finding.location}: {finding.call}", file=stream)
        print(f"    use:   {_render_equivalent(finding.equivalent)}", file=stream)
        print(f"    match: {finding.match}", file=stream)
        if finding.note:
            print(f"    note:  {finding.note}", file=stream)
        for watch in finding.watch:
            print(f"    watch ({watch.kind}): {watch.message}", file=stream)
        print(file=stream)
    for error in errors:
        print(f"error: {error}", file=stream)
    if errors:
        print(file=stream)
    print(
        f"{summary['uses']} librosa uses of {summary['functions']} functions in "
        f"{summary['files_with_librosa']} of {summary['files']} files",
        file=stream,
    )
    matches = summary["by_match"]
    print(
        f"  match: {matches[EQUIVALENT]} equivalent, {matches[WITH_DIFFERENCES]} with differences, "
        f"{matches[NOT_PROVIDED]} not provided, {matches[UNKNOWN]} unknown",
        file=stream,
    )
    if summary["by_watch"]:
        counts = ", ".join(f"{count} {kind}" for kind, count in summary["by_watch"].items())
        print(f"  watch: {counts}", file=stream)
    print(f"  blocking: {summary['blocking']}, errors: {summary['errors']}", file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m audio_samples.migrate",
        description="Find librosa usage in Python files and notebooks and say what replaces it. "
        "Reads the files; never changes them.",
    )
    parser.add_argument("paths", nargs="+", metavar="PATH", help="a .py file, an .ipynb notebook or a directory")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="output format (default: text)")
    parser.add_argument(
        "--fail-on",
        choices=("any", "blocking", "never"),
        default="any",
        help="what makes the exit status 1: any librosa usage (default), only usage with no equivalent or with an "
        "unsupported option, or nothing. Errors always give 2.",
    )
    args = parser.parse_args(argv)

    findings, errors, files = check_paths(args.paths)
    summary = _summary(findings, errors, files)

    if args.format == "json":
        json.dump(
            {
                "findings": [f.as_dict() for f in findings],
                "errors": list(errors),
                "summary": summary,
            },
            sys.stdout,
            indent=2,
        )
        print()
    else:
        _print_text(findings, errors, summary, sys.stdout)

    if errors:
        return 2
    if args.fail_on == "any" and findings:
        return 1
    if args.fail_on == "blocking" and summary["blocking"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
