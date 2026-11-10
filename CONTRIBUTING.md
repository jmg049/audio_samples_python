# Contributing to audio_samples

## Contributions

Contributions in the form of bug reports, feature requests, or pull requests are
welcome. For pull requests, please consider:

* Write clean and descriptive commit messages and keep the history clean.
* Ensure cargo fmt is run.
* Documentation and tests are required.

## Development setup

With this repository, `audio_samples` and `audio_samples_io` checked out side by side:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release --group test
```

This builds the extension into the environment and installs what the tests and hooks need. `pytest` then runs the
tests and `pre-commit run --all-files` runs the hooks. Without `--group`, `maturin develop` installs the `dev` group
instead, which holds the extras for notebooks and benchmarks. The `heavy` group (PyTorch and Qt) is needed only by
`audio_samples/io_benchmark.py` and interactive plotting: add it with `--group test,heavy`.

## Type stubs

The stubs in `audio_samples/*.pyi` are written by hand: they hold what Rust cannot express, such as
the generic `AudioSamples`, the overloads for dtype rules and argument-dependent results, and
`Literal` values. `stubs/generated/` holds the stubs pyo3 generates from the bindings, and the hand
stubs are checked against it. After adding or changing a binding, or editing a stub, run

```bash
scripts/stubs.sh
```

It builds the extension with the `stubs` feature (pyo3's introspection data; release wheels are
built without it), rewrites `stubs/generated/` and `tests/typing/reveals/`, and runs the checks.
Fix what it reports, then commit the regenerated files with the change. The pre-commit hook and the
`Type stubs` workflow run `scripts/stubs.sh --check`, which fails when `stubs/generated/` is out of
date or when:

* a binding (function, class, method, property) is missing from the hand stubs, or the stubs have
  an item the bindings do not;
* a parameter differs in name, kind or default;
* a stub type disagrees with the Rust type: each overload's parameter types must be accepted by the
  Rust type and its return type must be one the Rust type allows
  (`tests/typing/check_against_rust.py`, proven by mypy and pyright);
* a binding gives no Python type where the stub states one. Give it one in Rust:
  `#[pyo3(signature = (x: "NDArray[np.float64]") -> "float")]`, or `INPUT_TYPE` on an argument
  extractor (see `args::hints`). `tests/typing/untyped_bindings.txt` lists the bindings not yet
  typed in Rust; it may only shrink;
* any overload, method or property gives a different type under mypy or pyright than recorded in
  `tests/typing/reveals.snapshot`. For an intended change run
  `python tests/typing/check_precision.py --update` and commit the snapshot;
* `mypy.stubtest`, mypy, pyright or `tests/test_stub_types.py` fail.

The revealed types depend on the checker versions: install them with
`pip install --group test --group typing`.

## Building

The bindings build against checkouts of
[`audio_samples`](https://github.com/jmg049/audio_samples) and
[`audio_samples_io`](https://github.com/jmg049/audio_samples_io) next to this
one (`[patch.crates-io]` in `Cargo.toml`), until the matching releases are on
crates.io. `.github/workflows/wheels.yml` pins the commits it uses.

Wheels: Actions > Wheels > Run workflow builds and tests everything without
publishing. Only a pushed `v*` tag publishes, through trusted publishing.

An sdist needs those sources inside it, because maturin does not package the
targets of a patch table: run `tests/ci/bundle_siblings.py` on a throw-away
checkout first, then `maturin sdist`. The result builds on its own with a Rust
toolchain and access to crates.io (or a `cargo vendor` directory); `Cargo.lock`
is included and it has no C dependencies.

## Usage of AI Tools

> Copied from the [Rust audio_samples](https://github.com/jmg049/audio_samples/blob/main/CONTRIBUTING.md) contribution guidelines.

Since late 2022 there has been a rapid expansion of AI-based tooling for writing and interacting with code. Large Language Models such as ChatGPT, Gemini, and Claude allow users to query codebases, debug issues, discuss architectural ideas, and even generate tests or refactor modules. Some providers now offer agentic variants that execute actions, gather results, and iteratively update their own plan in response to intermediate outputs.

The effectiveness of these tools is debated. Some developers enjoy working with them, others avoid them entirely, and most fall somewhere between. Like many people, I have experimented with a range of these systems. The Git history reflects this: at the time of writing, I have 35 commits with large-scale additions and removals, while Claude contributes 7 commits totalling around one thousand lines of change. This does not capture cases where I used Claude Code interactively without letting it commit and push autonomously, but it does show my experimentation with these tools.

---

The existence of these tools does not change the responsibilities involved in software development. **They do not excuse poor judgement, weak design, or low standards.** A contributor must still produce code that matches the project’s conventions and constraints, regardless of whether an LLM generated an initial draft. Reviewers must still apply the same scrutiny before merging. Expectations remain the same irrespective of whether AI was used during development.

More broadly, the availability of these tools removes any remaining excuse for poor project supports. If anything, the rise of LLM-assisted development makes clear documentation, tests, and examples more important, not less. They anchor the project in well-defined behaviour, provide an authoritative reference when an AI tool produces something misguided, and ensure that humans and machines alike operate against the same ground truth.

**Use them as you see fit, but ultimately you are the one responsible.**

---

In practice, I treat these systems as tools that can accelerate specific, well-defined tasks. They do not replace architectural reasoning, design decisions, or accountability. For this project, Claude Code in particular struggled with core structural elements such as ownership, trait bounds, and the architectural model that underpins ``audio_samples``. As a result, I design and implement the architecture myself, using LLMs only for contained boilerplate or refactoring where the scope is clear and the risks are manageable.

> A final note on terminology: the word AI is used here only because it has become the default label in public and technical conversations. It is not a precise description. Using it uncritically encourages anthropomorphic language and obscures the fact that these tools are fundamentally computational systems. Clear terminology matters, especially when discussing responsibility, intent, and reliability.

## Code of conduct

* Be sound.
