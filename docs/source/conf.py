# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
project = "AudioSamples Python"
copyright = "2025, Jack Geraghty"
author = "Jack Geraghty"

def _crate_version() -> str:
    """Package version, read from Cargo.toml (the single source of truth)."""
    import tomllib

    with open(os.path.join(os.path.dirname(__file__), "../../Cargo.toml"), "rb") as f:
        return tomllib.load(f)["package"]["version"]


release = _crate_version()

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_gallery.gen_gallery",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# sphinx-gallery writes a notebook next to each generated page, for download; those
# must not be read as source documents by myst-nb.
exclude_patterns = ["**.ipynb_checkpoints", "gallery/**/*.ipynb"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Theme options
html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#009E73",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_preprocess_types = False
# napoleon_type_aliases = None
# napoleon_attr_annotations = True


# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


# -- Notebooks (tutorials) ----------------------------------------------------
# The tutorial notebooks are committed without outputs. When the compiled
# extension is importable (e.g. after `maturin develop`), the docs build runs
# them so the rendered pages show real results; otherwise (e.g. a docs-only CI
# job without the extension) they are rendered unexecuted. Their correctness
# is checked separately by `pytest --nbmake docs/source/tutorials`.
try:  # pragma: no cover - depends on the build environment
    import audio_samples as _aus  # noqa: F401

    nb_execution_mode = os.environ.get("AUDIO_SAMPLES_NB_EXECUTION", "auto")
except Exception:  # noqa: BLE001
    nb_execution_mode = "off"
nb_execution_timeout = 180
nb_execution_raise_on_error = True

# -- Example gallery ----------------------------------------------------------
# The sources are in docs/gallery. Every example runs during the build and a
# failing one stops the build. Without the compiled extension (a docs-only job)
# the pages are written without running them, as the notebooks above are.
sphinx_gallery_conf = {
    "examples_dirs": "../gallery",
    "gallery_dirs": "gallery",
    "subsection_order": [
        "../gallery/loading_saving",
        "../gallery/spectrograms",
        "../gallery/mel_mfcc",
        "../gallery/pitch",
        "../gallery/beats_onsets",
        "../gallery/separation",
        "../gallery/effects",
        "../gallery/resampling",
        "../gallery/filters",
        "../gallery/plotting",
    ],
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": r"/plot_",
    "plot_gallery": "True" if nb_execution_mode != "off" else "False",
    "abort_on_example_error": True,
    "only_warn_on_example_error": False,
    # An ``AudioSamples`` as the last expression of a cell renders its audio player.
    "capture_repr": ("_repr_html_", "__repr__"),
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib",),
    "backreferences_dir": None,
    "download_all_examples": False,
    "write_computation_times": False,
    "show_memory": False,
    "show_signature": False,
    "remove_config_comments": True,
}


def _gallery_index_sections_as_rubrics(app, docname, source):
    """Keep the gallery index's section titles out of the sidebar.

    sphinx-gallery writes each section as a heading on the index page and its toctree after
    the last one, so the sidebar would hang every section's page under the last section.
    As rubrics the titles look the same on the page, and the toctree belongs to the page title.
    """
    if docname == "gallery/index":
        import re

        source[0] = re.sub(r"^(.+)\n-{3,}$", r".. rubric:: \1", source[0], flags=re.M)


def setup(app):
    app.connect("source-read", _gallery_index_sections_as_rubrics)
