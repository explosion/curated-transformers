# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Curated Transformers"
copyright = "2021-2023, ExplosionAI GmbH"
author = "ExplosionAI GmbH"

release = "0.9"
version = "0.9.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": False,
    "titles_only": False,
}

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Display both the constructor and class docs.
autoclass_content = "both"
