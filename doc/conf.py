"""Sphinx configuration for the pyts documentation.

Built with a recent Sphinx release and the PyData Sphinx Theme, the same
stack used by numpy, scipy, scikit-learn, pandas and matplotlib.
"""

import os
import sys
import warnings

from sphinx_gallery.sorting import ExampleTitleSortKey

from pyts import __version__

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# -- General configuration ------------------------------------------------

needs_sphinx = "8.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "numpydoc",
    "pytsdtwdoc",
    "sphinx_gallery.gen_gallery",
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# Generate the plots for the gallery
plot_gallery = "True"

# The root document.
root_doc = "index"

# General information about the project.
project = "pyts"
copyright = "2017-2026, Johann Faouzi and all pyts contributors"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates"]

# -- Options for HTML output ----------------------------------------------

# PyData Sphinx Theme: the same theme used by numpy, scipy, scikit-learn,
# pandas and matplotlib.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "image_light": "_static/img/logo.png",
        "image_dark": "_static/img/logo.png",
    },
    "github_url": "https://github.com/johannfaouzi/pyts",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyts/",
            "icon": "fa-solid fa-box",
        },
    ],
    # Same navbar layout as numpy/scipy/scikit-learn: the nav links sit
    # right next to the logo instead of being centered over the (much
    # wider) content column, which is what was creating the large gap.
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "header_links_before_dropdown": 6,
    "header_dropdown_text": "More",
    "pygments_light_style": "sphinx",
    "pygments_dark_style": "monokai",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# Hide the primary (left) sidebar on the landing page; every other page
# keeps the auto-generated navigation from the toctree.
html_sidebars = {
    "index": [],
}

# Add any paths that contain custom static files (such as style sheets)
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
html_static_path = ["_static"]

# Custom CSS files, loaded after the theme's own stylesheet.
html_css_files = [
    "custom.css",
]

html_favicon = "_static/img/logo.png"

html_title = f"pyts {version}"

html_context = {
    "github_user": "johannfaouzi",
    "github_repo": "pyts",
    "github_version": "main",
    "doc_path": "doc",
}

# Output file base name for HTML help builder.
htmlhelp_basename = "pytsdoc"


# -- Options for LaTeX output ---------------------------------------------
latex_engine = "pdflatex"

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (root_doc, "pyts.tex", "pyts Documentation", "Johann Faouzi", "manual"),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "pyts", "pyts Documentation", ["Johann Faouzi"], 1)]


# -- Options for Texinfo output --------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "pyts",
        "pyts Documentation",
        "Johann Faouzi",
        "pyts",
        "A python package for time series transformation and classification",
        "Miscellaneous",
    ),
]


# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- sphinx-gallery configuration --------------------------------------------
sphinx_gallery_conf = {
    "doc_module": "pyts",
    "backreferences_dir": os.path.join("generated"),
    "within_subsection_order": ExampleTitleSortKey,
    "reference_url": {"pyts": None},
}


# Filter Matplotlib warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)
