"""Sphinx configuration for the generated Python reference."""

project = "TomoJAX"
extensions = ["sphinx.ext.autodoc", "numpydoc"]
autodoc_typehints = "none"  # Docstrings explain shapes and units as well as types.
autodoc_member_order = "bysource"
numpydoc_show_class_members = False  # Autodoc's members directive includes them once.
numpydoc_class_members_toctree = False
html_theme = "alabaster"
exclude_patterns = []
