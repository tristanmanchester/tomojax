"""Files to scans and back: :func:`load`, :class:`Frames`, :func:`inspect` and :func:`export`.

``loader`` builds scans and lazy frames from files; ``geometry`` resolves a
record's geometry before any pixels are read; ``frame_sources`` opens saved
scans, recognized acquisitions and TIFF stacks as frames; ``files`` inspects
files and exports volumes for other software, naming TIFF stacks through
``stack_files``.
"""
