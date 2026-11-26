"""The workflow operations behind ``tj.reconstruct``, ``tj.align``, ``tj.project`` and ``tj.save``.

``operations`` holds the operations themselves. Below it, from the top:
``resources`` estimates working memory; ``smooth_alignment`` runs the joint
alignment, publishing calibration through ``calibration``; ``planning`` resolves
settings before solver state is allocated; ``field_of_view`` and ``region`` choose
the reconstructed grid. ``evaluation`` sits above them all and scores fits on
held-out partitions.
"""
