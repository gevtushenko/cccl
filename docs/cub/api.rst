CUB API Reference
=================

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/device
   api/block
   api/warp
   api/thread
   api/grid
   api/iterator
   api/utility

Overview
--------

CUB provides state-of-the-art, reusable software components for every layer 
of the CUDA programming model:

* **Device-wide primitives**: Parallel algorithms across all threads
* **Block-wide primitives**: Parallel algorithms across thread blocks
* **Warp-wide primitives**: Parallel algorithms across warps
* **Thread primitives**: Sequential algorithms within threads

Device-wide API
---------------

.. doxygennamespace:: cub
   :project: cub
   :members:
   :undoc-members:
   :content-only: