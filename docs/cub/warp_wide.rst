.. _warp-module:

Warp-Wide "Collective" Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   api/warp

CUB warp-level algorithms are specialized for execution by threads in the same CUDA warp.
These algorithms may only be invoked by ``1 <= n <= 32`` *consecutive* threads in the same warp:

* :cpp:struct:`WarpExchange` rearranges data partitioned across a CUDA warp
* :cpp:class:`WarpLoad` loads a linear segment of items from memory into a CUDA warp
* :cpp:class:`WarpMergeSort` sorts items partitioned across a CUDA warp
* :cpp:struct:`WarpReduce` computes reduction of items partitioned across a CUDA warp
* :cpp:struct:`WarpScan` computes a prefix scan of items partitioned across a CUDA warp
* :cpp:class:`WarpStore` stores items partitioned across a CUDA warp to a linear segment of memory
