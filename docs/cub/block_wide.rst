.. _block-module:

Block-Wide "Collective" Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   api/block

CUB block-level algorithms are specialized for execution by threads in the same CUDA thread block:

* :cpp:class:`BlockAdjacentDifference` computes the difference between adjacent items partitioned across a CUDA thread block
* :cpp:class:`BlockDiscontinuity` flags discontinuities within an ordered set of items partitioned across a CUDA thread block
* :cpp:struct:`BlockExchange` rearranges data partitioned across a CUDA thread block
* :cpp:class:`BlockHistogram` constructs block-wide histograms from data samples partitioned across a CUDA thread block
* :cpp:class:`BlockLoad` loads a linear segment of items from memory into a CUDA thread block
* :cpp:class:`BlockMergeSort` sorts items partitioned across a CUDA thread block
* :cpp:class:`BlockRadixSort` sorts items partitioned across a CUDA thread block using radix sorting method
* :cpp:struct:`BlockReduce` computes reduction of items partitioned across a CUDA thread block
* :cpp:class:`BlockRunLengthDecode` decodes a run-length encoded sequence partitioned across a CUDA thread block
* :cpp:struct:`BlockScan` computes a prefix scan of items partitioned across a CUDA thread block
* :cpp:struct:`BlockShuffle` shifts items partitioned across a CUDA thread block
* :cpp:class:`BlockStore` stores items partitioned across a CUDA thread block to a linear segment of memory
