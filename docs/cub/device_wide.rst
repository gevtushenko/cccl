.. _device-module:

Device-Wide Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   api/device


CUB device-level single-problem parallel algorithms:

* :cpp:struct:`DeviceAdjacentDifference` computes the difference between adjacent elements residing within device-accessible memory
* :cpp:struct:`DeviceFor` provides device-wide, parallel operations for iterating over data residing within device-accessible memory
* :cpp:struct:`DeviceHistogram` constructs histograms from data samples residing within device-accessible memory
* :cpp:struct:`DevicePartition` partitions data residing within device-accessible memory
* :cpp:struct:`DeviceMerge` merges two sorted sequences in device-accessible memory into a single one
* :cpp:struct:`DeviceMergeSort` sorts items residing within device-accessible memory
* :cpp:struct:`DeviceRadixSort` sorts items residing within device-accessible memory using radix sorting method
* :cpp:struct:`DeviceReduce` computes reduction of items residing within device-accessible memory
* :cpp:struct:`DeviceRunLengthEncode` demarcating "runs" of same-valued items withing a sequence residing within device-accessible memory
* :cpp:struct:`DeviceScan` computes a prefix scan across a sequence of data items residing within device-accessible memory
* :cpp:struct:`DeviceSelect` compacts data residing within device-accessible memory


CUB device-level segmented-problem (batched) parallel algorithms:

* :cpp:struct:`DeviceSegmentedSort` computes batched sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`DeviceSegmentedRadixSort` computes batched radix sort across non-overlapping sequences of data residing within device-accessible memory
* :cpp:struct:`DeviceSegmentedReduce` computes reductions across multiple sequences of data residing within device-accessible memory
* :cpp:struct:`DeviceCopy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
* :cpp:struct:`DeviceMemcpy` provides device-wide, parallel operations for batched copying of data residing within device-accessible memory
