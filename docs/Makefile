# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?= --keep-going
SPHINXBUILD   ?= python3 -m sphinx.cmd.build
SOURCEDIR     = .
BUILDDIR      = _build
DOXYGEN       ?= doxygen

# Doxygen configuration
CUB_DOXYGEN_INPUT = ../cub/cub/*.cuh ../cub/cub/thread/*.cuh ../cub/cub/warp/*.cuh ../cub/cub/block/*.cuh ../cub/cub/device/*.cuh ../cub/cub/grid/*.cuh ../cub/cub/iterator/*.cuh
THRUST_DOXYGEN_INPUT = ../thrust/thrust/*.h ../thrust/thrust/iterator/*.h ../thrust/thrust/mr/*.h ../thrust/thrust/random/*.h ../thrust/thrust/system/*.h ../thrust/thrust/type_traits/*.h
CUDAX_DOXYGEN_INPUT = ../cudax/include/cuda/experimental/*.cuh ../cudax/include/cuda/experimental/__*/*.cuh

.PHONY: help clean clean-doxygen doxygen-cub doxygen-thrust doxygen-cudax doxygen html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  clean      to remove all build files"
	@echo "  doxygen    to generate Doxygen XML for all projects"

clean: clean-doxygen
	rm -rf $(BUILDDIR)/*

clean-doxygen:
	rm -rf $(BUILDDIR)/doxygen

# Generate Doxyfile for a project
define generate_doxyfile
	@mkdir -p $(BUILDDIR)/doxygen/$(1)/xml
	@echo "PROJECT_NAME = $(2)" > $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "OUTPUT_DIRECTORY = $(BUILDDIR)/doxygen/$(1)" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "INPUT = $(3)" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "GENERATE_HTML = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "GENERATE_LATEX = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "GENERATE_XML = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "XML_OUTPUT = xml" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "RECURSIVE = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXTRACT_ALL = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXTRACT_PRIVATE = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXTRACT_STATIC = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXTRACT_LOCAL_CLASSES = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "HIDE_UNDOC_MEMBERS = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "HIDE_UNDOC_CLASSES = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "SHOW_INCLUDE_FILES = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "INLINE_INHERITED_MEMB = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "FULL_PATH_NAMES = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "STRIP_FROM_PATH = $(4)" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "SHORT_NAMES = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "JAVADOC_AUTOBRIEF = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "QT_AUTOBRIEF = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "MULTILINE_CPP_IS_BRIEF = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "INHERIT_DOCS = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "SEPARATE_MEMBER_PAGES = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "TAB_SIZE = 4" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "BUILTIN_STL_SUPPORT = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "ENABLE_PREPROCESSING = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "MACRO_EXPANSION = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXPAND_ONLY_PREDEF = NO" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "SEARCH_INCLUDES = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "SKIP_FUNCTION_MACROS = YES" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXCLUDE_SYMBOLS = *detail* *__*" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "EXTENSION_MAPPING = cuh=C++ cu=C++" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "FILE_PATTERNS = *.h *.cuh *.hpp *.cpp *.cu" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
	@echo "PREDEFINED = __device__= __host__= __global__= __forceinline__= _CCCL_DOXYGEN_INVOKED" >> $(BUILDDIR)/doxygen/$(1)/Doxyfile
endef

check-doxygen:
	@which $(DOXYGEN) > /dev/null 2>&1 || { echo "Warning: Doxygen not found. API documentation will be limited."; echo "To install doxygen: apt-get install doxygen (Ubuntu/Debian) or brew install doxygen (macOS)"; false; }

doxygen-cub: check-doxygen
	@mkdir -p _build/doxygen/cub
	@echo "Generating Doxygen XML for CUB..."
	@cd cub && doxygen Doxyfile

doxygen-thrust: check-doxygen
	@mkdir -p _build/doxygen/thrust
	@echo "Generating Doxygen XML for Thrust..."
	@cd thrust && doxygen Doxyfile

doxygen-cudax: check-doxygen
	@echo "Generating Doxygen XML for CUDAX (not yet configured)"

doxygen-combined: check-doxygen
	@mkdir -p _build/doxygen
	@echo "Generating combined Doxygen XML for exhale..."
	@cp Doxyfile.in _build/doxygen/Doxyfile
	@cd _build/doxygen && $(DOXYGEN) Doxyfile

doxygen: 
	@which $(DOXYGEN) > /dev/null 2>&1 && ($(MAKE) doxygen-cub doxygen-thrust doxygen-cudax) && echo "Doxygen XML generation complete" || echo "Skipping Doxygen generation (not installed)"

html: 
	@$(MAKE) doxygen || true
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(SOURCEDIR) $(BUILDDIR)/dirhtml $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(SOURCEDIR) $(BUILDDIR)/singlehtml $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(SOURCEDIR) $(BUILDDIR)/pickle $(SPHINXOPTS)
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(SOURCEDIR) $(BUILDDIR)/json $(SPHINXOPTS)
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(SOURCEDIR) $(BUILDDIR)/htmlhelp $(SPHINXOPTS)
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(SOURCEDIR) $(BUILDDIR)/qthelp $(SPHINXOPTS)
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/CUDACoreCoreLibraries.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/CUDACoreCoreLibraries.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(SOURCEDIR) $(BUILDDIR)/devhelp $(SPHINXOPTS)
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/CUDACoreCoreLibraries"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/CUDACoreCoreLibraries"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub $(SPHINXOPTS)
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run 'make' in that directory to run these through (pdf)latex" \
	      "(use 'make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	$(SPHINXBUILD) -b text $(SOURCEDIR) $(BUILDDIR)/text $(SPHINXOPTS)
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(SOURCEDIR) $(BUILDDIR)/man $(SPHINXOPTS)
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

changes:
	$(SPHINXBUILD) -b changes $(SOURCEDIR) $(BUILDDIR)/changes $(SPHINXOPTS)
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS)
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

gettext:
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/locale $(SPHINXOPTS)
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."