"""
Sphinx extension to automatically generate API reference pages from Doxygen XML.
Replicates repo-docs' automatic API generation functionality.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def extract_doxygen_classes(xml_dir):
    """Extract all classes/structs from Doxygen XML."""
    classes = {
        'device': [],
        'block': [],
        'warp': [],
        'grid': [],
        'iterator': [],
        'thread': [],
        'utility': []
    }
    
    xml_path = Path(xml_dir)
    if not xml_path.exists():
        return classes
    
    # Parse index.xml to get all compounds
    index_file = xml_path / 'index.xml'
    if not index_file.exists():
        return classes
    
    try:
        tree = ET.parse(index_file)
        root = tree.getroot()
        
        for compound in root.findall('.//compound[@kind="class"]') + root.findall('.//compound[@kind="struct"]'):
            name = compound.find('name').text
            refid = compound.get('refid')
            
            # Skip internal/detail classes
            if 'detail' in name.lower() or '__' in name:
                continue
            
            # Remove namespace prefixes for categorization
            simple_name = name.split('::')[-1] if '::' in name else name
            
            # Categorize based on name (using simple name for matching)
            if 'Device' in simple_name:
                classes['device'].append((name, refid))
            elif 'Block' in simple_name:
                classes['block'].append((name, refid))
            elif 'Warp' in simple_name:
                classes['warp'].append((name, refid))
            elif 'Grid' in simple_name:
                classes['grid'].append((name, refid))
            elif 'Iterator' in simple_name.lower() or 'iterator' in simple_name.lower():
                classes['iterator'].append((name, refid))
            elif any(x in simple_name for x in ['Traits', 'Type', 'Allocator', 'Debug', 'Caching']):
                classes['utility'].append((name, refid))
    except Exception as e:
        logger.warning(f"Failed to parse Doxygen XML: {e}")
    
    return classes


def generate_api_page(category, classes, project_name):
    """Generate RST content for an API category page."""
    
    category_titles = {
        'device': 'Device-wide Primitives',
        'block': 'Block-wide Primitives',
        'warp': 'Warp-wide Primitives',
        'grid': 'Grid-level Primitives',
        'iterator': 'Iterator Utilities',
        'thread': 'Thread-level Primitives',
        'utility': 'Utility Components'
    }
    
    content = []
    content.append(category_titles.get(category, f'{category.title()} API'))
    content.append('=' * len(content[0]))
    content.append('')
    content.append('.. contents:: Table of Contents')
    content.append('   :local:')
    content.append('   :depth: 2')
    content.append('')
    
    # Sort classes by name
    classes.sort(key=lambda x: x[0])
    
    for class_name, refid in classes:
        # Use the full name including namespace for display
        display_name = class_name
        
        content.append(display_name)
        content.append('-' * len(display_name))
        content.append('')
        
        # Check if this is a struct by looking at the refid
        # Doxygen uses 'struct' prefix in the refid for structs
        directive = 'doxygenstruct' if refid.startswith('struct') else 'doxygenclass'
        
        # Use the full qualified name including namespace
        content.append(f'.. {directive}:: {class_name}')
            
        content.append(f'   :project: {project_name}')
        content.append('   :members:')
        content.append('   :undoc-members:')
        content.append('')
    
    return '\n'.join(content)


def generate_api_docs(app, config):
    """Generate API documentation pages during Sphinx build."""
    
    # Only generate for projects with breathe configuration
    if not hasattr(config, 'breathe_projects'):
        return
    
    for project_name, xml_dir in config.breathe_projects.items():
        # Skip if XML directory doesn't exist
        if not os.path.exists(xml_dir):
            continue
        
        # Extract classes from Doxygen XML
        classes = extract_doxygen_classes(xml_dir)
        
        # Determine output directory based on project
        if project_name == 'cub':
            api_dir = Path(app.srcdir) / 'cub' / 'api'
        elif project_name == 'thrust':
            api_dir = Path(app.srcdir) / 'thrust' / 'api'
        elif project_name == 'libcudacxx':
            api_dir = Path(app.srcdir) / 'libcudacxx' / 'api'
        elif project_name == 'cudax':
            api_dir = Path(app.srcdir) / 'cudax' / 'api'
        else:
            continue
        
        # Create API directory if it doesn't exist
        api_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate pages for each category
        for category, class_list in classes.items():
            if class_list:
                content = generate_api_page(category, class_list, project_name)
                output_file = api_dir / f'{category}.rst'
                
                # Write the generated content
                with open(output_file, 'w') as f:
                    f.write(content)
                logger.info(f"Generated API page: {output_file}")


def setup(app: Sphinx):
    """Setup the extension."""
    app.connect('config-inited', generate_api_docs)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }