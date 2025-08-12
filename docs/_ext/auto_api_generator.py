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


def extract_doxygen_items(xml_dir):
    """Extract all items (classes, structs, functions, etc.) from Doxygen XML."""
    items = {
        'classes': [],
        'structs': [],
        'functions': [],
        'typedefs': [],
        'enums': [],
        'variables': []
    }
    
    xml_path = Path(xml_dir)
    if not xml_path.exists():
        return items
    
    # Parse index.xml to get all compounds and members
    index_file = xml_path / 'index.xml'
    if not index_file.exists():
        return items
    
    try:
        tree = ET.parse(index_file)
        root = tree.getroot()
        
        # Get the namespace compound (e.g., cub, thrust)
        namespace_compound = None
        for compound in root.findall('.//compound[@kind="namespace"]'):
            name = compound.find('name').text
            # Match primary namespaces
            if name in ['cub', 'thrust', 'cuda']:
                namespace_compound = compound
                break
        
        # Extract classes and structs
        for compound in root.findall('.//compound[@kind="class"]'):
            name = compound.find('name').text
            refid = compound.get('refid')
            
            # Skip internal/detail classes
            if 'detail' in name.lower() or '__' in name:
                continue
            
            items['classes'].append((name, refid))
        
        for compound in root.findall('.//compound[@kind="struct"]'):
            name = compound.find('name').text
            refid = compound.get('refid')
            
            # Skip internal/detail structs
            if 'detail' in name.lower() or '__' in name:
                continue
            
            items['structs'].append((name, refid))
        
        # Extract functions, typedefs, enums, and variables from namespace
        if namespace_compound:
            for member in namespace_compound.findall('member[@kind="function"]'):
                name = member.find('name').text
                refid = member.get('refid')
                items['functions'].append((name, refid))
            
            for member in namespace_compound.findall('member[@kind="typedef"]'):
                name = member.find('name').text
                refid = member.get('refid')
                items['typedefs'].append((name, refid))
            
            for member in namespace_compound.findall('member[@kind="enum"]'):
                name = member.find('name').text
                refid = member.get('refid')
                items['enums'].append((name, refid))
            
            for member in namespace_compound.findall('member[@kind="variable"]'):
                name = member.find('name').text
                refid = member.get('refid')
                items['variables'].append((name, refid))
    
    except Exception as e:
        logger.warning(f"Failed to parse Doxygen XML: {e}")
    
    return items

def extract_doxygen_classes(xml_dir):
    """Extract classes categorized by type for category pages."""
    classes = {
        'device': [],
        'block': [],
        'warp': [],
        'grid': [],
        'iterator': [],
        'thread': [],
        'utility': []
    }
    
    items = extract_doxygen_items(xml_dir)
    
    # Categorize classes and structs
    for name, refid in items['classes'] + items['structs']:
        # Remove namespace prefixes for categorization
        simple_name = name.split('::')[-1] if '::' in name else name
        
        # Categorize based on name
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


def generate_individual_api_page(class_name, refid, project_name):
    """Generate RST content for a single class/struct API page."""
    content = []
    
    # Add title
    content.append(class_name)
    content.append('=' * len(class_name))
    content.append('')
    
    # Check if this is a struct by looking at the refid
    directive = 'doxygenstruct' if refid.startswith('struct') else 'doxygenclass'
    
    # Add the doxygen directive
    content.append(f'.. {directive}:: {class_name}')
    content.append(f'   :project: {project_name}')
    content.append('   :members:')
    content.append('   :undoc-members:')
    content.append('')
    
    return '\n'.join(content)


def generate_category_index(category, class_list, project_name):
    """Generate an index page for a category with links to individual class pages."""
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
    title = category_titles.get(category, f'{category.title()} API')
    content.append(title)
    content.append('=' * len(title))
    content.append('')
    
    # Add toctree for all classes in this category
    content.append('.. toctree::')
    content.append('   :maxdepth: 1')
    content.append('   :hidden:')
    content.append('')
    
    # Sort classes by name
    class_list.sort(key=lambda x: x[0])
    
    for class_name, refid in class_list:
        # Generate filename from refid (e.g., structcub_1_1DeviceAdjacentDifference)
        filename = refid
        content.append(f'   {filename}')
    
    content.append('')
    content.append('.. list-table::')
    content.append('   :widths: 50 50')
    content.append('   :header-rows: 1')
    content.append('')
    content.append('   * - Class/Struct')
    content.append('     - Description')
    
    for class_name, refid in class_list:
        filename = refid
        content.append(f'   * - :doc:`{class_name} <{filename}>`')
        content.append('     - ')  # Description would go here if available
    
    return '\n'.join(content)



def generate_namespace_api_page(project_name, items):
    """Generate a comprehensive namespace API reference page."""
    content = []
    
    # Determine namespace name
    namespace_name = project_name  # e.g., 'cub', 'thrust', etc.
    
    # Title
    content.append(f'API Reference')
    content.append('=' * len('API Reference'))
    content.append('')
    
    # Add namespace description if available
    content.append(f'Namespace ``{namespace_name}``')
    content.append('-' * (len(namespace_name) + 13))
    content.append('')
    
    # Classes section
    if items['classes']:
        content.append('Classes')
        content.append('~~~~~~~')
        content.append('')
        
        # Sort classes alphabetically
        items['classes'].sort(key=lambda x: x[0].lower())
        for name, refid in items['classes']:
            content.append(f'* :doc:`{name} <{refid}>`')
        content.append('')
    
    # Structs section
    if items['structs']:
        content.append('Structs')
        content.append('~~~~~~~')
        content.append('')
        
        # Sort structs alphabetically
        items['structs'].sort(key=lambda x: x[0].lower())
        for name, refid in items['structs']:
            content.append(f'* :doc:`{name} <{refid}>`')
        content.append('')
    
    # Functions section
    if items['functions']:
        content.append('Functions')
        content.append('~~~~~~~~~')
        content.append('')
        
        # Sort functions alphabetically
        items['functions'].sort(key=lambda x: x[0].lower())
        for name, refid in items['functions']:
            # For functions, we might need to handle overloads differently
            content.append(f'* ``{name}``')
        content.append('')
    
    # Typedefs section
    if items['typedefs']:
        content.append('Type Definitions')
        content.append('~~~~~~~~~~~~~~~~')
        content.append('')
        
        # Sort typedefs alphabetically
        items['typedefs'].sort(key=lambda x: x[0].lower())
        for name, refid in items['typedefs']:
            content.append(f'* ``{name}``')
        content.append('')
    
    # Enums section
    if items['enums']:
        content.append('Enumerations')
        content.append('~~~~~~~~~~~~')
        content.append('')
        
        # Sort enums alphabetically
        items['enums'].sort(key=lambda x: x[0].lower())
        for name, refid in items['enums']:
            content.append(f'* ``{name}``')
        content.append('')
    
    # Variables section
    if items['variables']:
        content.append('Variables')
        content.append('~~~~~~~~~')
        content.append('')
        
        # Sort variables alphabetically
        items['variables'].sort(key=lambda x: x[0].lower())
        for name, refid in items['variables']:
            content.append(f'* ``{name}``')
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
        
        # Extract all items from Doxygen XML
        items = extract_doxygen_items(xml_dir)
        
        # Also extract categorized classes for category pages
        classes = extract_doxygen_classes(xml_dir)
        
        # Determine output directory based on project
        api_dir = None
        if project_name == 'cub':
            api_dir = Path(app.srcdir) / 'cub' / 'api'
        elif project_name == 'thrust':
            api_dir = Path(app.srcdir) / 'thrust' / 'api'
        elif project_name == 'libcudacxx':
            api_dir = Path(app.srcdir) / 'libcudacxx' / 'api'
        elif project_name == 'cudax':
            api_dir = Path(app.srcdir) / 'cudax' / 'api'
        
        if not api_dir:
            continue
        
        # Create API directory if it doesn't exist
        api_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual pages for each class/struct
        for name, refid in items['classes'] + items['structs']:
            # Generate individual page
            content = generate_individual_api_page(name, refid, project_name)
            output_file = api_dir / f'{refid}.rst'
            
            # Write the individual class page
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Generated API page: {output_file}")
        
        # Generate the main namespace API reference page
        namespace_content = generate_namespace_api_page(project_name, items)
        namespace_file = api_dir / 'index.rst'
        with open(namespace_file, 'w') as f:
            f.write(namespace_content)
        logger.info(f"Generated namespace API reference: {namespace_file}")
        
        # Also write to the main api.rst file for better visibility
        main_api_file = Path(app.srcdir) / project_name / 'api.rst'
        if main_api_file.exists():
            # Generate the comprehensive API content directly in api.rst
            content = []
            
            # Title
            content.append(f'{project_name.upper()} API Reference')
            content.append('=' * len(f'{project_name.upper()} API Reference'))
            content.append('')
            
            # Add namespace description
            content.append(f'Namespace ``{project_name}``')
            content.append('-' * (len(project_name) + 13))
            content.append('')
            
            # Classes section
            if items['classes']:
                content.append('Classes')
                content.append('~~~~~~~')
                content.append('')
                
                # Sort classes alphabetically
                items['classes'].sort(key=lambda x: x[0].lower())
                for name, refid in items['classes']:
                    content.append(f'* :doc:`{name} <api/{refid}>`')
                content.append('')
            
            # Structs section
            if items['structs']:
                content.append('Structs')
                content.append('~~~~~~~')
                content.append('')
                
                # Sort structs alphabetically
                items['structs'].sort(key=lambda x: x[0].lower())
                for name, refid in items['structs']:
                    content.append(f'* :doc:`{name} <api/{refid}>`')
                content.append('')
            
            # Functions section
            if items['functions']:
                content.append('Functions')
                content.append('~~~~~~~~~')
                content.append('')
                
                # Sort functions alphabetically
                items['functions'].sort(key=lambda x: x[0].lower())
                for name, refid in items['functions']:
                    content.append(f'* ``{name}``')
                content.append('')
            
            # Typedefs section
            if items['typedefs']:
                content.append('Type Definitions')
                content.append('~~~~~~~~~~~~~~~~')
                content.append('')
                
                # Sort typedefs alphabetically
                items['typedefs'].sort(key=lambda x: x[0].lower())
                for name, refid in items['typedefs']:
                    content.append(f'* ``{name}``')
                content.append('')
            
            # Enums section
            if items['enums']:
                content.append('Enumerations')
                content.append('~~~~~~~~~~~~')
                content.append('')
                
                # Sort enums alphabetically
                items['enums'].sort(key=lambda x: x[0].lower())
                for name, refid in items['enums']:
                    content.append(f'* ``{name}``')
                content.append('')
            
            # Variables section
            if items['variables']:
                content.append('Variables')
                content.append('~~~~~~~~~')
                content.append('')
                
                # Sort variables alphabetically
                items['variables'].sort(key=lambda x: x[0].lower())
                for name, refid in items['variables']:
                    content.append(f'* ``{name}``')
                content.append('')
            
            
            # Write the comprehensive content directly to api.rst
            with open(main_api_file, 'w') as f:
                f.write('\n'.join(content))
            logger.info(f"Updated main API reference: {main_api_file}")
        
        # Generate category index pages (for backward compatibility)
        for category, class_list in classes.items():
            if class_list:
                content = generate_category_index(category, class_list, project_name)
                output_file = api_dir / f'{category}.rst'
                
                # Write the category index
                with open(output_file, 'w') as f:
                    f.write(content)
                logger.info(f"Generated category index: {output_file}")


def setup(app: Sphinx):
    """Setup the extension."""
    app.connect('config-inited', generate_api_docs)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }