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


def extract_param_summary(params):
    """Extract a simplified parameter summary for section headers."""
    if not params:
        return ""
    
    # Remove template details and namespaces for brevity
    params = params.strip()
    if params.startswith('(') and params.endswith(')'):
        params = params[1:-1]
    
    # If empty after removing parentheses
    if not params.strip():
        return ""
    
    # Split by comma (handling nested templates/parentheses)
    param_parts = []
    depth = 0
    current = []
    
    for char in params:
        if char in '<([':
            depth += 1
        elif char in '>)]':
            depth -= 1
        elif char == ',' and depth == 0:
            param_parts.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    
    if current:
        param_parts.append(''.join(current).strip())
    
    # Extract parameter names
    param_names = []
    
    for param in param_parts:
        param = param.strip()
        
        # Special case for execution policy - shorten for readability
        if 'execution_policy_base' in param:
            param_names.append('exec')
        else:
            # Split by spaces and find the parameter name (typically the last word)
            words = param.split()
            if words:
                # The parameter name is the last word
                param_name = words[-1]
                # Clean up reference/pointer markers
                param_name = param_name.strip('&*,')
                
                # Just use the parameter name as-is
                if param_name:
                    param_names.append(param_name)
    
    return ', '.join(param_names)


def extract_function_signatures(func_name, refids, xml_dir, namespace=''):
    """Extract exact function signatures from Doxygen XML for overloaded functions."""
    signatures = []
    xml_path = Path(xml_dir)
    
    if not xml_path.exists():
        return signatures
    
    # Parse both namespace and group XML files to get function signatures
    # Functions can be defined in either location (often in group_*.xml for thrust/cub)
    xml_files = list(xml_path.glob('namespace*.xml')) + list(xml_path.glob('group*.xml'))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find all function members with matching refids
            for memberdef in root.findall('.//memberdef[@kind="function"]'):
                member_refid = memberdef.get('id')
                if member_refid in refids:
                    # Get the function name
                    name_elem = memberdef.find('name')
                    if name_elem is not None and name_elem.text == func_name:
                        # Get the exact args string and definition
                        argsstring_elem = memberdef.find('argsstring')
                        definition_elem = memberdef.find('definition')
                        
                        if argsstring_elem is not None and argsstring_elem.text:
                            # Get the full qualified name from definition if available
                            if definition_elem is not None and definition_elem.text:
                                # Definition includes namespace and return type
                                # Extract just the qualified function name
                                definition = definition_elem.text
                                # Look for the function name in the definition
                                if '::' in definition and func_name in definition:
                                    # Extract namespace::function from definition
                                    parts = definition.split()
                                    for part in parts:
                                        if func_name in part and '::' in part:
                                            qualified_name = part
                                            break
                                    else:
                                        qualified_name = f"{namespace}::{func_name}" if namespace else func_name
                                else:
                                    qualified_name = f"{namespace}::{func_name}" if namespace else func_name
                            else:
                                qualified_name = f"{namespace}::{func_name}" if namespace else func_name
                            
                            # Build the complete signature for breathe
                            full_signature = qualified_name + argsstring_elem.text.strip()
                            signatures.append((member_refid, full_signature))
        except Exception as e:
            logger.debug(f"Failed to extract signatures from {xml_file}: {e}")
    
    return signatures


def extract_doxygen_items(xml_dir):
    """Extract all items (classes, structs, functions, etc.) from Doxygen XML."""
    items = {
        'classes': [],
        'structs': [],
        'functions': [],
        'typedefs': [],
        'enums': [],
        'variables': [],
        'function_groups': {}  # Group functions by name for overloads
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
        
        # Get the namespace compound (e.g., cub, thrust, cuda::experimental)
        namespace_compounds = []
        for compound in root.findall('.//compound[@kind="namespace"]'):
            name = compound.find('name').text
            # Match primary namespaces and nested namespaces for cudax
            if name in ['cub', 'thrust', 'cuda'] or name.startswith('cuda::experimental'):
                namespace_compounds.append(compound)
        
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
        
        # Extract functions, typedefs, enums, and variables from namespaces
        for namespace_compound in namespace_compounds:
            namespace_name = namespace_compound.find('name').text
            
            for member in namespace_compound.findall('member[@kind="function"]'):
                name = member.find('name').text
                refid = member.get('refid')
                # Only include full namespace for nested namespaces (cudax)
                # For simple namespaces like 'thrust', 'cub', just use the function name
                if namespace_name and '::' in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items['functions'].append((full_name, refid))
                
                # Also track function groups for overloads
                if full_name not in items['function_groups']:
                    items['function_groups'][full_name] = []
                items['function_groups'][full_name].append(refid)
            
            for member in namespace_compound.findall('member[@kind="typedef"]'):
                name = member.find('name').text
                refid = member.get('refid')
                # Only include full namespace for nested namespaces
                if namespace_name and '::' in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items['typedefs'].append((full_name, refid))
            
            for member in namespace_compound.findall('member[@kind="enum"]'):
                name = member.find('name').text
                refid = member.get('refid')
                # Only include full namespace for nested namespaces
                if namespace_name and '::' in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items['enums'].append((full_name, refid))
            
            for member in namespace_compound.findall('member[@kind="variable"]'):
                name = member.find('name').text
                refid = member.get('refid')
                # Only include full namespace for nested namespaces
                if namespace_name and '::' in namespace_name:
                    full_name = f"{namespace_name}::{name}"
                else:
                    full_name = name
                items['variables'].append((full_name, refid))
    
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


def check_function_in_namespace(member_name, xml_dir, namespace):
    """Check if a function is defined in the namespace XML (not just referenced)."""
    namespace_xml = os.path.join(xml_dir, f'namespace{namespace}.xml')
    if not os.path.exists(namespace_xml):
        return False
    
    try:
        tree = ET.parse(namespace_xml)
        root = tree.getroot()
        
        # Look for actual function definitions, not just references
        for memberdef in root.findall('.//memberdef[@kind="function"]'):
            name_elem = memberdef.find('name')
            if name_elem is not None and name_elem.text == member_name:
                # Check if it has a definition (not just a reference)
                definition = memberdef.find('definition')
                if definition is not None and definition.text:
                    return True
        return False
    except:
        return False

def generate_member_api_page(member_name, member_type, project_name, refid=None, overload_refids=None, xml_dir=None):
    """Generate RST content for a single function/typedef/enum/variable API page."""
    content = []
    
    # Add title
    content.append(f'``{member_name}``')
    content.append('=' * (len(member_name) + 4))
    content.append('')
    
    # Map member types to Doxygen directives
    directive_map = {
        'function': 'doxygenfunction',
        'typedef': 'doxygentypedef',
        'enum': 'doxygenenum',
        'variable': 'doxygenvariable'
    }
    
    directive = directive_map.get(member_type, 'doxygenfunction')
    
    # For thrust and cub, we need to use the namespace-qualified name
    # For cudax, the member_name already includes the namespace
    if project_name in ['thrust', 'cub']:
        # If the member_name doesn't already include the namespace, add it
        if '::' not in member_name:
            qualified_name = f'{project_name}::{member_name}'
        else:
            qualified_name = member_name
    else:
        qualified_name = member_name
    
    if member_type == 'function' and overload_refids:
        # First check if function is actually defined in namespace XML
        is_in_namespace = False
        if xml_dir:
            is_in_namespace = check_function_in_namespace(member_name, xml_dir, project_name)
        
        # Check if functions are in a group
        is_group_function = False
        group_name = None
        
        if overload_refids and '_1' in overload_refids[0]:
            parts = overload_refids[0].split('_1')
            if parts[0].startswith('group__'):
                is_group_function = True
                # Get the group refid (e.g., 'group__stream__compaction')
                group_refid = parts[0]
                
                # Look up the actual group name from index.xml
                group_name = None
                if xml_dir:
                    index_xml = os.path.join(xml_dir, 'index.xml')
                    if os.path.exists(index_xml):
                        try:
                            tree = ET.parse(index_xml)
                            root = tree.getroot()
                            # Find the compound with this refid
                            compound = root.find(f'.//compound[@refid="{group_refid}"]')
                            if compound is not None:
                                name_elem = compound.find('name')
                                if name_elem is not None:
                                    group_name = name_elem.text
                        except:
                            pass
                
                # Fallback: remove 'group__' prefix if lookup fails
                if not group_name:
                    group_name = group_refid[7:]
        
        if not is_in_namespace and is_group_function and group_name:
            # For group functions, just use doxygengroup
            # The TOC will be generated from the actual function signatures
            content.append(f'.. doxygengroup:: {group_name}')
            content.append(f'   :project: {project_name}')
            content.append(f'   :members:')
            content.append('')
        elif len(overload_refids) > 1 and xml_dir:
            # For functions with multiple overloads in namespace, extract signatures
            signatures = extract_function_signatures(member_name, overload_refids, xml_dir, namespace=project_name)
            
            if signatures:
                content.append('Overloads')
                content.append('---------')
                content.append('')
                
                for idx, (refid, full_sig) in enumerate(signatures, 1):
                    # Extract just the parameter list from the full signature
                    # Look for the function name and get everything after it
                    if member_name in full_sig:
                        sig_idx = full_sig.rfind(member_name)
                        if sig_idx != -1:
                            params = full_sig[sig_idx + len(member_name):].strip()
                            
                            # Create a simplified signature for the section header
                            # Extract key parameter types for identification
                            param_summary = extract_param_summary(params)
                            
                            # Add a section header for this overload
                            content.append(f'``{member_name}({param_summary})``')
                            content.append('^' * (len(member_name) + len(param_summary) + 6))
                            content.append('')
                            
                            # Use doxygenfunction with the specific parameter signature and qualified name
                            content.append(f'.. doxygenfunction:: {qualified_name}{params}')
                            content.append(f'   :project: {project_name}')
                            content.append(f'   :no-link:')
                            content.append('')
            else:
                # Fallback to simple directive with qualified name
                content.append(f'.. {directive}:: {qualified_name}')
                content.append(f'   :project: {project_name}')
                content.append('')
        else:
            # Single function with qualified name
            content.append(f'.. {directive}:: {qualified_name}')
            content.append(f'   :project: {project_name}')
            content.append('')
    elif member_type == 'function':
        # For single functions or when we don't have xml_dir, use qualified name
        content.append(f'.. {directive}:: {qualified_name}')
        content.append(f'   :project: {project_name}')
        content.append('')
    else:
        # For other types, use the qualified name
        content.append(f'.. {directive}:: {qualified_name}')
        content.append(f'   :project: {project_name}')
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



def generate_namespace_api_page(project_name, items, title=None, doc_prefix=''):
    """Generate a comprehensive namespace API reference page."""
    content = []
    
    # Determine namespace name
    namespace_name = project_name  # e.g., 'cub', 'thrust', etc.
    
    # Title - use provided title or default
    if not title:
        title = f'{project_name.upper()} API Reference'
    
    content.append(title)
    content.append('=' * len(title))
    content.append('')
    
    # Add namespace description
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
            if doc_prefix:
                content.append(f'* :doc:`{name} <{doc_prefix}{refid}>`')
            else:
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
            if doc_prefix:
                content.append(f'* :doc:`{name} <{doc_prefix}{refid}>`')
            else:
                content.append(f'* :doc:`{name} <{refid}>`')
        content.append('')
    
    # Functions section
    if items.get('function_groups'):
        content.append('Functions')
        content.append('~~~~~~~~~')
        content.append('')
        
        # Sort functions alphabetically by name
        sorted_functions = sorted(items['function_groups'].keys(), key=lambda x: x.lower())
        for func_name in sorted_functions:
            # Use the first refid for the link
            first_refid = items['function_groups'][func_name][0]
            if doc_prefix:
                content.append(f'* :doc:`{func_name} <{doc_prefix}{first_refid}>`')
            else:
                content.append(f'* :doc:`{func_name} <{first_refid}>`')
        content.append('')
    
    # Typedefs section
    if items['typedefs']:
        content.append('Type Definitions')
        content.append('~~~~~~~~~~~~~~~~')
        content.append('')
        
        # Sort typedefs alphabetically
        items['typedefs'].sort(key=lambda x: x[0].lower())
        for name, refid in items['typedefs']:
            if doc_prefix:
                content.append(f'* :doc:`{name} <{doc_prefix}{refid}>`')
            else:
                content.append(f'* :doc:`{name} <{refid}>`')
        content.append('')
    
    # Enums section
    if items['enums']:
        content.append('Enumerations')
        content.append('~~~~~~~~~~~~')
        content.append('')
        
        # Sort enums alphabetically
        items['enums'].sort(key=lambda x: x[0].lower())
        for name, refid in items['enums']:
            if doc_prefix:
                content.append(f'* :doc:`{name} <{doc_prefix}{refid}>`')
            else:
                content.append(f'* :doc:`{name} <{refid}>`')
        content.append('')
    
    # Variables section
    if items['variables']:
        content.append('Variables')
        content.append('~~~~~~~~~')
        content.append('')
        
        # Sort variables alphabetically
        items['variables'].sort(key=lambda x: x[0].lower())
        for name, refid in items['variables']:
            if doc_prefix:
                content.append(f'* :doc:`{name} <{doc_prefix}{refid}>`')
            else:
                content.append(f'* :doc:`{name} <{refid}>`')
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
        
        # Generate individual pages for functions (one per unique function name)
        # Use function_groups to handle overloads
        function_groups = items.get('function_groups', {})
        for func_name in function_groups:
            # Use the first refid as the filename for consistency
            first_refid = function_groups[func_name][0]
            content = generate_member_api_page(func_name, 'function', project_name, 
                                              refid=first_refid,
                                              overload_refids=function_groups[func_name],
                                              xml_dir=xml_dir)
            output_file = api_dir / f'{first_refid}.rst'
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Generated function API page: {output_file}")
        
        # Generate individual pages for typedefs
        for name, refid in items['typedefs']:
            content = generate_member_api_page(name, 'typedef', project_name, refid)
            output_file = api_dir / f'{refid}.rst'
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Generated typedef API page: {output_file}")
        
        # Generate individual pages for enums
        for name, refid in items['enums']:
            content = generate_member_api_page(name, 'enum', project_name, refid)
            output_file = api_dir / f'{refid}.rst'
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Generated enum API page: {output_file}")
        
        # Generate individual pages for variables
        for name, refid in items['variables']:
            content = generate_member_api_page(name, 'variable', project_name, refid)
            output_file = api_dir / f'{refid}.rst'
            with open(output_file, 'w') as f:
                f.write(content)
            logger.info(f"Generated variable API page: {output_file}")
        
        # Generate the main namespace API reference page for api/index.rst
        namespace_content = generate_namespace_api_page(project_name, items)
        namespace_file = api_dir / 'index.rst'
        with open(namespace_file, 'w') as f:
            f.write(namespace_content)
        logger.info(f"Generated namespace API reference: {namespace_file}")
        
        # Generate auto_api.rst file with comprehensive API listing
        auto_api_file = Path(app.srcdir) / project_name / 'auto_api.rst'
        
        # Determine the title for the auto-generated API page
        if project_name == 'cub':
            title = 'CUB API Reference'
        elif project_name == 'thrust':
            title = 'Thrust: The C++ Parallel Algorithms Library API'
        else:
            title = f'{project_name.upper()} API Reference'
        
        # Generate the auto API content with api/ prefix for links
        auto_api_content = generate_namespace_api_page(project_name, items, title=title, doc_prefix='api/')
        with open(auto_api_file, 'w') as f:
            f.write(auto_api_content)
        logger.info(f"Generated auto API reference: {auto_api_file}")
        
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