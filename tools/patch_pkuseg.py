#!/usr/bin/env python3
"""
Patch pkuseg setup.py for Python 3.11 compatibility.
Removes longintrepr.h dependency from generated .cpp files.
"""
import re
import os
import sys

def patch_setup_py(setup_py_path='setup.py'):
    """Patch setup.py to add custom build_ext that removes longintrepr.h"""
    
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Custom build_ext class that patches .cpp files
    build_ext_code = '''
from setuptools.command.build_ext import build_ext
from Cython.Distutils import build_ext as cython_build_ext
import os
import re
import glob

class Python311BuildExt(cython_build_ext):
    """Custom build_ext that patches .cpp files for Python 3.11 compatibility"""
    
    def build_extensions(self):
        # First build extensions (this generates .cpp files)
        super().build_extensions()
        # Then patch all .cpp files to remove longintrepr.h
        self._patch_cpp_files()
    
    def _patch_cpp_files(self):
        """Find and patch all .cpp files to remove longintrepr.h"""
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.cpp'):
                    cpp_path = os.path.join(root, file)
                    try:
                        with open(cpp_path, 'r', encoding='utf-8') as f:
                            c = f.read()
                        if 'longintrepr.h' in c:
                            # Remove longintrepr.h include
                            c = re.sub(r'#include\\s+["<]longintrepr\\.h[">]\\s*\\n', '', c)
                            with open(cpp_path, 'w', encoding='utf-8') as f:
                                f.write(c)
                            print(f"Patched {cpp_path} for Python 3.11")
                    except Exception as e:
                        print(f"Warning: Could not patch {cpp_path}: {e}")
'''
    
    # Insert the custom build_ext class after imports
    if 'from Cython' in content or 'from setuptools' in content:
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if 'from Cython' in line or ('from setuptools' in line and 'import' in line):
                insert_pos = i + 1
                break
        
        if insert_pos > 0:
            lines.insert(insert_pos, build_ext_code)
            content = '\n'.join(lines)
            
            # Add cmdclass to setup() call
            if 'cmdclass' not in content:
                # Find setup( call and add cmdclass
                content = re.sub(
                    r'(setup\s*\()',
                    r'\1\n    cmdclass={"build_ext": Python311BuildExt},',
                    content,
                    count=1
                )
    
    # Write patched setup.py
    with open(setup_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully patched {setup_py_path} for Python 3.11 compatibility")
    return True

if __name__ == '__main__':
    setup_py = sys.argv[1] if len(sys.argv) > 1 else 'setup.py'
    if os.path.exists(setup_py):
        patch_setup_py(setup_py)
    else:
        print(f"Error: {setup_py} not found")
        sys.exit(1)

