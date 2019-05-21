from distutils.core import setup, Extension

swipl_include_dir = '/usr/local/lib/swipl/include'
swipl_lib_dir = '/usr/local/lib/swipl/lib/x86_64-linux'

extra_args=["-std=c++11"]

module = Extension('LogicLayer',
     define_macros = [('MAJOR_VERSION', '1'),
                      ('MINOR_VERSION', '0')],
     language = 'cxx',
     include_dirs = [
          '.',
          '/usr/local/include',
          swipl_include_dir
     ],
     libraries = ['stdc++', 'swipl'],
     library_dirs = [
          '.',
          '/usr/local/lib',
          swipl_lib_dir
     ],
     extra_compile_args=extra_args,
     sources = ['logiclayer_module.cpp'])

setup(name = 'LogicLayer',
     version = '1.0',
     description = 'Package for logic abduction',
     author = 'Abductive Learning',
     author_email = 'abductivelearning@gmail.com',
     url = 'https://docs.python.org/extending/building',
     long_description = '''Nothing yet.''',
     ext_modules = [module],
     classifiers = [
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Topic :: Software Development :: Build Tools',
         'License :: OSI Approved :: MIT License',
         'Programming Language :: Python :: 3.3',
         'Programming Language :: Python :: 3.4',
         'Programming Language :: Python :: 3.5',
     ])
