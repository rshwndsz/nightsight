#!/usr/bin/env python
"""NightSight

TODO
Here be dragons
"""

DOCLINES = __doc__.split("\n")

import os
import sys
import subprocess
from setuptools import setup


if sys.version_info[0:2] >= (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

MAJOR = 1
MINOR = 10
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of nightsight.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('nightsight/version.py'):
        # must be a source distribution, use existing version file
        try:
            from nightsight.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "nightsight/version.py and the build directory " \
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='nightsight/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM NIGHTIGHT SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(
            cnt % {
                'version': VERSION,
                'full_version': FULLVERSION,
                'git_revision': GIT_REVISION,
                'isrelease': str(ISRELEASED)
            })
    finally:
        a.close()


if __name__ == '__main__':
    # Rewrite the version file everytime
    write_version_py()

    setup(
        name='nightsight',
        version=get_version_info()[0]
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        author="Russel",
        url="https://github.com/rshwndsz/nightsight",
        packages=find_packages(include=['nightsight', 'nightsight.*']),
        install_requires=[
            'torch>=1.5',
            'torchvision',
            'pytorch_lightning',
            'numpy',
            'matplotlib',
            'pillow',
        ],
        tests_require=['pytest'],
    )
