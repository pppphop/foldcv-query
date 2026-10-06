"""Build the package and native comparators in this portable directory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('FOLDCV_CXX', 'g++'))
    args = parser.parse_args()
    env = dict(os.environ, FOLDCV_CXX=args.cxx)
    subprocess.run([sys.executable, str(ROOT/'software/build_native.py')], env=env, check=True)
    flags = ['-O3', '-std=c++17', '-fno-fast-math', '-ffp-contract=off', '-shared']
    flags += ['-static', '-static-libgcc', '-static-libstdc++'] if sys.platform == 'win32' else ['-fPIC']
    suffix = '.dll' if sys.platform == 'win32' else '.so'
    for stem in ('native_cfcausal_flat', 'flat_batch'):
        subprocess.run([args.cxx, *flags, str(ROOT/(stem+'.cpp')), '-o', str(ROOT/(stem+suffix))], check=True)

if __name__ == '__main__':
    main()
