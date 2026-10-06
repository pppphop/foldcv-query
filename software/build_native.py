import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;pkg=root/"src/foldcv_query"
windows=sys.platform=="win32"
compiler=os.environ.get("FOLDCV_CXX","g++")
flags=["-O3","-std=c++17","-fno-fast-math","-ffp-contract=off","-shared"]
flags+= ["-static","-static-libgcc","-static-libstdc++"] if windows else ["-fPIC"]
subprocess.run([compiler,*flags,str(pkg/"weighted.cpp"),"-o",str(pkg/("weighted.dll" if windows else "weighted.so"))],check=True)
