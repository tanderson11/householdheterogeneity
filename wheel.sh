#!/bin/bash

ZLIB="$(brew --prefix zlib)"
export OPENBLAS="$(brew --prefix openblas)"
export OPENSSL_ROOT_DIR="$(brew --prefix openssl)"
export 	

export LDFLAGS="-L$ZLIB/lib -L$OPENSSL_ROOT_DIR/lib -L/opt/homebrew/opt/readline/lib -L$OPENBLAS/lib"
export CPPFLAGS="-I$ZLIB/include -I$OPENSSL_ROOT_DIR/include -I/opt/homebrew/opt/readline/include -I$OPENBLAS/include"

pip install wheel

#NUMPY_VERSION="1.20.2"
PYTHON_VERSION="39"
#pip wheel --no-deps --no-cache-dir --log build.log "numpy==$NUMPY_VERSION"
#mv "numpy-$NUMPY_VERSION-cp$PYTHON_VERSION-cp$PYTHON_VERSION-macosx_11_2_arm64.whl" "numpy-$NUMPY_VERSION-cp$PYTHON_VERSION-cp$PYTHON_VERSION-macosx_11_0_arm64.whl"

#pip wheel --no-deps --no-cache-dir --log build.log --no-binary :all "pandas>=1.2"

#SCIPY_VERSION="1.7.0"
#pip wheel --no-deps --no-cache-dir --log build.log "scipy==$SCIPY_VERSION"

#pip wheel --no-deps --no-cache-dir --log build.log "seaborn"

#pip wheel --no-deps --no-cache-dir --log build.log "seaborn"

pip wheel --no-deps --no-cache-dir --log build.log "torch"

#export PYARROW_BUNDLE_ARROW_CPP_HEADERS=0
#export PYARROW_WITH_PARQUET=1
#export ARROW_INCLUDE_DIR=$(brew --prefix apache-arrow)/include
#export ARROW_LIB_DIR=$(brew --prefix apache-arrow)/lib
#pip wheel --no-deps --no-cache-dir --log build.log "pyarrow==4.0.0"



#export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
#export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
