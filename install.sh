# Install libs

export CC=$(which gcc-14)
export CXX=$(which g++-14)
export CUDAHOSTCXX=$(which g++-14)
pip install --no-build-isolation submodules/diff-plane-rasterization
pip install submodules/simple-knn