rmdir /S build
mkdir build
cd build
cmake .. 
cmake --build . --config Release --verbose
cd ..