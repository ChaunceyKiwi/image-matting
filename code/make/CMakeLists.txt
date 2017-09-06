cmake_minimum_required(VERSION 3.7)
project(imageMatting)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        ../src/main.cpp
        ../src/AlphaCalculator.cpp
        ../src/Image.cpp
        ../src/ImagePrinter.cpp
        ../src/ImageReader.cpp
        ../src/LaplacianCalculator.cpp
        ../src/MattingPerformer.cpp
        ../src/SparseMatrixEquationSolver.cpp)

include_directories(../include)
include_directories(../SuiteSparse/include)

add_executable(imageMatting ${SOURCE_FILES})

find_package( OpenCV REQUIRED )
set (CMAKE_CXX_FLAGS "-lcurl")
target_link_libraries( imageMatting ${OpenCV_LIBS} )

file(GLOB LIBRARIES "../SuiteSparse/lib/*.dylib")
message("LIBRARIES = ${LIBRARIES}")
target_link_libraries(imageMatting ${LIBRARIES})