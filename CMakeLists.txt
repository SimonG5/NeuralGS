cmake_minimum_required(VERSION 3.4)
project(BackGS VERSION 0.1.0)

include_directories(include)
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/Activators/*.cpp" "src/Backgammon/*.cpp" "src/Losses/*.cpp" "src/Optimizers/*.cpp" main.cpp)

find_package(OpenMP REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(BackGS ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_compile_options(BackGS PRIVATE -Wall -fopenmp)
target_link_libraries(BackGS PRIVATE -fopenmp)
target_link_libraries(BackGS PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BackGS PRIVATE Eigen3::Eigen)
target_link_libraries(BackGS PRIVATE nlohmann_json::nlohmann_json)
