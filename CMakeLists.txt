cmake_minimum_required(VERSION 3.0.0)
project(BackGS VERSION 0.1.0)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/Activators/*.cpp" "src/Losses/*.cpp" "src/Optimizers/*.cpp" main.cpp)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(BackGS ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(BackGS PRIVATE Eigen3::Eigen)
target_link_libraries(BackGS PRIVATE nlohmann_json::nlohmann_json)
