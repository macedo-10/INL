project(metavision_sdk_get_started)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)

set (sample metavision_sdk_get_started)
add_executable(${sample} ${sample}.cpp)
target_link_libraries(${sample} MetavisionSDK::core MetavisionSDK::driver MetavisionSDK::ui)