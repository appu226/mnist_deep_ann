@echo off

set BD=@NATIVE_BINARY_PATH@
set BC=@CMAKE_BUILD_TYPE@
set PATH=%PATH%;%BD%\lib\ann\%BC%
set PATH=%PATH%;%BD%\lib\command_line_options\%BC%
set PATH=%PATH%;%BD%\lib\mnist_data\%BC%
set PATH=%PATH%;%BD%\bin\visualizer_app\%BC%
set PATH=%PATH%;%BD%\test\ann_test\%BC%