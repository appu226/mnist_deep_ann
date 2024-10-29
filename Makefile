all:	clean cmake build test

clean:
	rm -rf temp/build

cmake:
	cmake -S src -B temp/build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

build:
	make -C temp/build -j

run_visualizer_app:
	temp/build/bin/visualizer_app/visualizer_app --help

fast_test:
	temp/build/test/ann_test/ann_test

test:	build fast_test