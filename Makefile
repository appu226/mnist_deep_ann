all:	clean cmake build

clean:
	rm -rf temp/build

cmake:
	cmake -S src -B temp/build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

build:
	make -C temp/build -j

run_visualizer_app:
	temp/build/bin/visualizer_app/visualizer_app --help
