all:
	rm -rf build
	mkdir build
	cd build; cmake ..; make

test:
	@./build/facedetect test.jpg
