all:
	rm -rf build
	mkdir build
	cd build; cmake ..; make -j4
	@echo ""
	@echo "---------- compilation done ----------"
	@echo ""
	./build/facedetect gfx/test.jpg
