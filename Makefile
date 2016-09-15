all:
	rm -rf build
	mkdir build
	cd build; cmake ..; make
	@echo ""
	@echo "---------- compilation done ----------"
	@echo ""
	./build/facedetect gfx/test.jpg
