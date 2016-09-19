all:
	rm -rf build
	mkdir build
	cd build; cmake ..; make -j4
	@echo ""
	@echo "---------- compilation done ----------"
	@echo ""
	@make --no-print-directory run

run:
	./build/detect gfx/test.jpg classifiers/dumb.txt

.PHONY: test
test:
	@make --no-print-directory test_io

test_%:
	@./build/$@
