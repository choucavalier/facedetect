all:
	rm -rf build
	mkdir build
	cd build; cmake ..; make
	@echo ""
	@echo "---------- compilation done ----------"
	@echo ""
	@make --no-print-directory -j4 run

run:
	./build/train data/positive data/negative /tmp/classifier.txt
	#./build/detect gfx/test.jpg classifiers/dumb.txt

.PHONY: test
test:
	@make --no-print-directory test_io

test_%:
	@./build/$@
