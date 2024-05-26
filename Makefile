sources = Matrix.cpp
executable = Matrix.so
CXX = g++
CXXFLAGS = -std=c++14 -shared -fPIC `python3 -m pybind11 --includes` `python3-config --includes --ldflags`

all: $(sources)
	$(CXX) $(CXXFLAGS) $(shell python3 -m pybind11 --includes) $(sources) -o $(executable)

clear:
	rm -rf $(executable) __pycache__ .pytest* *.so

test:
	pytest test_Matrix.py