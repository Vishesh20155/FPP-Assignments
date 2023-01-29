all: fib vector matrix

vector: vector.cpp
	g++ -std=c++11 vector.cpp -o vector -I../library -L../library -lstamp -lpthread
	
matrix: matrix.cpp
	g++ -std=c++11 matrix.cpp -o matrix -I../library -L../library -lstamp -lpthread

fib: fib.cpp
	g++ -std=c++11 fib.cpp -o fib -I../library -L../library -lstamp -lpthread

clean:
	rm fib vector matrix

