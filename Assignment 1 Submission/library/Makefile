CC = g++

CFLAGS = -std=c++11

libstamp.so: stamp.o
	$(CC) $(CFLAGS) -shared -o libstamp.so stamp.o

stamp.o: stamp.cpp
	$(CC) $(CFLAGS) -fPIC -c stamp.cpp

clean:
	rm stamp.o libstamp.so