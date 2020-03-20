all: Kokkos_Core.cpp Kokkos_Core.hpp Kokkos_Atomic.hpp
	dpcpp -c Kokkos_Core.cpp -I./ 
	ar rcs libkokkos.a Kokkos_Core.o

clean:
	rm -f *.o *.a

