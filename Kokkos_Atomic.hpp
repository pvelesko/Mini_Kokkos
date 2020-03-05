#ifndef KOKKOS_ATOMIC_HPP
#define KOKKOS_ATOMIC_HPP
namespace Kokkos {
  template<class T>
  inline void atomic_add(T* target, T val) { 
    //std::cout << "atomic_add not yet implemented!" << std::endl;
    //throw();
    *target = val;
  };

  template<class T>
  inline T atomic_fetch_add(T* target, T val) {
    //std::cout << "atomic_fetch_add not yet implemented!" << std::endl;
    //throw();
    auto t = *target;
    *target = val;
    return t;
  };

  inline int atomic_fetch_add(unsigned int* target, int val) {
    //std::cout << "atomic_fetch_add not yet implemented!" << std::endl;
    //throw();
    auto t = *target;
    *target = val;
    return t;
  };
}
#endif
