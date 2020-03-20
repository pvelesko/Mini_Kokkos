#ifndef CPP_EPUSHSYCL_HPP
#define CPP_EPUSHSYCL_HPP

#include "CL/sycl.hpp"

#define KOKKOS_INLINE_FUNCTION inline
#define KOKKOS_LAMBDA [=]

auto exception_handler = [] (cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
  std::rethrow_exception(e);
    } catch(cl::sycl::exception const& e) {
  std::cout << "Caught asynchronous SYCL exception:\n"
        << e.what() << std::endl;
    }
  }
};


extern "C" cl::sycl::queue q;
extern "C" cl::sycl::device dev;
extern "C" cl::sycl::context ctx;
extern "C" int _max_threads;
using namespace cl::sycl;
namespace Kokkos {

    inline void fence() {};
    typedef struct{} HostSpace;
    typedef struct{} Serial;
    typedef struct{} OpenMP;
    typedef struct{} LayoutLeft;
    typedef struct{} LayoutRight;
    typedef struct{} Unmanaged;



    template<class ExSpace, class MemSpace>
    class Device {};

    template<class T, class ... args>
    class View {
      typedef typename std::remove_pointer<T >::type T1;
      typedef typename std::remove_pointer<T1>::type T2;
      typedef typename std::remove_pointer<T2>::type T3;
      typedef typename std::remove_pointer<T3>::type T4;
      typedef typename std::remove_pointer<T4>::type T5;
      typedef typename std::remove_pointer<T5>::type T6;
      typedef typename std::remove_pointer<T6>::type T7;
      int _size;
      int _maxDim;
      T7* _data;
      int* dims;
      int* cnt;
      void alloc() {
        _data = static_cast<T7*>(malloc_device(_size * sizeof(T7), dev, ctx));
        cnt = static_cast<int*>(malloc_device(1 * sizeof(int), dev, ctx));
        dims = static_cast<int*>(malloc_device(6 * sizeof(int), dev, ctx));
        cnt[0] = 1;
        if (_data == NULL) throw ("Failure to allocate memory!\n");
      }

      public:
      View() {};
#ifdef __SYCL_DEVICE_ONLY__
#else
      View(const View& other) {
        _size = other._size;
        _maxDim = other._maxDim;
        _data = other._data;
        dims = other.dims;
        cnt = other.cnt;
        cnt[0]++;
      }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#else
      ~View() {
        //std::cout << "Free!\n";
        cnt[0]--;
        if (cnt[0] == 1)
          free(_data, ctx);
          free(dims, ctx);
          free(cnt, ctx);
      };
#endif
      View(std::string name, T1 *ptr, int dim0) {
        _size = dim0; 
        _maxDim = 0;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
      };
      View(std::string name, T2 *ptr, int dim0, int dim1) {
        _size = dim0 * dim1; 
        _maxDim = 1;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
      };
      View(std::string name, T3 *ptr, int dim0, int dim1, int dim2) {
        _size = dim0 * dim1 * dim2; 
        _maxDim = 2;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
      };
      View(std::string name, T4 *ptr, int dim0, int dim1, int dim2, int dim3) {
        _size = dim0 * dim1 * dim2 * dim3; 
        _maxDim = 3;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
      };
      View(std::string name, T5 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4; 
        _maxDim = 4;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
      };
      View(std::string name, T6 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5; 
        _maxDim = 5;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
      };
      View(std::string name, T7 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5, int dim6) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6; 
        _maxDim = 6;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
        dims[6] = dim6;
      };

      View(std::string name, int dim0) {
        _size = dim0; 
        _maxDim = 0;
        alloc(); 
        dims[0] = dim0;
      };
      View(std::string name, int dim0, int dim1) {
        _size = dim0 * dim1; 
        _maxDim = 1;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
      };
      View(std::string name, int dim0, int dim1, int dim2) {
        _size = dim0 * dim1 * dim2; 
        _maxDim = 2;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
      };
      View(std::string name, int dim0, int dim1, int dim2, int dim3) {
        _size = dim0 * dim1 * dim2 * dim3 ; 
        _maxDim = 3;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
      };
      View(std::string name, int dim0, int dim1, int dim2, int dim3, int dim4) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4; 
        _maxDim = 4;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
      };
      View(std::string name, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5; 
        _maxDim = 5;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
       };
      View(std::string name, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5, int dim6) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6; 
        _maxDim = 6;
        alloc(); 
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
        dims[6] = dim6;
      };

      View(T1 *ptr, int dim0) {
        _size = dim0; 
        _maxDim = 0;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
      };
      View(T2 *ptr, int dim0, int dim1) {
        _size = dim0 * dim1; 
        _maxDim = 1;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
      };
      View(T3 *ptr, int dim0, int dim1, int dim2) {
        _size = dim0 * dim1 * dim2; 
        _maxDim = 2;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
      };
      View(T4 *ptr, int dim0, int dim1, int dim2, int dim3) {
        _size = dim0 * dim1 * dim2 * dim3; 
        _maxDim = 3;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
      };
      View(T5 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4; 
        _maxDim = 4;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
      };
      View(T6 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5; 
        _maxDim = 5;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
       };
      View(T7 *ptr, int dim0, int dim1, int dim2, int dim3, int dim4, int dim5, int dim6) {
        _size = dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6; 
        _maxDim = 6;
        alloc(); 
        memcpy(_data, ptr, _size * sizeof(T7));
        dims[0] = dim0;
        dims[1] = dim1;
        dims[2] = dim2;
        dims[3] = dim3;
        dims[4] = dim4;
        dims[5] = dim5;
        dims[6] = dim6;
      };

      T1& operator[](int id0) const {
        return _data[id0];
      }
      T1& operator()(int id0) const {
        return _data[id0];
      }
      T2& operator()(int id1, int id0) const {
        return _data[
            id1 * dims[0] 
          + id0];
      }
      T3& operator()(int id2, int id1, int id0) const {
        return _data[
            id2 * dims[1] * dims[0]
          + id1 * dims[0]
          + id0];
      }
      T4& operator()(int id3, int id2, int id1, int id0) const {
        return _data[
            id3 * dims[2] * dims[1] * dims[0] 
          + id2 * dims[1] * dims[0]
          + id1 * dims[0]
          + id0];
      }
      T5& operator()(int id4, int id3, int id2, int id1, int id0) const {
        return _data[
            id4 * dims[3] + dims[2] * dims[1] * dims[0]
          + id3 * dims[2] * dims[1] * dims[0] 
          + id2 * dims[1] * dims[0]
          + id1 * dims[0]
          + id0];
      }
      T6& operator()(int id5, int id4, int id3, int id2, int id1, int id0) const {
        return _data[
            id5 * dims[4] * dims[3] * dims[2] * dims [1] * dims[0]
          + id4 * dims[3] + dims[2] * dims[1] * dims[0]
          + id3 * dims[2] * dims[1] * dims[0] 
          + id2 * dims[1] * dims[0]
          + id1 * dims[0]
          + id0];
      }

      int extent(int dim) const {return dims[dim];};
      int size() const {return _size;};
      T7* data() const {return _data;};
    };

    template<class T, class ... args>
    View<T, args...> create_mirror_view(View<T, args...>& other) {
      return other;
    };

//    template<class T>
//    class create_mirror_view {
//      typedef typename std::remove_pointer<T >::type T1;
//      typedef typename std::remove_pointer<T1>::type T2;
//      typedef typename std::remove_pointer<T2>::type T3;
//      typedef typename std::remove_pointer<T3>::type T4;
//      typedef typename std::remove_pointer<T4>::type T5;
//      typedef typename std::remove_pointer<T5>::type T6;
//      typedef typename std::remove_pointer<T6>::type T7;
//      public:
//      static T7* operator()(View<T>& other) {
//        return other.data();
//      }
//      //delete create_mirror_view();
//    };

    template<class T>
    class MemoryTraits {
    };


    template<class T, class ... args, class ... arg1>
    static void deep_copy(const View<T, args...> &target, const View<T, arg1...> &source) {
      for(int i = 0; i < source.size(); i++)
        target.data()[i] = source.data()[i];
    };
    template<class T, class ... args>
    static void deep_copy(const View<T, args...> &target, const double d) {
       for(int i = 0; i < target.size(); i++)
         target.data()[i] = d;
    };

    template<class ExSpace>
    class RangePolicy {
      public:
      int start;
      int end;
      RangePolicy(int start_in, int end_in) {
        start = start_in;
        end = end_in;
      }
    };


    class Experimental {
      public:
      class WorkItemProperty {
        public:
        const static int HintLightWeight = 0;
      };

    class require {
      public:
      int start;
      int end;
      template<class ExSpace>
      require(RangePolicy<ExSpace> a, int hint_in) {
        start = a.start;
        end = a.end;
      };
    };
    };

    template<class T>
    inline void parallel_for(int n, T lambda_in) {
      std::string name_in("");
      Experimental::require require_in(RangePolicy<HostSpace>(0, n), 0);
      parallel_for(name_in, require_in, lambda_in);
    };

    template<class T>
    inline void parallel_for(std::string name_in, Experimental::require require_in, int hint_in, T lambda_in) {
      parallel_for(name_in, require_in, lambda_in);
    };

    template<class T>
    inline void parallel_for(std::string name_in, Experimental::require require_in, T lambda_in) {
      //std::cout << "GPU Run: " << name_in << std::endl;
      int start = require_in.start;
      int end = require_in.end;
      q.submit([&](handler& cgh) {
//        cgh.parallel_for(range<1>(end), [=](item<1> idx) [[cl::intel_reqd_sub_group_size(8)]] {
        cgh.parallel_for(range<1>(end), [=](item<1> idx) {
          int i = idx[0];
          lambda_in(i); 
        }); // parallel
      }); //queue

    };

   static inline void initialize() {
     std::string env(std::getenv("SYCL_DEVICE"));
     std::cout << "Using SYCL_DEVICE = " << env << std::endl;
     if (!env.compare("gpu") or !env.compare("GPU")) {
       q = cl::sycl::queue(cl::sycl::gpu_selector{}, exception_handler);
     } else if (!env.compare("cpu") or !env.compare("CPU")) {
       q = cl::sycl::queue(cl::sycl::cpu_selector{}, exception_handler);
     } else if (!env.compare("host") or !env.compare("HOST")) {
       q = cl::sycl::queue(cl::sycl::host_selector{}, exception_handler);
     }

     dev = q.get_device();
     ctx = q.get_context();
     std::cout << q.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
     // Set up Max Total Threads
     auto _num_groups = q.get_device().get_info<info::device::max_compute_units>();
     auto _work_group_size =q.get_device().get_info<info::device::max_work_group_size>();
     _max_threads = (int)(_num_groups * _work_group_size);
     
  }  
  static inline void finalize() {};
};

#endif
