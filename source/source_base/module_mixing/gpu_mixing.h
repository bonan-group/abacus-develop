#ifndef GPU_MIXING_H_
#define GPU_MIXING_H_

#include <cstddef>
#include <functional>
#include <vector>

namespace ModuleBase
{
class matrix;
}

namespace Base_Mixing
{

enum class MixingAlgorithm
{
    Broyden,
    Pulay
};

template <typename T>
class GpuMixing;

template <typename T>
class GpuMixingData
{
  public:
    GpuMixingData();
    GpuMixingData(int capacity, std::size_t length);
    ~GpuMixingData();

    void resize(int capacity, std::size_t length);
    void reset();
    void push(const T* data);

    int capacity() const;
    int size() const;
    std::size_t length() const;
    int current_slot() const;
    int index_move(int offset) const;
    T* device_slot(int slot) const;

  private:
    GpuMixingData(const GpuMixingData&) = delete;
    GpuMixingData& operator=(const GpuMixingData&) = delete;

    T* data_;
    int capacity_;
    std::size_t length_;
    int current_slot_;
    int size_;

    friend class GpuMixing<T>;
};

template <typename T>
class GpuMixing
{
  public:
    typedef std::function<void(const T*, int, int, ModuleBase::matrix&)> GramBuilder;
    typedef std::function<void(const T*, const T*, int, std::vector<double>&)> RhsBuilder;
    typedef std::function<void(T*)> Screen;
    typedef std::function<void(T*, const T*, const T*)> Mix;

    GpuMixing(MixingAlgorithm algorithm, int history_depth, double beta);
    ~GpuMixing();

    void reset(std::size_t coefficient_vector_length);
    void push_data(GpuMixingData<T>& data,
                   const T* input,
                   const T* output,
                   const Screen& screen,
                   const Mix& mix,
                   bool update_coefficients);
    void update_coefficients(const GpuMixingData<T>& data,
                             const GramBuilder& build_gram,
                             const RhsBuilder& build_rhs);
    void mix_data(const GpuMixingData<T>& data, T* result) const;

    int history_capacity() const;
    const std::vector<double>& coefficients() const;

  private:
    GpuMixing(const GpuMixing&) = delete;
    GpuMixing& operator=(const GpuMixing&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace Base_Mixing

#endif // GPU_MIXING_H_
