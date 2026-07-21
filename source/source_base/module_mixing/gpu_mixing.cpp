#include "gpu_mixing.h"

#include "mixing_coefficients.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/matrix.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <complex>

namespace Base_Mixing
{

template <typename T>
GpuMixingData<T>::GpuMixingData()
    : data_(nullptr), capacity_(0), length_(0), current_slot_(-1), size_(0)
{
}

template <typename T>
GpuMixingData<T>::GpuMixingData(const int capacity, const std::size_t length)
    : data_(nullptr), capacity_(0), length_(0), current_slot_(-1), size_(0)
{
    this->resize(capacity, length);
}

template <typename T>
GpuMixingData<T>::~GpuMixingData()
{
    if (this->data_ != nullptr)
    {
        base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->data_);
    }
}

template <typename T>
void GpuMixingData<T>::resize(const int capacity, const std::size_t length)
{
    if (this->data_ != nullptr)
    {
        base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->data_);
        this->data_ = nullptr;
    }

    this->capacity_ = capacity;
    this->length_ = length;
    this->reset();
    if (capacity > 0 && length > 0)
    {
        base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(
            this->data_, capacity * length, "GpuMixingData");
        base_device::memory::set_memory_op<T, base_device::DEVICE_GPU>()(
            this->data_, 0, capacity * length);
    }
}

template <typename T>
void GpuMixingData<T>::reset()
{
    this->current_slot_ = -1;
    this->size_ = 0;
}

template <typename T>
void GpuMixingData<T>::push(const T* data)
{
    this->current_slot_ = (this->current_slot_ + 1) % this->capacity_;
    this->size_ = std::min(this->size_ + 1, this->capacity_);
    base_device::memory::synchronize_memory_op<T,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(
        this->data_ + this->current_slot_ * this->length_, data, this->length_);
}

template <typename T>
int GpuMixingData<T>::size() const
{
    return this->size_;
}

template <typename T>
std::size_t GpuMixingData<T>::length() const
{
    return this->length_;
}

template <typename T>
int GpuMixingData<T>::current_slot() const
{
    return this->current_slot_;
}

template <typename T>
int GpuMixingData<T>::index_move(const int offset) const
{
    return (offset + this->current_slot_ + this->capacity_) % this->capacity_;
}

template <typename T>
struct GpuMixing<T>::Impl
{
    Impl(const MixingAlgorithm algorithm_in, const int history_depth_in, const double beta_in)
        : algorithm(algorithm_in),
          history_depth(history_depth_in),
          history_capacity(algorithm_in == MixingAlgorithm::Broyden ? history_depth_in + 1 : history_depth_in),
          beta(beta_in),
          length(0),
          residual(nullptr),
          residual_history(nullptr),
          temporary(nullptr),
          device_coefficients(nullptr),
          gram_cache(history_depth_in, history_depth_in, true),
          coefficients(history_capacity, 0.0),
          bound_data(nullptr),
          residual_slot(algorithm_in == MixingAlgorithm::Broyden ? -1 : 0),
          residual_count(0)
    {
    }

    MixingAlgorithm algorithm;
    int history_depth;
    int history_capacity;
    double beta;
    std::size_t length;
    T* residual;
    T* residual_history;
    T* temporary;
    T* device_coefficients;
    ModuleBase::matrix gram_cache;
    std::vector<double> coefficients;
    const GpuMixingData<T>* bound_data;
    int residual_slot;
    int residual_count;

    int residual_index_move(const int offset) const
    {
        return (this->residual_slot + offset + this->history_depth) % this->history_depth;
    }

    void release()
    {
        if (this->residual != nullptr)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->residual);
            this->residual = nullptr;
        }
        if (this->residual_history != nullptr)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->residual_history);
            this->residual_history = nullptr;
        }
        if (this->temporary != nullptr)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->temporary);
            this->temporary = nullptr;
        }
        if (this->device_coefficients != nullptr)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(this->device_coefficients);
            this->device_coefficients = nullptr;
        }
    }

    void upload_coefficients()
    {
        std::vector<T> typed_coefficients(this->coefficients.size());
        for (std::size_t i = 0; i < this->coefficients.size(); ++i)
        {
            typed_coefficients[i] = static_cast<T>(this->coefficients[i]);
        }
        base_device::memory::synchronize_memory_op<T,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            this->device_coefficients, typed_coefficients.data(), typed_coefficients.size());
    }

    void use_latest_coefficients(const GpuMixingData<T>& data)
    {
        std::fill(this->coefficients.begin(), this->coefficients.end(), 0.0);
        this->coefficients[data.current_slot_] = 1.0;
        this->upload_coefficients();
    }

    void check_binding(const GpuMixingData<T>& data) const
    {
        if (this->bound_data != nullptr && this->bound_data != &data)
        {
            ModuleBase::WARNING_QUIT(
                "GpuMixing",
                "One GpuMixing object can only bind one GpuMixingData object to calculate coefficients");
        }
    }
};

template <typename T>
GpuMixing<T>::GpuMixing(const MixingAlgorithm algorithm, const int history_depth, const double beta)
    : impl_(new Impl(algorithm, history_depth, beta))
{
}

template <typename T>
GpuMixing<T>::~GpuMixing()
{
    this->impl_->release();
    delete this->impl_;
}

template <typename T>
void GpuMixing<T>::reset(const std::size_t coefficient_vector_length)
{
    Impl& impl = *this->impl_;
    if (impl.length != coefficient_vector_length || impl.temporary == nullptr)
    {
        impl.release();
        impl.length = coefficient_vector_length;
        if (impl.length > 0)
        {
            base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(
                impl.temporary, impl.length, "GpuMixingTemporary");
            base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(
                impl.residual_history, impl.history_depth * impl.length, "GpuMixingResidualHistory");
            base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(
                impl.device_coefficients, impl.history_capacity, "GpuMixingCoefficients");
            if (impl.algorithm == MixingAlgorithm::Broyden)
            {
                base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(
                    impl.residual, impl.length, "GpuMixingResidual");
            }
        }
    }
    impl.bound_data = nullptr;
    impl.residual_slot = impl.algorithm == MixingAlgorithm::Broyden ? -1 : 0;
    impl.residual_count = 0;
    std::fill(impl.coefficients.begin(), impl.coefficients.end(), 0.0);
    impl.gram_cache.zero_out();
}

template <typename T>
void GpuMixing<T>::push_data(GpuMixingData<T>& data,
                             const T* input,
                             const T* output,
                             const Screen& screen,
                             const Mix& mix,
                             const bool update_coefficients)
{
    Impl& impl = *this->impl_;
    const int length = static_cast<int>(data.length_);
    ModuleBase::vector_add_vector_op<T, base_device::DEVICE_GPU>()(
        length, impl.temporary, output, 1.0, input, -1.0);
    if (screen)
    {
        screen(impl.temporary);
    }
    if (mix)
    {
        mix(impl.temporary, input, impl.temporary);
    }
    else
    {
        ModuleBase::vector_add_vector_op<T, base_device::DEVICE_GPU>()(
            length, impl.temporary, input, 1.0, impl.temporary, impl.beta);
    }
    data.push(impl.temporary);

    if (!update_coefficients)
    {
        return;
    }

    impl.check_binding(data);
    ModuleBase::vector_add_vector_op<T, base_device::DEVICE_GPU>()(
        length, impl.temporary, output, 1.0, input, -1.0);
    if (screen)
    {
        screen(impl.temporary);
    }

    if (impl.algorithm == MixingAlgorithm::Broyden)
    {
        if (data.size_ == 1)
        {
            impl.bound_data = &data;
            base_device::memory::synchronize_memory_op<T,
                                                       base_device::DEVICE_GPU,
                                                       base_device::DEVICE_GPU>()(
                impl.residual, impl.temporary, length);
        }
        else
        {
            impl.residual_count = std::min(impl.residual_count + 1, impl.history_depth);
            impl.residual_slot = (impl.residual_slot + 1) % impl.history_depth;
            T* difference = impl.residual_history + impl.residual_slot * data.length_;
            ModuleBase::vector_add_vector_op<T, base_device::DEVICE_GPU>()(
                length, difference, impl.residual, 1.0, impl.temporary, -1.0);
            base_device::memory::synchronize_memory_op<T,
                                                       base_device::DEVICE_GPU,
                                                       base_device::DEVICE_GPU>()(
                impl.residual, impl.temporary, length);
        }
    }
    else
    {
        if (data.size_ == 1)
        {
            impl.bound_data = &data;
            impl.residual_count = 1;
        }
        else
        {
            impl.residual_count = std::min(impl.residual_count + 1, impl.history_depth);
        }
        impl.residual_slot = data.current_slot_;
        base_device::memory::synchronize_memory_op<T,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_GPU>()(
            impl.residual_history + impl.residual_slot * data.length_,
            impl.temporary,
            length);
    }
}

template <typename T>
void GpuMixing<T>::update_coefficients(const GpuMixingData<T>& data,
                                       const GramBuilder& build_gram,
                                       const RhsBuilder& build_rhs)
{
    Impl& impl = *this->impl_;
    impl.check_binding(data);

    if (impl.algorithm == MixingAlgorithm::Broyden)
    {
        if (impl.residual_count > 0)
        {
            const int count = impl.residual_count;
            ModuleBase::matrix gram(count, count);
            for (int i = 0; i < count; ++i)
            {
                for (int j = i; j < count; ++j)
                {
                    if (i != impl.residual_slot && j != impl.residual_slot)
                    {
                        gram(i, j) = impl.gram_cache(i, j);
                    }
                    if (j != i)
                    {
                        gram(j, i) = gram(i, j);
                    }
                }
            }
            build_gram(impl.residual_history, count, impl.residual_slot, gram);
            for (int i = 0; i < count; ++i)
            {
                impl.gram_cache(impl.residual_slot, i) = gram(impl.residual_slot, i);
                impl.gram_cache(i, impl.residual_slot) = gram(i, impl.residual_slot);
            }

            std::vector<double> rhs(count);
            build_rhs(impl.residual_history, impl.residual, count, rhs);
            solve_broyden_system(gram, rhs);

            std::fill(impl.coefficients.begin(), impl.coefficients.end(), 0.0);
            impl.coefficients[data.current_slot_] = 1.0 + rhs[impl.residual_index_move(0)];
            for (int i = 1; i < count; ++i)
            {
                impl.coefficients[data.index_move(-i)]
                    = rhs[impl.residual_index_move(-i)] - rhs[impl.residual_index_move(-i + 1)];
            }
            impl.coefficients[data.index_move(-count)] = -rhs[impl.residual_index_move(-count + 1)];
            impl.upload_coefficients();
        }
        else
        {
            impl.use_latest_coefficients(data);
        }

        base_device::memory::synchronize_memory_op<T,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_GPU>()(
            impl.residual_history + impl.residual_index_move(1) * data.length_,
            impl.residual,
            static_cast<int>(data.length_));
        return;
    }

    const int count = data.size_;
    if (count > 1)
    {
        ModuleBase::matrix gram(count, count);
        for (int i = 0; i < count; ++i)
        {
            for (int j = i; j < count; ++j)
            {
                if (i != impl.residual_slot && j != impl.residual_slot)
                {
                    gram(i, j) = impl.gram_cache(i, j);
                }
                if (j != i)
                {
                    gram(j, i) = gram(i, j);
                }
            }
        }
        build_gram(impl.residual_history, count, impl.residual_slot, gram);
        for (int i = 0; i < count; ++i)
        {
            impl.gram_cache(impl.residual_slot, i) = gram(impl.residual_slot, i);
            impl.gram_cache(i, impl.residual_slot) = gram(i, impl.residual_slot);
        }
        std::vector<double> active_coefficients(count);
        solve_pulay_system(gram, active_coefficients);
        std::fill(impl.coefficients.begin(), impl.coefficients.end(), 0.0);
        for (int i = 0; i < count; ++i)
        {
            impl.coefficients[i] = active_coefficients[i];
        }
        impl.upload_coefficients();
    }
    else
    {
        impl.use_latest_coefficients(data);
    }
}

template <typename T>
void GpuMixing<T>::mix_data(const GpuMixingData<T>& data, T* result) const
{
    if (data.length_ == 0 || data.size_ == 0)
    {
        return;
    }
    if (data.size_ == 1)
    {
        base_device::memory::synchronize_memory_op<T,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_GPU>()(
            result,
            data.data_ + data.current_slot_ * data.length_,
            data.length_);
        return;
    }
    const T one = static_cast<T>(1.0);
    const T zero = static_cast<T>(0.0);
    ModuleBase::gemv_op<T, base_device::DEVICE_GPU>()(
        'N',
        static_cast<int>(data.length_),
        data.capacity_,
        &one,
        data.data_,
        static_cast<int>(data.length_),
        this->impl_->device_coefficients,
        1,
        &zero,
        result,
        1);
}

template <typename T>
int GpuMixing<T>::history_capacity() const
{
    return this->impl_->history_capacity;
}

template class GpuMixingData<double>;
template class GpuMixingData<std::complex<double>>;
template class GpuMixing<double>;
template class GpuMixing<std::complex<double>>;

} // namespace Base_Mixing
