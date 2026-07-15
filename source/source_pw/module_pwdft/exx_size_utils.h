#ifndef EXX_SIZE_UTILS_H
#define EXX_SIZE_UTILS_H

#include <cstddef>
#include <initializer_list>
#include <limits>

namespace hamilt
{
inline bool checked_exx_size_product(std::size_t lhs, std::size_t rhs, std::size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    {
        return false;
    }
    result = lhs * rhs;
    return true;
}

inline bool checked_exx_size_product(std::initializer_list<std::size_t> factors, std::size_t& result)
{
    result = 1;
    for (const std::size_t factor: factors)
    {
        if (!checked_exx_size_product(result, factor, result))
        {
            return false;
        }
    }
    return true;
}
} // namespace hamilt

#endif // EXX_SIZE_UTILS_H
