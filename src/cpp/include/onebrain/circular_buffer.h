#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace onebrain {

/// Fixed-capacity circular buffer. O(1) push, O(1) random access.
/// No heap allocation after construction.
template <typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : buf_(capacity), cap_(capacity), head_(0), size_(0) {
        if (capacity == 0)
            throw std::invalid_argument("CircularBuffer capacity must be > 0");
    }

    void push(const T& val) {
        buf_[head_] = val;
        head_ = (head_ + 1) % cap_;
        if (size_ < cap_) ++size_;
    }

    /// Access element by age: 0 = most recent, 1 = second most recent, ...
    const T& operator[](size_t age) const {
        if (age >= size_)
            throw std::out_of_range("CircularBuffer index out of range");
        size_t idx = (head_ + cap_ - 1 - age) % cap_;
        return buf_[idx];
    }

    /// Access oldest element
    const T& oldest() const {
        if (size_ == 0)
            throw std::out_of_range("CircularBuffer is empty");
        if (size_ < cap_)
            return buf_[0];
        return buf_[head_];
    }

    /// Access most recent element
    const T& newest() const {
        if (size_ == 0)
            throw std::out_of_range("CircularBuffer is empty");
        return buf_[(head_ + cap_ - 1) % cap_];
    }

    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }
    bool   full() const { return size_ == cap_; }
    bool   empty() const { return size_ == 0; }

    void clear() {
        head_ = 0;
        size_ = 0;
    }

private:
    std::vector<T> buf_;
    size_t cap_;
    size_t head_;
    size_t size_;
};

}  // namespace onebrain
