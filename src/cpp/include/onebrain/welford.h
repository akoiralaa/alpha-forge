#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace onebrain {

/// Welford online mean/variance with optional fixed window.
/// When window > 0, uses a circular buffer to subtract old values.
/// When window == 0, accumulates over all samples (running).
class WelfordAccumulator {
public:
    explicit WelfordAccumulator(size_t window = 0)
        : window_(window), count_(0), mean_(0.0), m2_(0.0) {
        if (window_ > 0) {
            ring_.resize(window_, 0.0);
        }
    }

    void update(double x) {
        if (window_ > 0) {
            if (count_ >= window_) {
                // Remove oldest value
                double old_val = ring_[ring_idx_];
                double old_mean = mean_;
                mean_ += (x - old_val) / static_cast<double>(window_);
                m2_ += (x - mean_) * (x - old_mean) - (old_val - mean_) * (old_val - old_mean);
                if (m2_ < 0.0) m2_ = 0.0;  // numerical guard
                ring_[ring_idx_] = x;
                ring_idx_ = (ring_idx_ + 1) % window_;
            } else {
                // Still filling
                ring_[ring_idx_] = x;
                ring_idx_ = (ring_idx_ + 1) % window_;
                ++count_;
                double delta = x - mean_;
                mean_ += delta / static_cast<double>(count_);
                double delta2 = x - mean_;
                m2_ += delta * delta2;
            }
        } else {
            // Unbounded accumulation
            ++count_;
            double delta = x - mean_;
            mean_ += delta / static_cast<double>(count_);
            double delta2 = x - mean_;
            m2_ += delta * delta2;
        }
    }

    double mean() const {
        size_t n = effective_count();
        return (n > 0) ? mean_ : std::numeric_limits<double>::quiet_NaN();
    }

    double variance() const {
        size_t n = effective_count();
        return (n >= 2) ? m2_ / static_cast<double>(n - 1) : std::numeric_limits<double>::quiet_NaN();
    }

    double stddev() const {
        double v = variance();
        return std::isnan(v) ? v : std::sqrt(v);
    }

    double zscore(double x) const {
        double s = stddev();
        if (std::isnan(s) || s < 1e-15) return std::numeric_limits<double>::quiet_NaN();
        return (x - mean_) / s;
    }

    size_t effective_count() const {
        if (window_ > 0)
            return std::min(count_, window_);
        return count_;
    }

    size_t window() const { return window_; }

    void reset() {
        count_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
        ring_idx_ = 0;
        if (window_ > 0) {
            std::fill(ring_.begin(), ring_.end(), 0.0);
        }
    }

private:
    size_t window_;
    size_t count_  = 0;
    double mean_   = 0.0;
    double m2_     = 0.0;

    // Ring buffer for windowed mode
    std::vector<double> ring_;
    size_t ring_idx_ = 0;
};

}  // namespace onebrain
