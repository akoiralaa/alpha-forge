#include "onebrain/welford.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace onebrain;

static bool approx(double a, double b, double tol = 1e-6) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::abs(a - b) < tol;
}

void test_running_mean_variance() {
    WelfordAccumulator w(0);  // unbounded
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};

    for (double x : data) w.update(x);

    assert(w.effective_count() == 8);
    assert(approx(w.mean(), 5.0));
    // sample variance of this data = 4.571428...
    assert(approx(w.variance(), 4.571428, 1e-3));
    assert(approx(w.stddev(), std::sqrt(4.571428), 1e-3));
}

void test_zscore() {
    WelfordAccumulator w(0);
    for (int i = 1; i <= 100; i++) w.update(static_cast<double>(i));

    // mean = 50.5, check z-score of mean is ~0
    assert(approx(w.zscore(50.5), 0.0, 0.01));
    // z-score of 100 should be positive
    assert(w.zscore(100.0) > 0.0);
}

void test_windowed() {
    WelfordAccumulator w(5);  // window of 5

    // Fill: 1,2,3,4,5
    for (int i = 1; i <= 5; i++) w.update(static_cast<double>(i));
    assert(w.effective_count() == 5);
    assert(approx(w.mean(), 3.0));

    // Push 6 — window is now [2,3,4,5,6], mean = 4.0
    w.update(6.0);
    assert(w.effective_count() == 5);
    assert(approx(w.mean(), 4.0, 0.1));

    // Push 7 — window is [3,4,5,6,7], mean = 5.0
    w.update(7.0);
    assert(approx(w.mean(), 5.0, 0.1));
}

void test_empty() {
    WelfordAccumulator w(0);
    assert(std::isnan(w.mean()));
    assert(std::isnan(w.variance()));
    assert(std::isnan(w.stddev()));
    assert(std::isnan(w.zscore(1.0)));
}

void test_single_value() {
    WelfordAccumulator w(0);
    w.update(5.0);
    assert(approx(w.mean(), 5.0));
    assert(std::isnan(w.variance()));  // need n>=2
}

void test_reset() {
    WelfordAccumulator w(10);
    for (int i = 0; i < 20; i++) w.update(static_cast<double>(i));
    w.reset();
    assert(w.effective_count() == 0);
    assert(std::isnan(w.mean()));
}

void test_constant_values() {
    WelfordAccumulator w(0);
    for (int i = 0; i < 100; i++) w.update(42.0);
    assert(approx(w.mean(), 42.0));
    assert(approx(w.variance(), 0.0, 1e-10));
}

int main() {
    test_running_mean_variance();
    test_zscore();
    test_windowed();
    test_empty();
    test_single_value();
    test_reset();
    test_constant_values();

    std::cout << "All welford tests passed." << std::endl;
    return 0;
}
