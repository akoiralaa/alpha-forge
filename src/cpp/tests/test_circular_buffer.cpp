#include "onebrain/circular_buffer.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace onebrain;

void test_basic_push_and_access() {
    CircularBuffer<double> buf(5);
    assert(buf.empty());
    assert(buf.capacity() == 5);

    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);

    assert(buf.size() == 3);
    assert(!buf.full());
    assert(buf[0] == 3.0);  // most recent
    assert(buf[1] == 2.0);
    assert(buf[2] == 1.0);  // oldest
    assert(buf.newest() == 3.0);
    assert(buf.oldest() == 1.0);
}

void test_wrap_around() {
    CircularBuffer<int> buf(3);
    buf.push(1); buf.push(2); buf.push(3);
    assert(buf.full());
    assert(buf[0] == 3);

    buf.push(4);  // overwrites 1
    assert(buf.size() == 3);
    assert(buf[0] == 4);
    assert(buf[1] == 3);
    assert(buf[2] == 2);
    assert(buf.oldest() == 2);

    buf.push(5);  // overwrites 2
    assert(buf[0] == 5);
    assert(buf[2] == 3);
}

void test_out_of_range() {
    CircularBuffer<double> buf(3);
    buf.push(1.0);

    bool threw = false;
    try { buf[1]; } catch (const std::out_of_range&) { threw = true; }
    assert(threw);
}

void test_clear() {
    CircularBuffer<double> buf(5);
    buf.push(1.0); buf.push(2.0);
    buf.clear();
    assert(buf.empty());
    assert(buf.size() == 0);
}

void test_zero_capacity() {
    bool threw = false;
    try { CircularBuffer<double> buf(0); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

void test_single_element() {
    CircularBuffer<double> buf(1);
    buf.push(10.0);
    assert(buf[0] == 10.0);
    assert(buf.newest() == 10.0);
    assert(buf.oldest() == 10.0);
    buf.push(20.0);
    assert(buf[0] == 20.0);
    assert(buf.size() == 1);
}

int main() {
    test_basic_push_and_access();
    test_wrap_around();
    test_out_of_range();
    test_clear();
    test_zero_capacity();
    test_single_element();

    std::cout << "All circular_buffer tests passed." << std::endl;
    return 0;
}
