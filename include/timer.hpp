#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    double elapsed_ms() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }

    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }
};

#endif // TIMER_HPP
