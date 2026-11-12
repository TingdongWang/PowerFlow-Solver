#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <condition_variable>
#include <mutex>
#include <stdexcept>

struct ManualResetEvent {
    std::mutex mutex;
    std::condition_variable condition;
    bool signaled;
    explicit ManualResetEvent(bool initial) : signaled(initial) {}
};

using HANDLE = ManualResetEvent*;
using CRITICAL_SECTION = std::mutex;
constexpr bool TRUE = true;
constexpr bool FALSE = false;
constexpr unsigned long INFINITE = ~0UL;

inline HANDLE CreateEvent(void*, bool manual_reset, bool initial, const char*) {
    if (!manual_reset) throw std::invalid_argument("Only manual-reset events are used");
    return new ManualResetEvent(initial);
}
inline void InitializeCriticalSection(CRITICAL_SECTION*) {}
inline void EnterCriticalSection(CRITICAL_SECTION* value) { value->lock(); }
inline void LeaveCriticalSection(CRITICAL_SECTION* value) { value->unlock(); }
inline void SetEvent(HANDLE value) {
    {
        std::lock_guard<std::mutex> lock(value->mutex);
        value->signaled = true;
    }
    value->condition.notify_all();
}
inline void ResetEvent(HANDLE value) {
    std::lock_guard<std::mutex> lock(value->mutex);
    value->signaled = false;
}
inline void WaitForSingleObject(HANDLE value, unsigned long timeout) {
    if (timeout != INFINITE) throw std::invalid_argument("Only infinite waits are used");
    std::unique_lock<std::mutex> lock(value->mutex);
    value->condition.wait(lock, [value] { return value->signaled; });
}
#endif
