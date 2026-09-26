/* Copyright 2022 Philip Miller
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ABLASTR_SIGNAL_HANDLING_H_
#define ABLASTR_SIGNAL_HANDLING_H_

#include <AMReX_Config.H>

#if defined(AMREX_USE_MPI)
#   include <mpi.h>
#endif

#include <atomic>
#include <string>

namespace ablastr::utils {

/**
 * \brief
 * Signal handling
 *
 * Rank 0 will accept signals and asynchronously broadcast the
 * configured response; other processes will ignore them and
 * follow the lead of rank 0 to avoid potential for deadlocks or
 * timestep-skewed response.
 *
 * Variables and functions are static rather than per-instance
 * because signal handlers are configured at the process level.
 */
class SignalHandling
{
public:
    //! The range of signal values to accept
    static constexpr int NUM_SIGNALS = 32;

    //! Labels for indexed positions in signal_actions_requests
    enum signal_action_requested_labels {
        //! Cleanly stop execution, as if the simulation reached its configured end
        SIGNAL_REQUESTS_BREAK = 0,
        //! Produce a checkpoint
        SIGNAL_REQUESTS_CHECKPOINT = 1,
        SIGNAL_REQUESTS_SIZE = 2 // This should always be 1 greater than the last valid value
    };

    //! Whether configuration requests the code take a particular action on a particular signal
    static bool signal_conf_requests[SIGNAL_REQUESTS_SIZE][NUM_SIGNALS];

    //! Take a string and convert it to a corresponding signal number if possible
    static int parseSignalNameToNumber (const std::string &str);

    //! Set up signal handlers based on input configuration provided in `signal_conf_requests_*`
    static void InitSignalHandling ();

    //! Check and clear signal flags and asynchronously broadcast them from process 0
    static void CheckSignals ();
    //! Complete the asynchronous broadcast of signal flags
    static void WaitSignals ();

    //! Initialize timer-based self-signaling from environment variables.
    //!
    //! Reads the following env vars (rank 0 only):
    //!   WARPX_BREAK_AFTER_SECONDS=<int>      — once after N wall seconds, self-signal break
    //!   WARPX_CHECKPOINT_EVERY_SECONDS=<int> — every N wall seconds, self-signal checkpoint
    //!
    //! The signal number for each action is looked up from signal_conf_requests[],
    //! so the user's existing warpx.break_signals / warpx.checkpoint_signals input
    //! mappings are honored. If a timer is set but no signal is configured for
    //! the corresponding action, a warning is printed and that timer is disabled.
    //!
    //! Must be called AFTER signal_conf_requests has been populated (i.e. after
    //! WarpX::ReadParameters has parsed break_signals / checkpoint_signals).
    static void InitTimers ();

    //! Check elapsed wall-clock time against configured timers, and poll for
    //! sentinel files; self-signal via std::raise(sig) when either fires.
    //! Cheap no-op when both timers are disabled and sentinel polling is off.
    //! Rank 0 only; other ranks return immediately. The existing per-step
    //! CheckSignals() then broadcasts the resulting action flag to all ranks,
    //! so no separate MPI coordination is needed.
    //!
    //! Sentinel files (in the working directory, rank 0 only):
    //!   SIGNAL_CHECKPOINT — request a checkpoint, then continue
    //!   SIGNAL_TERMINATE  — request a clean break (same action as a break signal)
    //!
    //! The file is removed before the signal is raised, so each request fires
    //! once. If both are present in the same poll, SIGNAL_CHECKPOINT is handled
    //! first so a checkpoint is written before the run stops.
    //!
    //! If a sentinel exists but cannot be removed, the request cannot be
    //! consumed and would otherwise re-fire on every poll. SIGNAL_TERMINATE is
    //! still honoured once (a repeated break is idempotent); SIGNAL_CHECKPOINT
    //! is not (a checkpoint storm costs more than a missed checkpoint). Either
    //! way a single warning is emitted and that file is then ignored for the
    //! rest of the run. Polling is
    //! throttled by WARPX_SIGNAL_FILE_POLL_SECONDS (default 5; 0 = every step;
    //! negative = disabled) to keep metadata traffic off the parallel
    //! filesystem on long runs.
    static void CheckTimers ();

    //! Check whether a given action has been requested, and reset the associated flag
    static bool TestAndResetActionRequestFlag (int action_to_test);

    // Don't allow clients to incorrectly try to construct and use an instance of this type
    SignalHandling () = delete;

private:
    //! Is any signal handling action configured in signal_conf_requests ?
    static bool m_any_signal_action_active;

    //! On process 0, whether a given signal has been received since the last check
    static std::atomic<bool> signal_received_flags[NUM_SIGNALS];

    //
    // Timer-based self-signaling state (rank 0 only). Configured by InitTimers
    // from environment variables; consulted by CheckTimers each step.
    //
    //! Wall time (amrex::second()) at InitTimers
    static double m_timer_wall_start;
    //! Configured WARPX_BREAK_AFTER_SECONDS, or -1 if disabled
    static int m_timer_break_after_seconds;
    //! Configured WARPX_CHECKPOINT_EVERY_SECONDS, or -1 if disabled
    static int m_timer_checkpoint_every_seconds;
    //! Cached signal number to fire for break (-1 if none configured)
    static int m_timer_break_signal;
    //! Cached signal number to fire for checkpoint (-1 if none configured)
    static int m_timer_checkpoint_signal;
    //! Whether the break timer has already fired (one-shot)
    static bool m_timer_break_fired;
    //! Next wall-time threshold at which a checkpoint should fire (relative to start)
    static double m_timer_next_checkpoint_at;

    //
    // Sentinel-file polling state (rank 0 only). Independent of the timers:
    // enabled whenever a signal is configured for the corresponding action,
    // even if no WARPX_*_SECONDS timer is set.
    //
    //! Configured WARPX_SIGNAL_FILE_POLL_SECONDS (default 5), or -1 if disabled
    static int m_sentinel_poll_seconds;
    //! Next elapsed-time threshold, in whole seconds, at which to poll for
    //! sentinel files. Integer because the interval is integer seconds and
    //! advances by fixed multiples — unlike m_timer_next_checkpoint_at, which
    //! re-anchors on measured elapsed time and is therefore fractional.
    static int m_sentinel_next_poll_at;

    //! Poll for SIGNAL_CHECKPOINT / SIGNAL_TERMINATE in the working directory.
    //! Removes the file and raises the mapped signal. Rank 0 only.
    static void CheckSentinelFiles ();

#if defined(AMREX_USE_MPI)
    //! MPI requests for the asynchronous broadcasts of the signal-requested actions
    static MPI_Request signal_mpi_ibcast_request;
#endif

    //! Signal handler to set flags on process 0 (other processes ignore configured signals)
    static void SignalSetFlag (int signal_number);

    //! Boolean flags transmitted between CheckSignals() and
    //! HandleSignals() to indicate actions requested by signals
    static bool signal_actions_requested[SIGNAL_REQUESTS_SIZE];
};

} // namespace ablastr::utils

#endif // ABLASTR_SIGNAL_HANDLING_H_
