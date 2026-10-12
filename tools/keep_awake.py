# -*- coding: utf-8 -*-
"""Keep the machine awake while SEAL experiments run, independent of power
scheme. Asserts ES_CONTINUOUS | ES_SYSTEM_REQUIRED via SetThreadExecutionState,
which overrides idle/unattended sleep timeouts as long as THIS process lives.
Exits (and releases the keep-awake) only when killed. Re-asserts every 60s as a
belt-and-suspenders against any state reset.

Run detached:
  start /b python tools\\keep_awake.py
Stop: kill the python process (or it dies with the session).
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# ES_AWAYMODE_REQUIRED keeps the system in a working state even on "away" sleep
ES_AWAYMODE_REQUIRED = 0x00000040

FLAGS = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED


def main():
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    # initial assertion
    set_state(FLAGS)
    print("keep_awake: ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_AWAYMODE_REQUIRED asserted", flush=True)
    while True:
        # re-assert periodically (cheap; guards against any reset)
        set_state(FLAGS)
        time.sleep(60)


if __name__ == "__main__":
    main()
