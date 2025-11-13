import ctypes


class Struct_mallinfo2(ctypes.Structure):
    """A ctypes Structure that matches the libc mallinfo2 struct.

    Docs:
    - https://man7.org/linux/man-pages/man3/mallinfo.3.html
    - https://www.gnu.org/software/libc/manual/html_node/Statistics-of-Malloc.html

    struct mallinfo2 {
        size_t arena;     /* Non-mmapped space allocated (bytes) */
        size_t ordblks;   /* Number of free chunks */
        size_t smblks;    /* Number of free fastbin blocks */
        size_t hblks;     /* Number of mmapped regions */
        size_t hblkhd;    /* Space allocated in mmapped regions (bytes) */
        size_t usmblks;   /* See below */
        size_t fsmblks;   /* Space in freed fastbin blocks (bytes) */
        size_t uordblks;  /* Total allocated space (bytes) */
        size_t fordblks;  /* Total free space (bytes) */
        size_t keepcost;  /* Top-most, releasable space (bytes) */
    };
    """

    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]

    def __str__(self) -> str:
        # Precompute divisor
        GB = 2**30

        # Avoid repeated attribute and division lookups
        arena_gb = self.arena / GB
        ordblks = self.ordblks
        smblks = self.smblks
        hblks = self.hblks
        hblkhd_gb = self.hblkhd / GB
        usmblks = self.usmblks
        fsmblks_gb = self.fsmblks / GB
        uordblks_gb = self.uordblks / GB
        fordblks_gb = self.fordblks / GB
        keepcost_gb = self.keepcost / GB

        # List of strings, join at the end for efficiency
        lines = [
            f"{'arena': <10}= {arena_gb:15.5f}   # Non-mmapped space allocated (GB) (uordblks + fordblks)\n",
            f"{'ordblks': <10}= {ordblks: >15}   # Number of free chunks\n",
            f"{'smblks': <10}= {smblks: >15}   # Number of free fastbin blocks \n",
            f"{'hblks': <10}= {hblks: >15}   # Number of mmapped regions \n",
            f"{'hblkhd': <10}= {hblkhd_gb:15.5f}   # Space allocated in mmapped regions (GB)\n",
            f"{'usmblks': <10}= {usmblks: >15}   # Unused\n",
            f"{'fsmblks': <10}= {fsmblks_gb:15.5f}   # Space in freed fastbin blocks (GB)\n",
            (f"{'uordblks': <10}= {uordblks_gb:15.5f}   # Space used by in-use allocations (non-mmapped) (GB)\n"),
            f"{'fordblks': <10}= {fordblks_gb:15.5f}   # Space in free blocks (non-mmapped) (GB)\n",
            f"{'keepcost': <10}= {keepcost_gb:15.5f}   # Top-most, releasable space (GB)\n",
        ]
        return "".join(lines)


class LibcUtil:
    """A utility class for interacting with the C Standard Library (`libc`) via ctypes.

    Note that this class will raise on __init__() if 'libc.so.6' can't be found. Take care to handle environments where
    this shared library is not available.

    TODO: Improve cross-OS compatibility of this class.
    """

    def __init__(self) -> None:
        self._libc = ctypes.cdll.LoadLibrary("libc.so.6")

    def mallinfo2(self) -> Struct_mallinfo2:
        """Calls `libc` `mallinfo2`.

        Docs: https://man7.org/linux/man-pages/man3/mallinfo.3.html
        """
        mallinfo2 = self._libc.mallinfo2
        mallinfo2.restype = Struct_mallinfo2
        result: Struct_mallinfo2 = mallinfo2()
        return result
