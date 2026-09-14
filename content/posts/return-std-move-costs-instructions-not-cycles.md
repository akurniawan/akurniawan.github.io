---
title: "I Set Out to Learn NRVO and Ended Up Measuring My Command Line"
date: 2026-09-14T00:00:00+07:00
draft: false
tags: ["cpp", "benchmarking", "compilers", "performance", "arm"]
categories: ["systems", "performance"]
description: "What return std::move(b) costs against return b on a wide ARM core, why returning a pointer instead is 4× slower and none of that is the pointer, and how a benchmark under 10 ns turned out to be measuring where its stack landed. GCC 13.3, -O2, GB10 Cortex-X925."
summary: "I wanted to understand named return value optimisation, got curious what the pessimised version costs, and my first numbers were reproducible to 0.01% and wrong. Instructions and cycles instead of wall-clock, and the pointer question turned out to be an allocator question."
toc: true
---

I sat down to learn what named return value optimisation (NRVO) actually is. I knew
`return std::move(b);` was supposed to be wrong, because GCC warns about it, but I
didn't know what it actually cost compared with `return b;`. So I wrote the two
smallest functions that differ by one `std::move`, looked at what GCC did with them,
and then wanted a number.

A second question came from my own habit. I'd always thought returning a pointer was
a good idea here, since it "avoids the copy", and I was curious whether it was really
as fast as NRVO. And if it is, does it matter whether the pointer is a `unique_ptr`
or a raw `new`?

The short answers: the move costs 2.8× the instructions and *fewer* cycles at 36
bytes, and only starts to cost time at 16 KiB. The pointer is 4× slower, and all of
it is `malloc` and `free`; `unique_ptr` versus raw `new` is seven instructions. But
the thing I actually learned is none of those. My first set of timings were
reproducible to a hundredth of a percent and completely meaningless, because at this
scale wall-clock time was measuring where the process's stack happened to land,
which is a polite way of saying it was measuring the length of my command line.

Everything here is on the machine and method from [the benchmarking
post](/posts/how-i-benchmark/): NVIDIA GB10, pinned to CPU 15 (a Cortex-X925
performance core), GCC 13.3 at `-O2 -march=native`, Google Benchmark, and hardware
counters read with `perf`.

## Summary

| finding | section |
|---|---|
| `return std::move(b)` forbids elision: 33 instructions instead of 7, plus a stack frame and a stack-protector canary the elided version never needed | §1 |
| Below ~10 ns, padding the `--benchmark_filter` string moved the result 2.3×, reproducibly; the fix is counting instructions and cycles, and padding the `perf` filters too | §2 |
| 2.8× the instructions cost 0.9× the cycles at 36 B; +12% cycles at 16 KiB | §3 |
| Returning a pointer is 4× slower, and placement-new proves 100% of it is the allocator | §4 |
| A relaxed atomic in my counting `operator new` was 56% of the profile and inflated that 4× to 7.3× | §5 |
| Inlining saved 3.5 cycles per call, and the calling convention doesn't explain it; the removed opacity does | §6 |

## 1. The code, and what the move forces

`Buf` is 36 bytes, an `int` and eight more `int`s stored inline, and the two producers
differ by exactly one `std::move`:

```cpp
#include <cstring>
#include <utility>
struct Buf {
    int n; int data[8];
    Buf(int v) : n(v) { for (int& d : data) d = v; }
    Buf(Buf&& o) noexcept : n(o.n) { std::memcpy(data, o.data, sizeof data); o.n = 0; }
    ~Buf() { n = -1; }          // non-trivial: forces the hidden-pointer return ABI
};
Buf make_plain(int v) { Buf b(v); return b; }
Buf make_move (int v) { Buf b(v); return std::move(b); }
```

Two things about the type are deliberate. The payload is inline, so there's no
pointer for a move to steal; "moving" a `Buf` can only ever be a copy of 32 bytes.
And the destructor and move constructor are user-written, which makes the type
non-trivial, so on ARM64 the caller sets aside space for the result and passes its
address to the callee. With NRVO, that address *is* the caller's variable: the
callee's local `b` and the caller's `x` are one object, constructed once.

`return std::move(b);` breaks that. Elision is permitted when a return statement
names a local; `std::move(b)` is a `Buf&&`, not a name, so elision is forbidden, the
move constructor has to run, and it needs a real `b` to copy from. On the GB10, GCC
13.3 at `-O2` turns `make_plain` into **7 instructions** that write the object
straight into the caller's slot, with no stack frame at all. `make_move` is **33
instructions**: the constructor writes `b` to the stack, the move constructor reads
it back and writes it again to the caller's slot, and because the function now has a
local array, Ubuntu's default `-fstack-protector-strong` adds a canary load, store,
compare and branch. NRVO deleted the stack frame, so it deleted the hardening too.

GCC says so if you ask:

```
warning: moving a local object in a return statement prevents copy elision
         [-Wpessimizing-move]
note: remove 'std::move' call
```

Neither `-Wpessimizing-move` nor its sibling `-Wredundant-move` is part of
`-Wall -Wextra`. Both are in my `CMakeLists.txt` now.

## 2. The trap: at 36 bytes, wall-clock measures the stack address

Seven versus thirty-three is a fact about the code. I wanted a fact about time, so I
wrapped both producers in Google Benchmark, added the pointer variants, and ran it.
The first run looked wrong. `MovedReturn` came out **faster** than `Nrvo`, which 33
instructions against 7 says is impossible:

```
BM_Nrvo<kSmall>         1.68 ns
BM_MovedReturn<kSmall>  1.56 ns
```

I checked the obvious thing first. `objdump` confirmed both benchmark loops really
call the non-inlined producers, and that the producers are the 7- and 33-instruction
bodies. The code was right. The measurement was wrong.

The cause took a while to believe. Same binary, same core, but the answer depended
on **how many characters were on the command line**. Padding the
`--benchmark_filter` string shifts where the process's initial stack starts, which
shifts every stack address in the program, which changes how the local `b` and the
return slot interact in the core's memory pipeline. Two mechanisms are candidates:
4K aliasing, where the core mistakes two addresses that agree in their low 12 bits
for the same location and stalls, and store-to-load forwarding, where a load that
follows a store to the same address can be served straight from the store buffer,
or can't, depending on alignment. I haven't confirmed which with counters.

| padding chars | `Nrvo` (ns) | `MovedReturn` (ns) | |
|---:|---:|---:|---|
| 0 | 0.771 | 2.49 | theory holds |
| 1 | 1.08 | 4.04 | |
| 2 | 1.26 | 2.54 | |
| 3 | 1.80 | 1.56 | theory "violated" |
| 7 | 1.80 | 1.56 | |

Every row is stable to under 0.01% variation across repetitions. **Reproducible and
meaningless.** A 2.3× swing in `Nrvo` from `argv`. I had followed all my own rules
from the benchmarking post, the spread was tiny, and the number still described
nothing about the code.

The fix is to count something that doesn't move when the stack does: instructions
retired and cycles retired, from the hardware performance counters. But the fix
needs the same discipline applied to itself. Running one benchmark per `perf`
invocation means every invocation has a *different* `--benchmark_filter` string,
which is exactly the thing that was moving the answer. So each filter is padded with
a non-matching alternation to an identical 64 bytes:

```sh
perf stat -e armv8_pmuv3_1/instructions/,armv8_pmuv3_1/cycles/ \
  taskset -c 15 ./build/bench_nrvo --benchmark_min_time=20000000x \
  --benchmark_filter="^BM_Nrvo<kSmall>\$|ZZZZ...ZZZZ"
```

Two details worth knowing. `--benchmark_min_time=20000000x` pins the iteration
count exactly, so dividing the counter totals per iteration is honest. And
`armv8_pmuv3_1` is the performance monitoring unit for the big-core cluster on this
chip; PMU 0 simply doesn't count anything on CPU 15, which I learned by staring at a
column of zeros.

As it turned out, instruction counts were immune anyway: 12.5 or 12.6 per iteration
in every run, padded or not. **Cycles are not immune**, and that's the whole point.
The same instruction stream takes a different number of cycles depending on where
the stack landed.

## 3. The headline numbers

Per call, with harness overhead included. The overhead is the same in every row, so
read the deltas, not the absolutes. Filters padded to equal length, reproducible
across passes. IPC is instructions per cycle, how much of the core's width the loop
is actually using.

| 36 B | insn/iter | cyc/iter | IPC |
|---|---:|---:|---:|
| `Inlined` (control) | 7.6 | 3.9 | 1.96 |
| `PlacementNew` (pointer, allocator removed) | 11.7 | 3.9 | 3.02 |
| `OutParam` (`f(Buf&)`) | 11.5 | 6.9 | 1.67 |
| `Nrvo` (`return b;`) | 12.6 | 7.4 | 1.71 |
| `MovedReturn` (`return std::move(b);`) | **35.5** | **6.4** | 5.54 |
| `RawPtr` (`new` / `delete`) | **196.6** | **29.5** | 6.66 |
| `UniquePtr` (`make_unique`) | **203.6** | **31.4** | 6.48 |

| 1 KiB | insn | cyc | | 16 KiB | insn | cyc |
|---|---:|---:|---|---|---:|---:|
| `Inlined` | 776 | 265 | | `Inlined` | 12334 | 4135 |
| `Nrvo` | 783 | 275 | | `Nrvo` | 12351 | 4143 |
| `PlacementNew` | 781 | 276 | | `PlacementNew` | 12337 | 4140 |
| `RawPtr` | 966 | 274 | | `RawPtr` | 12711 | 4186 |
| `UniquePtr` | 972 | 274 | | `UniquePtr` | 12717 | 4186 |
| `MovedReturn` | 950 | 270 | | `MovedReturn` | 14443 | **4652** |

At 36 bytes the move is **2.8× the instructions** (35.5 versus 12.6) and **fewer
cycles** (6.4 versus 7.4), at an IPC of 5.54. The out-of-order core swallows all 23
extra instructions whole. They're independent from one iteration to the next,
there's no long dependency chain, and the loop is bottlenecked somewhere else
entirely. **Retired instruction count is not cost.** I knew that as a slogan; this
is the first time I've watched a 2.8× instruction increase come out as a 0.9× cycle
decrease in my own data.

The cost only shows up once the payload is big enough for store bandwidth to bind.
At 16 KiB, `MovedReturn` is **+12% cycles** (4652 versus 4143), which is the
write, read, write round trip finally becoming visible.

So the pessimisation is real but small, and it's *not* small because the compiler
rescued it. It's small because the machine is wide. `return b;` is never worse:
elision is permitted, and if the compiler declines, C++11's implicit move on a
returned local gives you what `std::move` would have. Delete the four characters.

## 4. Returning a pointer: the cost is the allocator, and nothing else

Three variants isolate this. All three return a `Buf*` from a non-inlined function
and differ only in where the storage came from.

| 36 B | insn | cyc | vs `Nrvo` |
|---|---:|---:|---:|
| `Nrvo` (by value) | 12.6 | 7.4 | — |
| `PlacementNew` (storage allocated once, outside the loop) | 11.7 | 3.9 | **0.5×** |
| `RawPtr` (`new` + `delete`) | 196.6 | 29.5 | **4.0×** |
| `UniquePtr` (`make_unique`) | 203.6 | 31.4 | **4.2×** |

**Pointer semantics cost nothing.** `PlacementNew` returns the same pointer through
the same non-inlined call with the same indirection, and lands at 11.7 instructions
and 3.9 cycles, cheaper than NRVO and tied with the fully inlined control. So
**100% of the ~185 extra instructions and ~22 extra cycles is `malloc` plus
`free`.** So my habit had it backwards: the copy I was avoiding was never there,
and the allocation I added in its place is the whole cost. If a pointer return is
slow, the allocator is the thing to change (arena, pool, freelist), not the pointer
type.

**Raw `new` beats `unique_ptr` by about 6%, and the gap is seven instructions.**
196.6 versus 203.6, reproducible to ±0.5 across passes. The extra seven are
`make_unique`'s exception-safety scaffolding: 1.9 cycles per call, for RAII. At 1
KiB and 16 KiB the two are identical to within 0.05%.

Where the 185 instructions go, from `perf record` on `RawPtr` at 36 bytes:

```
60.16%  libc.so.6    _int_free
15.10%  libc.so.6    malloc
11.73%  libc.so.6    cfree@GLIBC_2.17
 5.50%  bench_nrvo   operator new(unsigned long)
 3.97%  bench_nrvo   make_raw<8>(int)          <- the actual constructor
 1.55%  bench_nrvo   BM_RawPtr<8>
```

**About 96% of the cycles are memory management; about 4% is constructing the
object.** Note that `free` is 4× `malloc`. On glibc's fast path (the tcache, a small
per-thread cache of recently freed blocks) allocation is close to popping the head of
a linked list. Freeing has to re-derive the chunk size from its header, validate
alignment, scan the tcache bin for a double-free, and decide about consolidation. And
this *is* the fast path, single-threaded and hot. Miss the tcache and you're into
bins and arena locking.

## 5. The instrumentation was the measurement, again

The first version of these numbers said the pointer was 7.3× slower, not 4.0×. `perf
record` put **56% of all cycles in `__aarch64_ldadd8_relax`**, GCC's outlined
helper for a relaxed atomic add. That was `counters::bytes.fetch_add` in the
counting `operator new` I had copied from a different benchmark. That other file
allocates 64 MiB per call and can afford a relaxed atomic without noticing. Here the
allocation is 36 bytes, and the instrument was dominating the thing it was
instrumenting. The counters are plain `uint64_t` now (the timed loop is
single-threaded and they're only ever read as a delta), and `RawPtr` dropped from
53.6 to 29.5 cycles.

This is §2 in a different costume: **check that you're measuring the program and not
the harness.** [The dead-code post](/posts/the-compiler-deleted-your-benchmark-loop/)
asks whether the code is still there. This asks whether it's the only thing there.

## 6. The control: what a call actually costs

`Inlined` is 7.6 instructions and 3.9 cycles against `Nrvo`'s 12.6 and 7.4. Five
instructions vanish when the call goes away: the branch and return, the ABI's echo
of the result pointer into `x0`, a register shuffle that only existed because that
echo clobbered the argument, and one store that didn't disappear at all but **moved
out of the loop**. Once the callee is visible, the address is provably the same every
iteration, and loop-invariant code motion hoists it.

That's the larger point. Four instructions of calling-convention ceremony don't
explain going from 7.4 cycles to 3.9. Inlining's real payoff is removing the opacity,
which unlocks constant propagation, dead-store elimination and code motion across
what used to be a wall. `Nrvo` and `Inlined` are within 0.15% at 16 KiB, and for a
small function in a header all seven variants compile to the same code. **This whole
post is about what happens across a call the optimiser can't see through.**

## The one-line version

**`return b;` is never slower than `return std::move(b);`, the pointer isn't the
cost of returning a pointer, and if your benchmark takes under 10 ns, check whether
it's measuring your code or your command line.**

<!-- FIGURE: the padding table as a strip plot, Nrvo vs MovedReturn per padding length -->
<!-- FIGURE: stacked bar of the perf record breakdown for RawPtr, allocator vs constructor -->
<!-- LINK: publish nrvo_move.cpp, bench_nrvo.cpp and the padded perf script -->
