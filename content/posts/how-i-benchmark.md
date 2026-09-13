---
title: "How I Benchmark on the GB10 CPU"
date: 2026-09-13T00:00:00+07:00
draft: false
tags: ["cpp", "benchmarking", "performance", "arm", "methodology"]
categories: ["systems", "performance"]
description: "The method behind every measured post on this blog: run on one fixed core, repeat, check how much the repetitions disagree before reading the mean, prefer the median, and compare variants inside the same run. Google Benchmark 1.9.5, GCC 13.3, heterogeneous ARM."
summary: "The same benchmark landed at 3.47 ms and 7.01 ms on different afternoons while reporting under 0.3% variation both times. These are the habits that survived that."
toc: true
---

I ran the same benchmark on two different afternoons and got 3.47 ms one day and
7.01 ms the other, and both times the tool told me its repeated runs agreed with each
other to a fraction of a percent. That was the moment I stopped treating a benchmark
number as a fact about the code. This post is the short list of habits I picked up
afterwards, so that the numbers in every later post here actually mean something.

## Summary

| habit | why | section |
|---|---|---|
| Force the program onto one fixed CPU core | on this machine it decides which chip you're measuring | §1 |
| Touch the memory once before the timed part | the first touch is the operating system doing page work, not the cache | §2 |
| Let Google Benchmark decide how many times to loop | reading the clock costs 25 ns | §3 |
| Always repeat, check the spread first, use the median | past about 5% spread the run can't tell two variants apart | §4 |
| Compare variants inside one run, as a ratio | absolute numbers drift 2× between sessions | §4 |
| Don't trust the cache sizes Google Benchmark prints | it read them from the wrong core | §5 |

Everything below was measured on one machine, and I'll describe it once so the rest
makes sense. It's an NVIDIA GB10, the chip in the DGX Spark. Its CPU side is ARM, with two kinds of core on the same chip: ten fast
"performance" cores (ARM calls them Cortex-X925) and ten slower, cooler "efficiency"
cores (Cortex-A725). ARM's name for that arrangement is big.LITTLE. The timing tool
is Google Benchmark, a C++ library that runs a function in a loop and reports how
long each pass took, and the compiler is GCC 13.3 with normal optimisation on
(`-O2`). Unless a post says otherwise, the number came from CPU 15, one of the fast
cores.

## 1. Run on one fixed core, because it decides which chip you measure

By default the operating system is free to move a running program from one core to
another whenever it likes. You can stop that with a command called `taskset`, which
tells the kernel "only ever run this program on core 15". People usually do this to
make the timings less noisy, and that's what I thought it was for too. Then I ran the
same binary on a slow core and a fast core in the same minute.

The benchmark walks a matrix two ways: along the rows, which is how the numbers sit
in memory, and down the columns, which jumps around in memory and is kind to no
cache.

| | CPU 0 (efficiency core) | CPU 15 (performance core) |
|---|---:|---:|
| along the rows | 7.65 ms | 6.68 ms |
| down the columns | **138 ms** | **36.1 ms** |
| **how much worse columns are** | **18×** | **5.4×** |

That bottom row is the point. On one core I'd write down "column order costs 18×",
and on the other I'd write "5.4×", and both are true of real hardware. A run that
gets moved between cores partway through gives you a blend of the two, which is a
number that describes no chip at all.

Fixing the core prevents three separate problems. You stay on the same silicon. The
hardware performance counters, which are per core type on this machine, all count
the one core you're on rather than half each. And you keep the small, fast caches
that sit next to the core (L1 and L2) warm with your data, instead of abandoning
them every time the program moves. On a machine where every core is the same, only
the third one matters.

One thing I got wrong: `taskset -c 15` takes a *list* of cores, so that means "core
15", not "fifteen cores". And it doesn't reserve the core for you. Other programs
can still run there. The longer story is in
[the cache-misses post](/posts/cache-misses-heterogeneous-arm/).

## 2. Touch the memory once before you start timing

The first time I pointed the Linux profiler `perf` at a cache benchmark, its counts
of cache misses didn't line up with the timings at all. The explanation turned out
to be boring. When a program allocates a big block of memory, the operating system
doesn't actually hand over the pages until the first time each one is touched, and
that first touch is comparatively slow. It has nothing to do with the cache; it
shows up in the "system time" column, the time the kernel spent working on your
behalf.

So the array gets allocated and written once *before* the timed loop, and by the
time the timing starts the pages are all real. Two things I should be honest about.
I haven't measured how much the result shifts if you skip this step, only that
`perf` blamed the wrong thing until I did it. And if what you actually want to
measure is that first-touch cost, say for a cold start, then warming up is the
mistake rather than the fix.

## 3. Let Google Benchmark decide how many times to loop

For a long while I read Google Benchmark's `Time` column as the time for one pass
through the loop. What it really does is pick a number of passes big enough that
the whole batch takes at least a set minimum (the `--benchmark_min_time` option),
start the clock once, run the batch, stop the clock, and divide. So `Time` is an
average over a batch. If one pass in that batch was unusually slow, it gets smeared
into the average and there's no way to see it afterwards.

There's a good reason it works like that. Reading the clock isn't free. On this
machine a call to `steady_clock::now()` costs about 26 ns, even though the clock
itself ticks every nanosecond. If you read the clock around each pass of a loop
whose body takes a few nanoseconds, you're mostly timing the clock. So you time the
whole loop and divide, which is exactly what the library does for you. (Use
`steady_clock` for this, not `system_clock`. The system clock is the wall clock,
and the network time service is allowed to nudge it between your two readings.)

That pass count, shown in the `Iterations` column, has a second use. If it comes out
in the millions or trillions next to a time of nearly zero, the compiler has quietly
deleted your loop as dead code. [That post](/posts/the-compiler-deleted-your-benchmark-loop/)
covers how and why.

## 4. Repeat, check the spread first, then use the median

Google Benchmark will only give you statistics if you ask it to run the whole thing
more than once (`--benchmark_repetitions=5` and so on), which is reason enough to
always ask. The first statistic I look at is the spread between those repetitions.
The library reports it as `_cv`, the coefficient of variation, which is the standard
deviation as a percentage of the mean. If it says 10%, the repetitions typically
disagreed with each other by about a tenth.

Across one session on this machine I saw spreads of 0.07%, 0.23%, 5.06%, 7.56%,
7.61%, 10.70%, 10.84% and 17.81%, and the difference was almost entirely what else
the machine was doing at the time. The wide ones were wider than several of the
effects I was trying to measure that day. **So if the spread is above about 5%, I
don't use the run to compare anything.** That threshold comes from my data, not from
any theory.

### The part where the spread misled me

Here's the thing I got wrong. A benchmark called `BM_AoS` came out at **3.47 ms** in
one session and **7.01 ms** in another, and the spread was under 0.3% both times.
The repetitions inside each session agreed with each other beautifully. They just
agreed about a different machine, because the clock speed and the temperature of the
chip had drifted between the two sessions, and nothing inside a single run can see
that.

The rule that survived is to **put the variants you're comparing in the same
command and look at the ratio between them.** I never compare a number from today
against a table from last week. That's why the tables on this blog carry ratios, and
why the absolute columns are there to give you a sense of scale, not to compare.

### Read the aggregate rows one column at a time

Google Benchmark's built-in statistics are mean, median, standard deviation and
that spread figure. There's no minimum unless you register one yourself, which I do,
because the fastest repetition is a useful thing to see. Here's a two-repetition run
of a benchmark that reports both time and throughput, with my `_min` row underneath:

| repetition | time | throughput |
|---|---:|---:|
| 3 | **6.89 ms** | 9.06 GiB/s |
| 2 | 8.40 ms | **7.44 GiB/s** |
| `_min` | **6.89 ms** | **7.44 GiB/s** |

Every cell in that table is a real measurement. The bottom *row* is not. The library
takes the minimum of each column on its own, so the row pairs the time from
repetition 3 with the throughput from repetition 2, a combination that never ran.
The same goes for the wall-clock and CPU-time columns, so a minimum row can quietly
mix three repetitions.

The median is computed the same column-by-column way. It only looked coherent in my tables because
with an odd number of repetitions the middle time and the middle throughput happen
to come from the same run. With an even number, the two middle values get averaged,
and averaging doesn't commute with "one over": the median row for the table above
would say 7.645 ms and 8.25 GiB/s, when 7.645 ms actually corresponds to 8.17. Off
by 1% instead of 20%, but still a row that never happened.

### Why I still report the median

The usual case for the minimum goes: noise can only make a run slower, so the fastest
run is the cleanest. That's true if every repetition is the same cost plus some
random delay on top. It isn't. Between one repetition and the next the clock speed
and chip temperature shift, the pages land somewhere else, and the allocator and the
caches are in a different state, and any of those can make one run persistently
faster than the code deserves. The minimum then reports the luckiest configuration.
Worse, it's not stable: add more repetitions and it can only ever go down. The
median settles.

So the median is my default estimate of what a run costs, and I print the minimum
next to it when the floor itself is the question, knowing that it's a floor.

### And look at the worst case when there's a latency budget

One more reason not to stop at the average. Calling `reserve` on a container, which
asks it to grab all its memory up front instead of growing as it goes, improves
total throughput by about 25%. That's the number the usual advice quotes. Measured as
the single slowest operation out of a million, it improves `vector::push_back` by
22× and `unordered_map::insert` by 63×, because without it one insert in a million
had to rebuild the whole table and stalled for five milliseconds. No average shows
that. When a post here has a latency budget in mind, it reports the worst operation.

## 5. Where the tools disagreed with me

Google Benchmark prints a description of the machine at the top of every run:

```
Run on (20 X 2808 MHz CPU s)
L2 Unified 512 KiB (x20)
L3 Unified 8192 KiB (x2)
```

Those are the efficiency-core numbers, because the library reads them from CPU 0 and
has no idea the machine has two kinds of core. Sitting on CPU 15 I'm actually on a
3900 MHz core with 2 MB of L2 cache and 16 MB of L3, four times and twice what's
printed. That matters because cache sizes are exactly what you size a test against.
If I'd believed the header and built a "just bigger than L3" test at 8 MB, it would
have sat comfortably inside the real 16 MB cache and measured nothing I intended.
The per-core numbers are under `/sys/devices/system/cpu/`, and those are the ones I
use.

The other disagreement was with my own prediction. In a test where I made the array
64 times bigger, I expected it to take 64 times longer. It took 110 times longer,
because the small array fit in the L3 cache and the big one didn't. If I hadn't
written "64×" down before running it, I'd have glanced at 110× and called it roughly
linear. Writing the prediction down first is what turned a number into a finding.

## The command I run

```bash
taskset -c 15 ./build/bench --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_filter='BM_(RowMajor|ColMajor)$'
```

One fixed core, five repetitions, both variants in the same run, and I read the
spread before anything else. When I'm collecting hardware counters with `perf` the
rule flips: `perf` counts the whole process, not one function, so each variant runs
on its own and I compare the difference between two runs.

<!-- FIGURE: the eight spread values as a strip plot, with the 5% line -->
<!-- LINK: publish bench_cache.cpp, README_BENCH.md and the run script -->
