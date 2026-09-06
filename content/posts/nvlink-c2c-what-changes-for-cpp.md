---
title: "NVLink-C2C Removes the Copy, Not the Synchronization"
date: 2026-09-06T00:00:00+07:00
draft: false
tags: ["cuda", "gpu", "gb10", "dgx-spark", "memory", "cpp", "coherence", "systems-design"]
categories: ["systems", "performance"]
description: "What NVLink-C2C is, how it differs from PCIe, how cache coherence behaves once the CPU and GPU share one memory, and what a C++ programmer does differently because of it."
summary: "The CUDA Programming Guide assumes the host and the device keep separate memories with a copy between them. My DGX Spark doesn't. So I went and learned what NVLink-C2C actually is compared to PCIe, how coherence behaves across it, and what that changes for someone writing C++ on this machine."
toc: true
---

The CUDA C++ Programming Guide tells you what kind of machine it's describing in one
sentence, in §2.4. The model "assumes that the CUDA threads execute on a physically
separate device that operates as a coprocessor to the host", and that "both the host
and the device maintain their own separate memory spaces in DRAM, referred to as host
memory and device memory". Everything else in the guide follows from that. You
allocate on the device, you copy across, you launch, you copy back.

The machine on my desk doesn't match the sentence. A DGX Spark has a GB10 in it: a
Grace CPU die and a Blackwell GPU die on one package, sharing a single pool of 128 GB
LPDDR5X, joined by NVLink-C2C rather than PCIe. I'd already
[measured what that does to `cudaMemcpy`](/posts/zero-copy-gb10/) — in every sweep I
ran, explicit staging never won, and it cost twice the memory — but I'd done that
without really understanding why. I knew the copy was pointless. I didn't know what
had replaced it.

So I spent an afternoon learning what NVLink-C2C actually is, how it differs from
PCIe, and what "coherent" means when it's the hardware doing it rather than me. This
is what I found, and what I think it changes for a C++ programmer.

## PCIe and NVLink-C2C are two different kinds of link

A CPU chip and a GPU chip each have their own cores and their own caches. When one
of them needs a byte the other one has, that byte has to cross a physical link.
There are two links in this story, and the difference between them isn't mainly
speed.

**PCIe** is the standard one. It's wires on the motherboard and a packet protocol,
and every add-in card uses it: GPUs, NICs, NVMe drives. The sender wraps each chunk
of data in a header, the receiver unwraps it. It's slow compared to memory — PCIe
5.0 x16 gives about 64 GB/s per direction, while H100's HBM3 gives about 3350 GB/s,
so on a discrete card the bus is roughly fifty times slower than the memory behind
it. But the property that matters most here is a different one: PCIe has no opinion
about whether the two memories on either end agree. A CPU write to DRAM is
invisible to the GPU until software copies it over. That copy is `cudaMemcpy`, and
no amount of bandwidth makes it go away.

**NVLink-C2C** joins two dies that sit on the same package. NVIDIA's Grace Hopper
architecture post puts the GH200 link at 900 GB/s total; for GB10, the DGX Spark
announcement calls it "a CPU+GPU-coherent memory model with 5x the bandwidth of
fifth-generation PCIe", and the Hot Chips 2025 figure is about 600 GB/s aggregate.
That's more than double what the 273 GB/s LPDDR5X pool can supply, so on this
machine the link is never the thing you wait for.

But bandwidth is the least interesting part. Because the dies are close, the link
can be wide and short. And because NVIDIA owns both ends of the wire, it doesn't
have to carry packets — it can carry **memory requests**. The GPU can read a CPU
address as if it were its own, and the hardware takes on the job of keeping both
sides in agreement.

The picture I ended up with: PCIe is a public highway with toll booths. NVLink-C2C
is a private bridge between two buildings owned by the same company. The bridge
being wider is nice. The bridge being *private* is what lets you leave the doors
open at both ends.

## What the link gives you, and what it doesn't

Put plainly, NVLink-C2C gives you three things where PCIe gives you one:

1. **Bandwidth.** The wide, short link.
2. **One address space.** The CPU and the GPU use the same pointer for the same
   byte.
3. **Hardware coherence.** When one side writes a byte, the other side sees the new
   value without a software copy.

PCIe can't give you the second or the third at any speed. The change isn't a wider
road. It's a road that carries a different kind of traffic.

The second one took me a while to get right, so it's worth slowing down on.

### One address space is not one memory

"One address space" sounds like "one memory", and on GB10 it happens to also be
true, but the two are separate ideas and the difference is exactly what separates
GB10 from its datacenter siblings.

An address space is a numbering scheme for bytes. Your program uses virtual
addresses, and a page table maps each virtual page to a physical frame. The CPU's
MMU — its memory management unit — walks that table on every access. The GPU has its
own MMU doing the same job. On a PCIe card, each MMU has its own page table, so
address `0x1000` on the CPU and `0x1000` on the GPU are two different bytes in two
different DRAMs. On NVLink-C2C, both MMUs read the *same* page table, so `0x1000` is
the same byte whoever asks.

Notice what that doesn't say. It says nothing about how many physical memories
there are.

| | address spaces | physical pools |
|---|---|---|
| PCIe GPU | two | two (DRAM + HBM) |
| GH200 | one | two (LPDDR5X + HBM) |
| GB10 | one | **one** (128 GB LPDDR5X) |

GH200 has one address space over two pools. On that machine the page table decides
which pool holds each page, speed depends on which side the byte lives on, and
moving pages between pools is real work — the guide's Unified Memory chapter, with
its prefetch and advice calls, is largely about managing that. GB10 has one address
space over *one* pool. There's no second place for a byte to be. That's why a 70B
model fits on a Spark: the GPU can use nearly all 128 GB, where a PCIe card is stuck
with whatever HBM is soldered onto it.

Two things I'd assumed that turned out to be wrong. First, I'd imagined some
translator sitting between the two processors, because they run different
instruction sets — the CPU runs Arm, the GPU runs SASS, its own native instruction
set. There isn't one. Nothing translates between them. Each processor runs its own
code; what they share is the map, not the language. The page table is a phone book,
and each MMU is a person reading it. Second, "unified" doesn't mean the compute is
unified. Host code still runs on the Arm cores, kernels still run on the GPU, you
still launch with `<<<grid, block>>>`. Memory is shared. Compute is not.

And unified doesn't mean equally fast. 273 GB/s of LPDDR5X is a long way from the
3000-plus of datacenter HBM. GB10 traded bandwidth for capacity and simplicity.

One last piece of the address-space story. Each MMU caches recent lookups in a TLB,
so when the CPU changes a page table entry, the GPU's TLB may still hold the old
answer. NVLink-C2C carries the message that evicts it — on GB10 this goes by the
name ATS, Address Translation Services, and `cudaDeviceGetAttribute` reports it as
`pageableMemoryAccessUsesHostPageTables=1`. That's the *translation* half of keeping
two processors in agreement. The *data* half is coherence, and that's the part that
actually changes how you write code.

## How coherence behaves on NVLink-C2C

Caches exist because DRAM is slow. A read of `0x1000` pulls a 64-byte line into the
processor's cache, and later reads hit there.

Now put two caches in the picture. The CPU and the GPU both read `0x1000`, and both
cache the line. The CPU writes a new value; it lands in the CPU's cache. The GPU's
cache still holds the old one. The shared page table did nothing to help — both
sides agree perfectly on *where* the byte lives, and one of them is holding a stale
copy of it anyway.

Coherence is the rule that says this can't stand: when one side writes a line,
every other copy either goes invalid or gets the new value. Someone has to enforce
it, and who does is the whole difference between the two links.

On PCIe, nobody enforces it in hardware. Software does, at `cudaMemcpy` and
`cudaDeviceSynchronize`, and between those calls the two sides are allowed to
disagree. You write your code so that it doesn't care.

On NVLink-C2C, the hardware enforces it. A CPU write sends a message across the
link, the GPU marks its copy invalid, and the GPU's next read misses and fetches the
fresh line. Per cache line, all the time, no driver call. NVIDIA's Grace Hopper post
says it plainly: hardware coherency "enables the Grace CPU to cache GPU memory at
cache-line granularity and for the GPU and CPU to access each other's memory
without page-migrations." Underneath it's Arm's AMBA CHI protocol, which keeps track
of who has what with a snoop filter — a small hardware table recording which side
holds which 64-byte line.

So the CPU and GPU can genuinely work on one data structure at the same time. A CPU
thread appends to a queue while a kernel drains it. A GPU `atomicAdd` and a CPU
`std::atomic` on the same counter produce the right total.

### The part coherence doesn't do

This is the thing I'd have got wrong in production, and it's why the title says
what it says.

Coherence promises that when the GPU reads a line, it gets the latest value of
*that line*. It promises nothing about the order in which two *different* lines
became visible. So the CPU can write a block of data and then write a "ready" flag,
and the GPU can see the flag set while the data behind it is still old. Each line
is perfectly up to date. Their relationship isn't.

That's exactly the situation between two CPU cores, and it has exactly the same
fix: the producer stores the flag with release ordering, the consumer loads it with
acquire ordering, and that pair is what turns "these lines are each current" into
"this one is current *because* that one is". On the GPU side the atomic also needs a
scope — how far its promise reaches: one block, the whole GPU, or the whole system —
and across the die boundary it has to be system scope. A device-scope atomic keeps
GPU threads in order with each other and says nothing at all about the CPU.

Coherence removes the copy. It does not remove the release and acquire pair.

### What it costs

Having the hardware send invalidations around isn't free, and my next question was
how much it costs and whether you can turn it off for data the GPU never looks at.

It costs something in three places. Every write to a line the other side also
caches sends an invalidation across the link — false sharing, except the two cores
are on different dies. Each invalidation turns the other side's next access into a
miss, and where a local L2 hit is tens of nanoseconds, a fetch across the link ought
to be hundreds. (That's a guess from how two-socket servers behave; I haven't
measured it, and no document I found states it.) And the snoop filter itself is
silicon that was paid for once, plus a small constant delay on some misses.

The "can you turn it off" question turned out to have a better answer than yes or
no. When the CPU writes a line, the tracking logic asks a single question: does the
GPU hold a copy? If the GPU never read that line, no message is sent and the write
completes at CPU speed. A CPU-only program pays nothing — not because it opted out,
but because the expensive step never runs.

Coherence is a doorbell, not a heartbeat. It rings only when someone who holds a
copy needs to hear about a change. Nobody rings for an empty room.

## What we do about it as C++ programmers

That doorbell reframes the whole thing. What you control isn't coherence. It's **who
touches which line**.

Data that one side writes once and the other side only reads — model weights are
the clean example — costs one fetch and then nothing, forever. Data that both sides
hammer on the same line is the one case that can actually lose to PCIe, because
PCIe pays one large copy per handoff in software, and NVLink-C2C pays many small
invalidations in hardware, on exactly the lines you share.

So the practical rule is one you already know from two-socket servers. Keep the
data one side writes often out of the other side's working set. Pad your shared
structures so a CPU-owned field and a GPU-owned field never sit in the same 64-byte
line. Share a line only at the small handoff points: a flag, a queue head.

The allocators mostly cooperate with this. Ordinary `malloc`, `new`, and
`std::vector` memory is all reachable by the GPU, and the GPU only touches it if a
kernel actually dereferences the pointer. `cudaMalloc` is the exception: the DGX
Spark Porting Guide says memory from it is *not* CPU-coherent, so it's the one
allocator still playing by the old rules. (The zero-copy post found a second odd one
in `cudaHostAlloc`, for an unrelated reason.)

Two things I'd assumed about the change that turned out to be off. I'd thought
`cudaMallocManaged` was the new part — it isn't. It's been in CUDA since version 6,
and on a PCIe card it works by taking page faults and doing a hidden copy over the
bus. On GB10 the copy isn't hidden, it's *gone*, and you don't need a special
allocator to get that. A plain `malloc` pointer is a valid kernel argument. And I'd
thought the thing to worry about was data residency — which side the data lives on.
But there's one pool, so nothing resides anywhere else. The worry that's left is
*ownership*: which side is caching a line right now, and which side is about to
write it.

That's the job change. It used to be about moving data. Now it's about partitioning
it. And once I saw it that way, the two sentences I'd been trying to write all
afternoon fell out on their own:

> **On GB10, any pointer your C++ code holds is also a valid GPU pointer, so the
> host-to-device copy step disappears and a kernel can read a `std::vector` in
> place. What remains is the same discipline you already use for multi-core code:
> decide which side owns each piece of data, keep the handoff points small, and
> hand off with release and acquire atomics rather than with copies.**

The first sentence is what disappears. The second is what stays, and it ties this
hardware to something a C++ programmer has already been trained on. Every later
post in this series is going to be one of those two sentences, measured.

## What I still don't know

A few of the claims above are reasoning rather than something I've seen, so they're
still open.

**How much a cross-die miss actually costs.** "Hundreds of nanoseconds" is a guess
borrowed from two-socket servers. The experiment is small — one 64-byte line, the
two sides taking turns writing it, then the same thing with the two fields padded
apart — and I haven't done it.

**Whether a device-scope atomic is enough across the die.** The language says you
need system scope for the CPU to be included in the promise. Whether GB10's hardware
happens to make device scope work anyway, I can only find out by counting.

**Whether the `cudaMalloc` exception is as strict as the porting guide says.** I
don't know whether a CPU read of it faults, reads stale, or quietly works.

**Whether GB10 tracks coherence with the same snoop filter NVIDIA documents for
GH200.** Same CPU family, same link, so I've assumed yes. It's an assumption.

If any of those come back differently from what I've written, this post gets a
correction at the top.

<!-- FIGURE: two-road diagram, PCIe packets vs NVLink-C2C memory requests -->
<!-- FIGURE: two MMUs reading one page table over one pool, GH200 beside it with two pools -->
<!-- FIGURE: coherence timeline, CPU write -> invalidation -> GPU miss -> fresh fetch -->
<!-- SOURCES: CUDA C++ Programming Guide 12.8, §2.4 Heterogeneous Programming
     (docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide, the quoted sentences in the opening);
     NVIDIA DGX Spark announcement (nvidianews.nvidia.com, "5x the bandwidth of fifth-generation PCIe");
     NVIDIA Hot Chips 2025 via chiplog.io (600 GB/s aggregate C2C);
     NVIDIA Grace Hopper Superchip Architecture In-Depth (developer.nvidia.com blog,
     900 GB/s, cache-line-granularity coherence, "without page-migrations");
     NVIDIA NVLink-C2C product page (AMBA CHI);
     DGX Spark Porting Guide, CUDA section (cudaMalloc not CPU-coherent);
     CUDA Programming Guide, Unified Memory (pageable memory access, cudaMemAdvise).
     PCIe 5.0 x16 and H100 HBM3 figures are the vendor spec sheets. -->
