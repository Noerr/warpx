.. _developers-amrex-fork:

AMReX fork used by this branch
==============================

This branch does **not** build against ``AMReX-Codes/amrex``. It pulls AMReX
from ``https://github.com/Noerr/amrex.git``, branch
``noerr/restart-1x-read-26.08``, which is the upstream ``26.08`` tag plus
seven commits.

They address three *different* defects. The first two both surface during
checkpoint **restart** and were conflated during the investigation; the third
is on the **write** side and is unrelated to either:

.. list-table::
   :header-rows: 1

   * - commit
     - problem
     - resource
   * - ``219119eb6``
     - ``ReadParticles`` holds two full-box **host** copies at once
     - host (pinned + pageable)
   * - ``beb8c444b``
     - restart reads boxes on ranks that do not own them, forcing a full
       redistribute
     - **device** (arena)
   * - ``470aac36b`` ``cbad3f124``
     - async particle **writes** are unchecked, so a failed write produces a
       short checkpoint and the run still exits 0
     - correctness (silent data loss)
   * - ``dbed1035b`` ``41cfd2910`` ``10d5dac2e``
     - asynchronous output was invisible: not profiled, and nothing recorded
       when a write finished or when the drain at exit began
     - observability
   * - ``f4f45de84``
     - temporary diagnostics, off unless ``AMREX_RESTART_DEBUG`` is set
     - —

Of the two restart defects, ``beb8c444b`` is the one that actually blocks
restart at scale. ``219119eb6`` is real but was never the blocker, and earlier
revisions of this document wrongly implied it was.

Problem 3 is the most serious of the three in kind, if not in frequency: the
other two make a run fail loudly, whereas an unchecked write lets a run succeed
while producing output that cannot be restarted from. It took two commits,
because the obvious fix does not work; see below.

Problem 1: two full-box host copies (``219119eb6``)
---------------------------------------------------

``amrex::ParticleContainer::ReadParticles`` (``Src/Particle/AMReX_ParticleIO.H``)
reads one grid's entire int block and entire real block into full-box host
buffers, and then reassembles the whole box into a second set of full-box host
buffers before any copy to device. Both full-box host copies are live at the
same time, so the restart host transient is roughly twice the checkpoint write
transient — the write path stages only one full-box host copy per grid in
``packIODataGpu``.

For a pure-SoA container with seven real components that is about
128 B/particle on read against about 64 B/particle on write. Because the box
array is frozen at write time, this decides whether a large-particle checkpoint
can be restarted at all: a run that writes successfully on N nodes can fail to
read back on the same N nodes purely because of the 2x.

The patch keeps the (comparatively small) int block resident and streams the
real block in bounded segments, capping the **host** transient at approximately
the write footprint.

.. important::

   This is a host-memory change only. It measurably reduced the pinned arena
   (9,125 → 7,480 MB, −18% in one Aurora run) and does not touch device memory
   at all. Restart on Aurora still failed after it, on a device allocation —
   see Problem 2. Any claim that this patch brings "the restart" to write
   parity refers to host memory alone.

Why only the real block is streamed
-----------------------------------

Two reasons, of which the second is the decisive one.

**Size.** For a pure-SoA WarpX species the int block is
``(2 + NStructInt + NumIntComps) x 4 B`` = 8 B/particle (just id and cpu),
against ``(NStructReal + NumRealComps) x 8 B`` = 56 B/particle of reals. The
reals are ~87% of the raw payload, so eliminating the raw *real* buffer captures
almost all of the available saving:

.. list-table::
   :header-rows: 1

   * -
     - raw ints
     - raw reals
     - reassembled ``host_*``
     - total
   * - unpatched read
     - 8 B/p (full box)
     - 56 B/p (full box)
     - 64 B/p (full box)
     - **128 B/p**
   * - patched read
     - 8 B/p (full box)
     - bounded segment
     - 64 B/p (full box)
     - **~72 B/p**
   * - write path
     - —
     - —
     - 64 B/p (full box)
     - **64 B/p**

Streaming the ints as well would take ~72 B/p down to ~64 B/p, a further ~11%.

**File layout.** Per grid the checkpoint is written as
``[entire int block][entire real block]`` and ``ReadParticles`` consumes it with
a single forward-moving ``ifstream``: ``readIntData()`` reads the whole int
block, then the reals follow. Only the *trailing* block can be streamed with
purely sequential reads. Chunking the ints too would require two cursors
alternating between two file regions, with ``tellg``/``seekg`` bookkeeping per
chunk and a read pattern that jumps between offsets hundreds of MiB apart —
trading one large sequential read for a seek-heavy one, which is a poor trade on
Lustre. Given the ~11% remaining, that complexity is not justified.

Note what the residual actually is: after the patch the floor is the **full-box
reassembly buffer** (``host_real_attribs`` / ``host_idcpu``), which is built
across the whole ``for i in 0..cnt`` loop and copied to device only afterwards.
It is 1x by construction and cannot be reduced by chunking the file read at all;
removing it needs an on-device deinterleave (scatter the raw buffer into the
tile in a kernel) instead. The patch works because the *raw* buffer was the
redundant copy.

**When this reasoning stops holding.** The residual full-box int term scales
with ``2 + NStructInt + NumIntComps``. WarpX's species carry no compile-time int
attributes, so it is 8 B/particle and negligible. A configuration that adds
runtime int components would grow that term linearly and shrink the patch's
margin, at which point streaming the ints — or the on-device deinterleave —
would start to earn its complexity.

Problem 2: restart reads boxes on the wrong ranks (``beb8c444b``)
-----------------------------------------------------------------

This is the one that blocks restart at scale, and it is an upstream regression
between AMReX ``26.04`` and ``26.08``, introduced by AMReX PR #5497 ("Remove
dual_grid path from particle restart"). Reported as
`AMReX-Codes/amrex#5652 <https://github.com/AMReX-Codes/amrex/issues/5652>`__.

In ``26.04`` the normal path had every rank read the boxes it owned, using the
container's own ``DistributionMapping`` (``MFIter`` over ``m_dummy_mf``). The
``MaxReaders()`` cap of 64 existed, but only on the AMR level-loss fallback,
which a single-level run never reaches.

``26.08`` collapsed both paths into one, always distributing the read by a
mapping derived from the *file*, and always subject to the 64-reader cap.
Two consequences follow:

* **Read parallelism is capped at 64 ranks.** Per-reader payload grows as
  ``nranks/64`` — 6 boxes per reader at 384 ranks, 243 at 15,552.
* **Particles are read by ranks that do not own them**, so the
  ``Redistribute()`` at the end of ``Restart()`` must move essentially every
  particle.

The second is the more serious, because it is *not* fixed by restoring read
parallelism. A rank then holds concurrently: the foreign box's tile, a send
buffer sized to essentially all of it, a receive buffer for its own box
arriving from elsewhere, and the destination tile — plus partitioning scratch
(~11 GiB against ~26 GiB of read payload in one measurement). The restart
transient is therefore several times the steady-state particle payload.

.. important::

   That caps how much device memory a run may use **in steady state** and still
   be restartable. For a strong-scaling campaign that deliberately drives GPU
   utilisation toward 100%, this binds well before the run itself does. It is a
   scaling limit, not merely a restart inconvenience.

The patch reads each box on the rank that owns it whenever the checkpoint's box
array is the one the container is already using. Particles land in their final
tile, and ``Redistribute()`` takes its ``tot_snds == 0 && tot_rcvs == 0``
early-out, allocating neither a send nor a receive buffer. When the box arrays
genuinely differ, #5497's path runs unchanged, so its purpose is preserved.

Complementary runtime workaround
--------------------------------

``particles.nreaders = <nranks>`` restores full read parallelism without any
patch, by making ``NProcs() <= MaxReaders()`` true so the load-balanced branch
is taken. On Aurora this alone turned a failing restart into a successful one:

.. list-table::
   :header-rows: 1

   * -
     - default (``nreaders`` unset → 64)
     - ``nreaders = 384``
   * - ranks reading
     - 64 of 384
     - 384 of 384
   * - boxes per reader
     - 6
     - 1
   * - max payload / rank / species
     - 474 M particles
     - 79.1 M particles
   * - arena OOM aborts
     - 44–88
     - 0
   * - outcome
     - died in ``InitData``
     - restart completed, 257 steps

Keep this in the input decks regardless — it helps on stock AMReX builds. But
note it lowers the transient without removing it: each rank still generally
reads a box it does not own, so the redistribute still happens. Only the patch
eliminates it.

Problem 3: unchecked asynchronous writes (``470aac36b``, ``cbad3f124``)
-----------------------------------------------------------------------

Unrelated to the two restart defects above, and different in kind: this one does
not make a run fail. It makes a run *succeed* while producing a checkpoint that
cannot be restarted from.

``WriteBinaryParticleDataAsync`` validates the Header stream and aborts on
failure, but never checks the particle **data** stream. ``std::ofstream`` does
not throw by default, so ``writeIntData``, ``writeDoubleData`` and ``flush()``
all fail silently — a full disk, an exceeded quota or a transient OST error
produces a short file while execution continues normally — and the stream is
then closed by RAII, where a failed close is discarded.

The result is worse than a plain truncated file. The Header's per-grid particle
counts are computed from the in-memory container, not measured from what reached
disk, so the Header still claims the full count. Header and data disagree, and
nothing notices until a later restart trusts the Header and reads past EOF, in a
different job, as an abort deep in the read path.

Observed on Aurora at 3072 ranks: a checkpoint with 39 of 2688 electron data
files short by 31.4 GiB in total, written by a job that exited 0 with seven
minutes of walltime to spare; the chained successor died in
``RealDescriptor::convertToNativeDoubleFormat``.

The first patch (``470aac36b``) closes the stream explicitly and checks
``good()`` afterwards. Because the write runs on the ``BackgroundThread``, where
``amrex::Abort()`` may not produce a usable backtrace or a clean ``MPI_Abort``,
the failure is *recorded* rather than acted on there:

* ``AsyncOut::RecordWriteFailure()`` logs immediately with ``AllPrint`` (the
  failing rank is generally not the I/O process, and that line survives a job
  killed before any abort) and sets an atomic flag.
* ``AsyncOut::CheckWriteFailures()`` runs on the main thread and aborts. It is
  called when submitting a new job, so a run does not keep emitting output after
  its output has started failing, and again after the final drain in
  ``Finalize()`` — the latter is what guarantees a failed write cannot exit 0.

Why that was not enough (``cbad3f124``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``470aac36b`` was tested in production and **did not fire**. A run with the
check in place wrote a short checkpoint and still exited 0; nothing on the
stream ever reported an error.

The cause is delayed allocation. ``write(2)`` only dirties page cache — blocks
are not allocated and the quota is not charged until the kernel writes back — so
by the time the quota is exceeded, both ``write(2)`` and ``close(2)`` have
already returned success and the ``ofstream`` has no way to learn otherwise.
Checking the stream is necessary but detects only the subset of failures the
stream itself can see, which on a delayed-allocation filesystem excludes the
case that actually happened.

``fsync(2)`` documents this failure mode explicitly under ``EDQUOT``, "some
previous write failed due to insufficient storage space", and is the point at
which the error is still reportable. ``cbad3f124`` therefore forces write-back
and checks it: a second descriptor is opened on the file, the stream is written
and closed as before, and ``fsync()`` then runs on that descriptor, with a
failure from either it or its ``close()`` routed to
``AsyncOut::RecordWriteFailure()`` like any other.

The descriptor is opened **before** the writes, not after the close. Write-back
errors are delivered to the descriptors that were open on the file when the
error was recorded (``fsync(2)``, ``EIO``, Linux 4.13 and later), and write-back
can fire at any point once ``write(2)`` has returned, so a descriptor opened
after the fact may report success for a write that has already failed.
``std::ofstream`` exposes no portable way to reach its own descriptor, hence the
separate open; ``fsync()`` acts on the file rather than on the descriptor that
dirtied it, so it covers everything the stream wrote.

Controlled by ``amrex.async_out_fsync``, **default true**. The cost is real:
``fsync`` blocks until the data is durable where write-back is otherwise lazy
and overlapped, which lengthens the drain at exit and serialises ranks that
share an output file. It runs on the writer thread, so stepping is unaffected,
and a silently truncated checkpoint costs far more than a slower one. Set it to
``0`` only to establish what the syncing is costing.

One caveat on attribution: ``fsync`` flushes the whole inode, so where several
ranks share a file, one rank may surface another's error. At
``async_out_nfiles = nranks`` this is moot.

.. important::

   Detecting this in existing checkpoints requires comparing expected against
   actual bytes, using the Header's ``(which, count, where)`` triplets. A file
   census does not find it: file counts, headers and directory structure are all
   intact in a damaged checkpoint. Neither does ``find -size 0`` — a box with no
   particles legitimately writes an empty data file, and in the observed case
   384 files per species were legitimately zero-length while the real damage lay
   in files that were merely short.

Not addressed: the field/plotfile async path has not been audited for the same
gap.

Diagnostics (``f4f45de84`` and ``beb8c444b``)
---------------------------------------------

Set ``AMREX_RESTART_DEBUG=1`` to have the restart path report what it is doing.
Nothing prints unless the variable is set::

    [restart-dbg] lev=0 boxarray_matches=yes -> read_dm=container (no shuffle)
    [restart-dbg] rank=N lev=0 boxes_in_file=? boxes_this_rank=? particles_this_rank=? largest_box=?
    [restart-dbg] resize rank=N lev=0 grid=G tile=T old=? new=? cap=? delta_MiB=?
    [restart-dbg] copyPlan  rank=N tot_snds=0 tot_rcvs=0 -> EARLY-OUT, no buffers allocated
    [restart-dbg] packBuffer rank=N snd_buffer_bytes=0 (0 MiB) arena=device
    [restart-dbg] rcvBuffer  rank=N rcv_buffer_bytes=0 (0 MiB) nrcvs=0

The buffer lines print their size even when it is zero, so "no buffer was
allocated" appears as an explicit ``0`` rather than as an absent line. Note that
when the early-out fires, ``packBuffer`` and ``rcvBuffer`` do not print for that
call at all, because ``buildMPIStart`` returns first — so the positive
confirmation is the ``EARLY-OUT`` line together with the absence of nonzero
buffer lines during ``InitData``. Buffer lines reappear once stepping begins,
which is the check that the instrumentation is live rather than silently off.

``AMREX_RESTART_PRESIZE=1`` also exists, reserving destination tiles to their
known final size. Measurement showed every tile is sized exactly once
(``old=0`` on every resize line), so it is a confirmed no-op and is retained
only for completeness.

Status
------

``beb8c444b`` (Problem 2) is **validated on Aurora**. Same case as above,
384 ranks, one GPU per rank, instrumented build:

.. list-table::
   :header-rows: 1

   * - check
     - result
   * - read mapping
     - ``boxarray_matches=yes -> read_dm=container (no shuffle)``, both species,
       no ``=no`` lines
   * - redistribute
     - ``tot_snds=0 tot_rcvs=0 -> EARLY-OUT``
   * - send buffer
     - ``snd_buffer_bytes=0 (0 MiB)``
   * - receive buffer
     - ``rcv_buffer_bytes=0 (0 MiB) nrcvs=0``
   * - read spread
     - 768 reader entries, 0 idle, max 1 box/rank, 79.1 M particles
   * - per-step cost
     - 3.61 s against a 3.60 s baseline
   * - outcome
     - restart clean past STEP 566, zero OOM

Both buffer sizes are printed even when zero, so those are positive
confirmations of non-allocation rather than absent log lines. The read volume
is unchanged — it is simply performed by the rank that ends up owning the data
— which is why there is no measurable per-step cost.

Note that ``particles.nreaders`` has **no effect** on this path.
``MaxReaders()`` is reached only in the ``else`` branch, so when the box arrays
match, every rank that owns a box reads it and the setting is never consulted.
Keep the deck setting anyway: it is harmless here, it still governs
``InitFromAsciiFile`` / ``InitFromBinaryFile``, and it is the only mitigation on
a stock-AMReX build.

.. warning::

   ``cbad3f124`` (Problem 3, the ``fsync``) is **compile-verified only**. Take
   that seriously here rather than as boilerplate: ``470aac36b`` was also
   compile-verified, was also reasoned from the documented semantics, and then
   failed in production against the exact case it was written for. Only a run
   that genuinely exhausts its quota will establish that this one fires. Until
   then, keep checking checkpoints with the byte comparison described above.
   Neither commit has been reported upstream yet.

   ``219119eb6`` (Problem 1, host streaming) remains **unvalidated**. It has not
   been shown to reproduce an unpatched restart bit-for-bit. Its host benefit
   was measured (pinned arena −18%) but its correctness was not. It is also
   independent of the restart fix above — the Aurora restart succeeded because
   of ``beb8c444b``, not because of it.

   ``f4f45de84`` is explicitly temporary. Its ``AMREX_RESTART_PRESIZE`` path is
   a confirmed no-op; its logging is what made the diagnosis possible and is
   worth retaining until Frontier has run the same test.

Upstream status
---------------

Reported as `AMReX-Codes/amrex#5652
<https://github.com/AMReX-Codes/amrex/issues/5652>`__, with the fix commit and
the Aurora verification posted as a follow-up comment. If upstream takes the
change, revert this fork's ``beb8c444b`` and the two selection values below,
and return to stock AMReX.

Validation plan (for what remains unvalidated)
-----------------------------------------------

The restart fix (``beb8c444b``) has been validated — see Status above. What
follows applies to ``219119eb6``, the host streaming change, which has not.

The gating production problem type is **full-PIC (ions + electrons)**, not
hybrid: it carries two kinetic species rather than one, so it has roughly twice
the particle payload per box and is the case that actually decides whether a
restart fits. Validate against that, not against a hybrid case.

Because both selection values can be overridden at configure time, patched and
stock builds can be produced from identical WarpX source::

    # stock AMReX, same WarpX tree
    cmake -S . -B build_stock \
        -DWarpX_amrex_repo=https://github.com/AMReX-Codes/amrex.git \
        -DWarpX_amrex_branch=26.08

Two things to establish, in order:

1. **Correctness.** Restart the same checkpoint under both builds and compare
   particle data. The device-resident result should be identical; the patch
   changes only how the host staging buffer is filled.
2. **Host-memory benefit.** Restart a checkpoint whose box array forces the
   unpatched read over the node memory limit, and confirm the patched build
   completes at a node count where the stock build OOMs. Peak RSS should
   approach the checkpoint *write* footprint rather than twice it.

Reverting to upstream AMReX
---------------------------

Two committed values select the fork. To go back to stock AMReX:

* ``cmake/dependencies/AMReX.cmake`` — set ``WarpX_amrex_repo`` back to
  ``https://github.com/AMReX-Codes/amrex.git``
* ``dependencies.json`` — set ``commit_amrex`` back to ``26.08``

Both can also be overridden at configure time without editing the tree, as
shown above, which is the quickest way to A/B the patch.

Note the pin is by **branch name**, not by SHA, so a rebuild picks up any new
commit pushed to that branch. Pin a SHA instead if a build needs to be frozen.

Rebasing the patch onto a newer AMReX
-------------------------------------

When the AMReX pin moves, rebase the fork branch rather than editing the
fetched tree::

    git clone git@github.com:Noerr/amrex.git
    cd amrex
    git remote add upstream https://github.com/AMReX-Codes/amrex.git
    git fetch upstream --tags
    git checkout -b noerr/restart-1x-read-<new> <new-tag>
    git cherry-pick noerr/restart-1x-read-26.08
    git push origin noerr/restart-1x-read-<new>

then update ``commit_amrex`` here to the new branch name.

The same patch on the ``26.04`` tag is preserved as
``noerr/restart-1x-read-26.04`` on the fork. It is archival only — the
``WarpX_26.04+patched`` branch deliberately still builds against stock AMReX,
because restarts have been running acceptably there with the 2x read and there
is no reason to carry an unvalidated patch into a working production base.

Note on the build
-----------------

``ReadParticles`` is a template member function defined in a header, so it is
instantiated in the *consuming* translation unit — here
``Source/Diagnostics/ParticleIO.cpp.o`` in WarpX, not in ``libamrex_3d.so``
(which contains no ``ReadParticles`` symbol at all). A build cannot pick up this
patch by relinking against a different AMReX library: **WarpX must be recompiled
from source with the fork pinned.** That is why the selection lives in
``dependencies.json`` and ``WarpX_amrex_repo`` rather than in a deployment step.
