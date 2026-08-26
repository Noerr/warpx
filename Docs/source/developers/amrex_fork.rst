.. _developers-amrex-fork:

AMReX fork used by this branch
==============================

This branch does **not** build against ``AMReX-Codes/amrex``. It pulls AMReX
from ``https://github.com/Noerr/amrex.git``, branch
``noerr/restart-1x-read-26.08``, which is the upstream ``26.08`` tag plus a
single commit.

Why
---

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
real block in bounded segments, capping the host transient at approximately the
write footprint.

Status
------

.. warning::

   The patch is **not yet validated**. It has not been shown to reproduce an
   unpatched restart bit-for-bit, and its host-memory benefit has not been
   measured on this branch. It is also not yet proposed upstream. Treat restarts
   built from this branch as needing verification, not as a settled fix.

Reverting to upstream AMReX
---------------------------

Two committed values select the fork. To go back to stock AMReX:

* ``cmake/dependencies/AMReX.cmake`` — set ``WarpX_amrex_repo`` back to
  ``https://github.com/AMReX-Codes/amrex.git``
* ``dependencies.json`` — set ``commit_amrex`` back to ``26.08``

Both can also be overridden at configure time without editing the tree::

    cmake -S . -B build \
        -DWarpX_amrex_repo=https://github.com/AMReX-Codes/amrex.git \
        -DWarpX_amrex_branch=26.08

which is the quickest way to A/B the patch against stock AMReX.

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
``WarpX_26.04+patched`` branch deliberately still builds against stock AMReX.
