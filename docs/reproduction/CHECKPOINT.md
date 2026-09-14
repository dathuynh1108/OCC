# Reproduction delivery checkpoint

All selected experiments completed on 14 September 2026: native PatchCore45/45,
Deep SVDD400/400, DROCC60/60 exports, shallow400/400, common MVTec495/495 metric
rows. All matrix exits report no open failures. Independent audit passed on
8,262,100 image predictions and native per-epoch image/optimizer-step counts.

Declared NBD:78.66 ±1.99% image AUROC;90.99 ±0.77% AP. All weak results and ablations
remain visible. Original v2 files and NBD source hashes are unchanged.

User chose about two hours and reduced epochs. Native Deep AE5/SVDD12; DROCC5;
common AE5/Deep15/DROCC15. Full datasets/classes/seeds remained selected. This is
not full-schedule historical numerical reproduction; see SOURCE_AUDIT.md.

The final 29-page Beamer PDF and Overleaf bundle have compiled and been visually
reviewed. Root CODEX_OCC_PAPER_REPRODUCTION.md and the result handoff point to the
measured tables, source references and rerun commands.

Lifecycle status is authoritative in results/reproduction-2026-09-14/execution_closure.json.
Instance 50996199 was destroyed after verified off-box backup and Git delivery.
Provider listing confirmed its absence at 2026-09-14T12:16:26.979079+00:00. All 5,999 snapshot files
are verified across the local checkout and GitHub Release; see BACKUP_RESTORE.md.
Initial long-run checkpoints remain separate.
