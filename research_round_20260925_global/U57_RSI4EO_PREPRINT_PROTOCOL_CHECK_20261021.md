# RSI4EO preprint: protocol check for the SAR study

Checked 2026-10-21; literature evidence only. No reproduction or new project evaluation.

Zhao et al., *The Last Remote Sensing Model Built by Humans? Casting Any Earth Observation Task as Recursive Self-Improvement*, arXiv:2610.18845v2 (first submitted 13 October; revised 15 October 2026), reports HRSID mask AP 68.7. Its split is 3,090 training, 552 verifier and 1,962 test images; the verifier holds 20 training scenes. Each task is run once per agent within 12 hours on four GB200 GPUs. Appendix A.3 reports a half-pixel mask-refinement correction improving local AP from 56.4 to 60.7. These are author reports, not independent verification. [Official paper, Table 1, §4.1, Table 2(c), Appendix A.3](https://arxiv.org/html/2610.18845v2).

The [author repository](https://github.com/Yokoya-Lab-UTokyo/RSI4EO) still states that code is coming soon; no executable HRSID implementation was verified.

For this project, the new result is a reference to investigate, not a comparable score: our internal split and fixed-budget structure experiment differ. The coordinate example motivates checking existing nonzero geometry tests; it does not establish a bug here. Keep the locked training unchanged. No new AP, gain, novelty or TGRS-readiness claim follows.
