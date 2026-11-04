"""Experimental extensions around AlignFormer, kept apart from it.

Nothing here modifies ``alignformer_pva`` or Chenyi's checkpoint path: it
reads the unmodified model's Propose output (cached by ``propose_cache``) and
runs Verify/Abstain in C++ (``verify_core``). The default settings reproduce
the deployed AlignFormer exactly; variants are opt-in and named separately.
"""
