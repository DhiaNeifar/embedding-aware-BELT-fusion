"""AlignFormer as used in the paper, around the unmodified ``alignformer_pva`` model.

``propose_cache`` caches Propose (the unmodified network) per noise condition; ``verify_cpp`` runs Verify and
Abstain in C++ (``verify_core``); ``temporal`` follows each CAV over time with each vehicle's odometry
(``odometry``); ``latency``, ``async_cache``, ``fusion`` and ``freealign_clock`` cover delayed messages in late
fusion; ``evaluate`` scores late fusion (NMS and AP in C++, ``fusion_core``). ``variants`` names what the sweeps run.
Nothing here reads the noise level: only the simulator in ``propose_cache`` does.
"""
