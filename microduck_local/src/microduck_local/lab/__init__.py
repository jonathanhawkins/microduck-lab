"""The lab server's own package — what `viz_server.py` is made of.

`viz_server.py` is 4,500 lines: the 50 Hz loop, the socket, the HTTP surface,
teach jobs, captures, policy discovery, run management *and* every robot
branch. `docs/mars-roadmap.md` §6.4 names the extraction that makes the
registry stick: `lab/robots.py`, so the NEXT body never opens `viz_server.py`.
"""
