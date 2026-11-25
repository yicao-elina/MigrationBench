# Production uncertainty audit

- Balance/equilibration exclusion: first `100 ps` removed.
- Production blocks: `8`.
- Cr MSD/Dxy: multi-origin, fixed-cell unwrapped coordinates, origin stride `10`; Dxy uses the 2D Einstein factor 4.
- Family replica standard errors are reported only when at least two training-seed replicas pass the production gates.
- A completed 1 ns trajectory remains exploratory until fit-window, block and replica estimates agree.
- Status: BLOCKED; zero candidate runs passed the complete-step and finite-state production gates, so no block/replica estimate is reported.
