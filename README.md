 # NetSVG – Cylinder Net Generator

 Generates an SVG that looks like a “net” made from nodes and connections.
 The base shape is a circular sector (“cake slice”), and the cylinder edge can be made organic via a boundary profile.

 ## Features
 - SVG output in millimeters (viewBox + mm sizing)
 - Sector rendering (start angle + sweep angle)
 - “Depth look” via z-distribution + color strength + node radius scaling
 - Organic cylinder edge (“Rand”) via boundary profile (control points + smoothing)
 - Optional progress display (tqdm if installed, otherwise console fallback)

 ## Requirements
 - Python 3.10+ (recommended)
 - Packages:
   - svgwrite
   - tqdm (optional, for nicer progress)

 ## Setup
 ```bash
 python -m venv .venv
 .venv\Scripts\activate          # Windows PowerShell
 # source .venv/bin/activate     # Linux/macOS
 pip install svgwrite
 pip install tqdm               # optional
 ```

 ## Run
 ```bash
 python net_graph.py
 ```
 This writes `net_cylinder.svg` to the project folder.

 ## Configuration (Parameters at the top of the script)
 The script is controlled by grouped parameters near the top. Typical ones:

 ### Output
 - output_filename: target SVG filename
 - random_seed: set to an integer for reproducible results, or None for randomness

 ### Cylinder (real world)
 - cylinder_radius_mm: base radius (R)
 - cylinder_height_mm: thickness in z-direction (depth)
 - canvas_margin_mm: padding around the shape

 ### Sector (cake slice)
 - sector_start_angle_degrees: clock-angle start (0° = 12 o’clock, clockwise positive)
 - sector_sweep_angle_degrees: sweep in degrees (90 = quarter circle)

 ### Organic cylinder edge (the actual “Rand”)
 - cylinder_boundary_irregularity_strength:
   - 0.0 => perfect circle boundary (exactly radius R everywhere)
   - 1.0 => boundary radius can reach up to 2R (depending on profile)
 - boundary_profile_control_point_count:
   - lower => chunkier lobes / “zipfel”
   - higher => finer waviness
 - boundary_profile_smoothing_passes:
   - higher => smoother edge (less spiky)
   - lower => more “growth” structures

 ### Camera / projection
 - camera_distance_mm: must be > cylinder_height_mm/2 for perspective math
 - perspective_strength:
   - 0.0 => effectively no perspective scaling (flat-ish)
   - >0.0 => stronger perspective scaling with z

 ### Density / performance
 - node_density_per_square_millimeter: controls node_count (higher => more nodes)
 NOTE: Edge building uses a naive nearest-neighbor approach that is O(n^2).
 High densities will get slow very quickly.

 ### Depth look
 - z_distribution_exponent: shapes z distribution (lower can exaggerate front/back)
 - node_radius_min_mm / node_radius_max_mm: node size range by depth
 - node_color_strength_background / node_color_strength_foreground: color intensity by depth
 - edge_fade_strength: 0.0 disables edge fade completely

 ## Reproducibility
 Set `random_seed` to a fixed integer (e.g., 12345).
 If random_seed is None, each run produces a different SVG.

 ## Output
 - net_cylinder.svg: generated SVG (millimeter-based sizing)

 ## Troubleshooting
 - “It’s slow”:
   - Reduce node_density_per_square_millimeter
   - Reduce required_neighbor_count
   - Consider replacing the O(n^2) neighbor search with a spatial index (k-d tree)
 - “I want no edge fade”:
   - Set edge_fade_strength = 0.0

 ## License
 Choose one:
 - MIT License (recommended for simple permissive open source)
 - Apache-2.0 (permissive + explicit patent grant)
 - GPL-3.0 (copyleft; derivatives must remain open under GPL when distributed)

 ## Project Structure (example)
 - net_graph.py
 - net_cylinder.svg (generated)
 - .gitignore
 - README.md
 - .venv/ (local virtual env; should be ignored by git)
 ```