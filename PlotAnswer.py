"""
import numpy as np
import matplotlib.pyplot as plt

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector


def line_to_xy(line_string):
    xs = [p.x for p in line_string]
    ys = [p.y for p in line_string]
    return np.array(xs), np.array(ys)


# -----------------------------
# CONFIG
# -----------------------------
OSM_PATH = "lanelet2_map_.osm"

# Origin close to your map (Waterloo)
origin = Origin(43.47718, -80.54801)
projector = UtmProjector(origin)

# LineString IDs (ways) you want to hide in the plot
SKIP_LINESTRING_IDS = {3523, 3532, 426, 3478}

# Choose behavior:
#   - "skip_lanelet": if a lanelet uses any skipped bound, skip the whole lanelet
#   - "skip_boundary": still draw lanelet, but omit only the skipped boundary line(s)
MODE = "skip_boundary"   # or "skip_lanelet"

# Optional: plot centerlines (if available)
PLOT_CENTERLINE = False


# -----------------------------
# LOAD MAP
# -----------------------------
m = load(OSM_PATH, projector)

print("Loaded:")
print("  lanelets:", len(m.laneletLayer))
print("  linestrings:", len(m.lineStringLayer))
print("  points:", len(m.pointLayer))


# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

affected = 0

for ll in m.laneletLayer:
    left_id = ll.leftBound.id
    right_id = ll.rightBound.id

    uses_skipped = (left_id in SKIP_LINESTRING_IDS) or (right_id in SKIP_LINESTRING_IDS)
    if uses_skipped:
        affected += 1

    if MODE == "skip_lanelet" and uses_skipped:
        continue

    # left boundary
    if MODE == "skip_boundary":
        if left_id not in SKIP_LINESTRING_IDS:
            xl, yl = line_to_xy(ll.leftBound)
            ax.plot(xl, yl, linewidth=1)
    else:
        xl, yl = line_to_xy(ll.leftBound)
        ax.plot(xl, yl, linewidth=1)

    # right boundary
    if MODE == "skip_boundary":
        if right_id not in SKIP_LINESTRING_IDS:
            xr, yr = line_to_xy(ll.rightBound)
            ax.plot(xr, yr, linewidth=1)
    else:
        xr, yr = line_to_xy(ll.rightBound)
        ax.plot(xr, yr, linewidth=1)

    # optional centerline
    if PLOT_CENTERLINE and hasattr(ll, "centerline") and len(ll.centerline) > 0:
        xc, yc = line_to_xy(ll.centerline)
        ax.plot(xc, yc, linestyle="--", linewidth=0.8)

print(f"Lanelets affected by skipped bounds: {affected}")

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Bird's-eye view: lanelet map (filtered)")

# Important: autoscale to what you actually plotted
ax.relim()
ax.autoscale_view()

plt.tight_layout()
plt.show()

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib as mpl

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector


def line_to_xy(line_string):
    xs, ys = [], []
    for p in line_string:
        if p.id in local_xy:
            x, y = local_xy[p.id]
        else:
            # fallback (in case a point has no local tags)
            x, y = p.x, p.y
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def lanelet_polygon_xy(ll):
    xs, ys = [], []

    for p in ll.leftBound:
        x, y = local_xy.get(p.id, (p.x, p.y))
        xs.append(x); ys.append(y)

    for p in reversed(ll.rightBound):
        x, y = local_xy.get(p.id, (p.x, p.y))
        xs.append(x); ys.append(y)

    return np.array(xs), np.array(ys)


def parse_traj_txt(path):
    """
    Expected columns (7):
      x,y,heading,steer,speed,player_id(or type),time
    Returns dict[pid] -> {x,y,speed,t}
    """
    tracks = {}
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 7:
                continue

            x = float(parts[0])
            y = float(parts[1])
            speed = float(parts[4])
            pid = int(float(parts[5]))
            t = float(parts[6])

            if pid not in tracks:
                tracks[pid] = {"x": [], "y": [], "speed": [], "t": []}
            tracks[pid]["x"].append(x)
            tracks[pid]["y"].append(y)
            tracks[pid]["speed"].append(speed)
            tracks[pid]["t"].append(t)

    # sort by time
    for pid in tracks:
        for k in tracks[pid]:
            tracks[pid][k] = np.asarray(tracks[pid][k], dtype=float)
        order = np.argsort(tracks[pid]["t"])
        for k in ["x", "y", "speed", "t"]:
            tracks[pid][k] = tracks[pid][k][order]

    return tracks


def add_speed_colored_path(ax, x, y, speed, linewidth=3.5, norm=None):
    """Line colored by speed; returns the LineCollection."""
    if len(x) < 2:
        return None
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = mcoll.LineCollection(segs, linewidth=linewidth)
    lc.set_array(speed[:-1])   # one value per segment
    if norm is not None:
        lc.set_norm(norm)      # shared scaling across all players
    ax.add_collection(lc)
    return lc


# -----------------------------
# CONFIG
# -----------------------------
OSM_PATH = "lanelet2_map_.osm"
TRAJ_PATH = "result1.txt"

origin = Origin(43.47718, -80.54801)
projector = UtmProjector(origin)

SKIP_LINESTRING_IDS = {3523, 3532, 426, 3478}
MODE = "skip_boundary"  # or "skip_lanelet"

BACKGROUND_COLOR = "#f3f1ea"
ROAD_FILL_COLOR  = "#a9b9c7"
ROAD_ALPHA = 1.0
BOUNDARY_COLOR   = "#8fa2b3"
BOUNDARY_WIDTH = 1.0
PADDING_METERS = 20.0



import xml.etree.ElementTree as ET

X_OFFSET = -36512.0
Y_OFFSET = -13979.0

# Read local_x/local_y from OSM nodes: node_id -> (x_local, y_local)
local_xy = {}
root = ET.parse(OSM_PATH).getroot()
for node in root.findall("node"):
    nid = int(node.get("id"))
    lx = ly = None
    for tag in node.findall("tag"):
        k = tag.get("k")
        if k == "local_x":
            lx = float(tag.get("v"))
        elif k == "local_y":
            ly = float(tag.get("v"))
    if lx is not None and ly is not None:
        local_xy[nid] = (lx + X_OFFSET, ly + Y_OFFSET)

# -----------------------------
# LOAD
# -----------------------------
m = load(OSM_PATH, projector)
tracks = parse_traj_txt(TRAJ_PATH)

# global color normalization for one shared colorbar
all_speeds = np.concatenate([tracks[pid]["speed"] for pid in tracks if tracks[pid]["speed"].size > 0])
vmin = float(np.nanmin(all_speeds)) if all_speeds.size else 0.0
vmax = float(np.nanmax(all_speeds)) if all_speeds.size else 1.0
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# PLOT MAP
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))
fig.set_facecolor(BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

for ll in m.laneletLayer:
    left_id = ll.leftBound.id
    right_id = ll.rightBound.id
    uses_skipped = (left_id in SKIP_LINESTRING_IDS) or (right_id in SKIP_LINESTRING_IDS)

    if MODE == "skip_lanelet" and uses_skipped:
        continue

    # road fill
    px, py = lanelet_polygon_xy(ll)
    ax.fill(px, py, color=ROAD_FILL_COLOR, alpha=ROAD_ALPHA, zorder=1)

    # boundaries
    if MODE == "skip_boundary":
        if left_id not in SKIP_LINESTRING_IDS:
            xl, yl = line_to_xy(ll.leftBound)
            ax.plot(xl, yl, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
        if right_id not in SKIP_LINESTRING_IDS:
            xr, yr = line_to_xy(ll.rightBound)
            ax.plot(xr, yr, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
    else:
        xl, yl = line_to_xy(ll.leftBound)
        xr, yr = line_to_xy(ll.rightBound)
        ax.plot(xl, yl, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
        ax.plot(xr, yr, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)

# -----------------------------
# PLOT TRAJECTORIES (speed-colored)
# -----------------------------
for pid, tr in sorted(tracks.items()):
    lc = add_speed_colored_path(ax, tr["x"], tr["y"], tr["speed"], linewidth=3.5, norm=norm)
    if lc is not None:
        lc.set_zorder(5)
        # optional: show start point
        ax.plot(tr["x"][0], tr["y"][0], marker="o", markersize=4, zorder=6)

# shared colorbar
sm = mpl.cm.ScalarMappable(norm=norm)  # uses default colormap
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("speed")

ax.set_aspect("equal", adjustable="box")
ax.relim()
ax.autoscale_view()

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin - PADDING_METERS, xmax + PADDING_METERS)
ax.set_ylim(ymin - PADDING_METERS, ymax + PADDING_METERS)

ax.set_title("Trajectories over Lanelet2 map (colored by speed)")
plt.tight_layout()
plt.show()
