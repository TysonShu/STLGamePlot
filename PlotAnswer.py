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


'''
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
'''

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET
from PIL import Image

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector


# -----------------------------
# CONFIG
# -----------------------------
OSM_PATH  = "lanelet2_map_.osm"
TRAJ_PATH = "WithRule.txt"

origin = Origin(43.47718, -80.54801)
projector = UtmProjector(origin)

SKIP_LINESTRING_IDS = {3523, 3532, 426, 3478}
MODE = "skip_boundary"  # or "skip_lanelet"

BACKGROUND_COLOR = "#f3f1ea"
ROAD_FILL_COLOR  = "#a9b9c7"
ROAD_ALPHA       = 1.0
BOUNDARY_COLOR   = "#8fa2b3"
BOUNDARY_WIDTH   = 1.0
PADDING_METERS   = 20.0


# local_x/local_y offsets (same as your C++)
X_OFFSET = -36512.0
Y_OFFSET = -13979.0

# Snapshot settings
T_START = 9.0
T_END   = 13.0
DT_SNAP = 0.5
SNAP_TIMES = np.arange(T_START, T_END + 1e-9, DT_SNAP)

# Car icons (provided)
CAR_ICON_PATHS = [
    "blue_car.png",
    "resized_orange_car.png",
    "resized_purple_car.png",
    "resized_white_car.png",
    "ped.png"
]

PED_ICON_BASE_HEADING = -np.pi / 2

# Icon size on plot (tune this)
ICON_ZOOM = 0.14

# Assumption: the car icon points NORTH by default, which corresponds to heading = pi/2
ICON_BASE_HEADING = np.pi / 2


# -----------------------------
# MAP HELPERS (local coords)
# -----------------------------
def load_local_xy_from_osm(osm_path: str, x_off: float, y_off: float):
    local_xy = {}
    root = ET.parse(osm_path).getroot()
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
            local_xy[nid] = (lx + x_off, ly + y_off)
    return local_xy


def line_to_xy(line_string, local_xy):
    xs, ys = [], []
    for p in line_string:
        x, y = local_xy.get(p.id, (p.x, p.y))
        xs.append(x); ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float)


def lanelet_polygon_xy(ll, local_xy):
    xs, ys = [], []
    for p in ll.leftBound:
        x, y = local_xy.get(p.id, (p.x, p.y))
        xs.append(x); ys.append(y)
    for p in reversed(ll.rightBound):
        x, y = local_xy.get(p.id, (p.x, p.y))
        xs.append(x); ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float)

def slice_traj_in_window(tr, t0, t1):
    t = tr["t"]
    if len(t) < 2:
        return None
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return None
    return tr["x"][mask], tr["y"][mask], tr["heading"][mask], t[mask]

def add_time_markers(ax, tr, t0, t1, every_sec=1.0,
                     marker="o", size=20,
                     color="black", zorder=7):
    """
    Plot markers at fixed time intervals along a trajectory.
    """
    times = np.arange(t0, t1 + 1e-9, every_sec)
    xs, ys = [], []

    for tq in times:
        st = interp_state_at_time(tr, tq)
        if st is None:
            continue
        x, y, _ = st
        xs.append(x)
        ys.append(y)

    if xs:
        ax.scatter(xs, ys, s=size, marker=marker,
                   color=color, edgecolors="white",
                   linewidths=0.6, zorder=zorder)

# -----------------------------
# TRAJ PARSE (include heading)
# -----------------------------
def parse_traj_txt_with_heading(path):
    """
    Expected columns (>=7):
      0:x, 1:y, 2:heading, 3:steer, 4:speed, 5:pid, 6:time
    Returns dict[pid] -> {x,y,heading,t}
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
            heading = float(parts[2])
            pid = int(float(parts[5]))
            t = float(parts[6])

            if pid not in tracks:
                tracks[pid] = {"x": [], "y": [], "heading": [], "t": []}
            tracks[pid]["x"].append(x)
            tracks[pid]["y"].append(y)
            tracks[pid]["heading"].append(heading)
            tracks[pid]["t"].append(t)

    # sort by time
    for pid in tracks:
        for k in tracks[pid]:
            tracks[pid][k] = np.asarray(tracks[pid][k], dtype=float)
        order = np.argsort(tracks[pid]["t"])
        for k in ["x", "y", "heading", "t"]:
            tracks[pid][k] = tracks[pid][k][order]

    return tracks


def interp_state_at_time(tr, t_query):
    """
    Interpolate (x,y,heading) at time t_query.
    Returns None if out of range.
    """
    t = tr["t"]
    if t_query < t[0] or t_query > t[-1] or len(t) < 2:
        return None

    x = np.interp(t_query, t, tr["x"])
    y = np.interp(t_query, t, tr["y"])

    # heading interpolation with wrap handling
    h = tr["heading"]
    h_unwrap = np.unwrap(h)
    hq_unwrap = np.interp(t_query, t, h_unwrap)
    hq = (hq_unwrap + np.pi) % (2 * np.pi) - np.pi  # back to [-pi, pi]
    return x, y, hq


# -----------------------------
# ICON DRAWING
# -----------------------------
def add_rotated_icon(ax, img_pil, x, y, heading_rad, zoom=0.2):
    """
    Rotate car image based on heading.
    Icon default is assumed to face ICON_BASE_HEADING (pi/2).
    """
    # rotation relative to icon base direction
    rot_rad = heading_rad - ICON_BASE_HEADING
    rot_deg = np.degrees(rot_rad)

    # If rotation looks flipped, uncomment this:
    # rot_deg = -rot_deg

    img_rot = img_pil.rotate(rot_deg, resample=Image.Resampling.BICUBIC, expand=True)
    im = OffsetImage(np.asarray(img_rot), zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
    ax.add_artist(ab)


# -----------------------------
# MAIN
# -----------------------------
def main():
    # load local map coords
    local_xy = load_local_xy_from_osm(OSM_PATH, X_OFFSET, Y_OFFSET)
    if len(local_xy) == 0:
        print("WARNING: no local_x/local_y tags found; falling back to projected coords for map points.")

    # load map topology
    m = load(OSM_PATH, projector)

    # load tracks (with heading)
    tracks = parse_traj_txt_with_heading(TRAJ_PATH)
    if len(tracks) == 0:
        raise RuntimeError("No trajectories loaded.")

    pids = sorted(tracks.keys())

    # assign a fixed color per player
    cmap = mpl.cm.get_cmap("tab10")
    pid_color = {pid: cmap(i % 10) for i, pid in enumerate(pids)}

    # load icons (cycle if more vehicles than icons)
    icons = [Image.open(p).convert("RGBA") for p in CAR_ICON_PATHS]

    # set up figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # plot map
    for ll in m.laneletLayer:
        left_id = ll.leftBound.id
        right_id = ll.rightBound.id
        uses_skipped = (left_id in SKIP_LINESTRING_IDS) or (right_id in SKIP_LINESTRING_IDS)
        if MODE == "skip_lanelet" and uses_skipped:
            continue

        px, py = lanelet_polygon_xy(ll, local_xy)
        ax.fill(px, py, color=ROAD_FILL_COLOR, alpha=ROAD_ALPHA, zorder=1)

        if MODE == "skip_boundary":
            if left_id not in SKIP_LINESTRING_IDS:
                xl, yl = line_to_xy(ll.leftBound, local_xy)
                ax.plot(xl, yl, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
            if right_id not in SKIP_LINESTRING_IDS:
                xr, yr = line_to_xy(ll.rightBound, local_xy)
                ax.plot(xr, yr, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
        else:
            xl, yl = line_to_xy(ll.leftBound, local_xy)
            xr, yr = line_to_xy(ll.rightBound, local_xy)
            ax.plot(xl, yl, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)
            ax.plot(xr, yr, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2)

    # set bounds based on map points
    mx, my = [], []
    for ll in m.laneletLayer:
        xl, yl = line_to_xy(ll.leftBound, local_xy)
        xr, yr = line_to_xy(ll.rightBound, local_xy)
        mx.extend(xl); mx.extend(xr)
        my.extend(yl); my.extend(yr)
    mx = np.asarray(mx, float)
    my = np.asarray(my, float)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mx.min() - PADDING_METERS, mx.max() + PADDING_METERS)
    ax.set_ylim(my.min() - PADDING_METERS, my.max() + PADDING_METERS)

    # --- draw snapshots
    # For readability: slightly fade earlier snapshots and emphasize later ones
    alphas = np.linspace(0.35, 1.0, len(SNAP_TIMES))

    # --- draw trajectories within the time window [T_START, T_END]
    for j, pid in enumerate(pids):
        tr = tracks[pid]
        sliced = slice_traj_in_window(tr, T_START, T_END)
        if sliced is None:
            continue
        xw, yw, hw, tw = sliced

        ax.plot(xw, yw, color=pid_color[pid], linewidth=2.5, alpha=0.9, zorder=6)
        ax.plot(
            xw, yw,
            color=pid_color[pid],
            linewidth=2.5,
            alpha=0.9,
            zorder=6
        )

        add_time_markers(
            ax, tr,
            T_START, T_END,
            every_sec=1.0,
            size=20,
            color=pid_color[pid]   # ← per-player color
        )

    # --- draw vehicle icons ONLY at the first frame (t = T_START)
    t_first = T_START
    ped_pid = pids[-1]   # last player is pedestrian
    for j, pid in enumerate(pids):
        st = interp_state_at_time(tracks[pid], t_first)
        if st is None:
            continue
        x, y, heading = st
        icon = icons[j % len(icons)]
        # add_rotated_icon(ax, icon, x, y, heading, zoom=ICON_ZOOM)
        
        
        # 🚶 pedestrian: force heading = -pi/2
        if pid == ped_pid:
            add_rotated_icon(
                ax, icon,
                x, y,
                heading_rad=np.pi / 2,
                zoom=ICON_ZOOM * 0.35   # optional: make ped slightly smaller
            )
        else:
            add_rotated_icon(
                ax, icon,
                x, y,
                heading_rad=heading,
                zoom=ICON_ZOOM * 0.8
            )        

    # ax.set_title("Vehicle snapshots (t = 0.0–1.5 s, every 0.5 s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()

    fname = f"With_Rule_trajectory_snapshot_t{T_START:.1f}_to_{T_END:.1f}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fname}")

    plt.show()


if __name__ == "__main__":
    main()


