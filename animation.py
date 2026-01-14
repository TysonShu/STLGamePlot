

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import xml.etree.ElementTree as ET

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector


# -----------------------------
# CONFIG
# -----------------------------
OSM_PATH  = "lanelet2_map_.osm"
TRAJ_PATH = "result1.txt"

# lanelet2 projector (only for loading topology; plotting uses local_x/local_y)
origin = Origin(43.47718, -80.54801)
projector = UtmProjector(origin)

# Hide some bounds if you want
SKIP_LINESTRING_IDS = {3523, 3532, 426, 3478}
MODE = "skip_boundary"  # or "skip_lanelet"

# Map style
BACKGROUND_COLOR = "#f3f1ea"
ROAD_FILL_COLOR  = "#a9b9c7"
ROAD_ALPHA       = 1.0
BOUNDARY_COLOR   = "#8fa2b3"
BOUNDARY_WIDTH   = 1.0
PADDING_METERS   = 20.0

# Your C++ offsets for local_x/local_y
X_OFFSET = -36512.0
Y_OFFSET = -13979.0

# Animation settings
FPS = 25                 # frames per second in the animation
DT  = 0.1                # seconds per frame (simulation time step for rendering)
TRAIL_SEC = 3.0          # how many seconds of "tail" to show behind each vehicle
DOT_SIZE = 40            # moving dot size
TRAIL_LW = 4.0           # tail thickness
TRAIL_ALPHA = 0.9        # tail opacity
TAIL_FADE = True         # fade older tail segments
SHOW_TIME_TEXT = True    # show current time label


# -----------------------------
# Helpers: load local_x/local_y from OSM
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


# -----------------------------
# Trajectory parsing
# -----------------------------
def parse_traj_txt(path):
    """
    Expected columns (>=7):
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


# -----------------------------
# Resample tracks to a common time grid (so animation is smooth)
# -----------------------------
def resample_track_to_grid(tr, t_grid):
    """
    Linear interpolation to a shared time grid.
    Returns dict with x,y,speed,t on the grid + a mask for valid range.
    """
    t = tr["t"]
    x = tr["x"]
    y = tr["y"]
    sp = tr["speed"]

    if t.size < 2:
        mask = np.zeros_like(t_grid, dtype=bool)
        return {"t": t_grid, "x": np.full_like(t_grid, np.nan), "y": np.full_like(t_grid, np.nan),
                "speed": np.full_like(t_grid, np.nan), "mask": mask}

    t0, t1 = t[0], t[-1]
    mask = (t_grid >= t0) & (t_grid <= t1)

    # np.interp requires increasing x; we have sorted already
    xg = np.interp(t_grid, t, x, left=np.nan, right=np.nan)
    yg = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
    spg = np.interp(t_grid, t, sp, left=np.nan, right=np.nan)

    # Ensure out-of-range are nan (np.interp left/right already nan but keep mask authoritative)
    xg[~mask] = np.nan
    yg[~mask] = np.nan
    spg[~mask] = np.nan

    return {"t": t_grid, "x": xg, "y": yg, "speed": spg, "mask": mask}


def make_segments(x, y):
    """Build 2D segments for a LineCollection."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


# -----------------------------
# Main
# -----------------------------
def main():
    # Load local plotting coordinates from OSM
    local_xy = load_local_xy_from_osm(OSM_PATH, X_OFFSET, Y_OFFSET)
    if len(local_xy) == 0:
        print("WARNING: No local_x/local_y tags found. Map plotting will fall back to lanelet2 projected coords.")

    # Load map topology
    m = load(OSM_PATH, projector)

    # Load trajectories
    tracks_raw = parse_traj_txt(TRAJ_PATH)
    if len(tracks_raw) == 0:
        raise RuntimeError("No trajectories loaded from result file.")

    # Global speed norm (shared colorbar)
    all_speeds = np.concatenate([tracks_raw[pid]["speed"] for pid in tracks_raw if tracks_raw[pid]["speed"].size > 0])
    vmin = float(np.nanmin(all_speeds)) if all_speeds.size else 0.0
    vmax = float(np.nanmax(all_speeds)) if all_speeds.size else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    # Build global time grid for animation
    t_min = min(tracks_raw[pid]["t"][0] for pid in tracks_raw if tracks_raw[pid]["t"].size > 0)
    t_max = max(tracks_raw[pid]["t"][-1] for pid in tracks_raw if tracks_raw[pid]["t"].size > 0)
    t_grid = np.arange(t_min, t_max + 1e-9, DT)

    # Resample all tracks to the shared grid
    tracks = {pid: resample_track_to_grid(tracks_raw[pid], t_grid) for pid in tracks_raw}
    pids = sorted(tracks.keys())

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Plot map (2D) once
    for ll in m.laneletLayer:
        left_id = ll.leftBound.id
        right_id = ll.rightBound.id
        uses_skipped = (left_id in SKIP_LINESTRING_IDS) or (right_id in SKIP_LINESTRING_IDS)

        if MODE == "skip_lanelet" and uses_skipped:
            continue

        # road fill
        px, py = lanelet_polygon_xy(ll, local_xy)
        ax.fill(px, py, color=ROAD_FILL_COLOR, alpha=ROAD_ALPHA, zorder=1)

        # boundaries
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

    # Set plot extents based on map (local frame)
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

    # Colorbar (speed)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("speed")

    # --- Visual strategy to show movement despite overlap:
    # 1) Keep the full spaghetti OFF (don’t plot full trajectories)
    # 2) Animate only: (a) moving dots, (b) a short tail behind each dot, colored by speed
    # This makes motion readable even when paths overlap.

    # Create one tail LineCollection + one moving dot per agent
    tail_collections = {}
    dots = {}

    # Give each agent a distinct edge/marker style for identity
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    for i, pid in enumerate(pids):
        # empty tail
        lc = mcoll.LineCollection([], linewidth=TRAIL_LW, cmap=cmap, norm=norm, alpha=TRAIL_ALPHA)
        lc.set_zorder(5)
        ax.add_collection(lc)
        tail_collections[pid] = lc

        # moving dot (face colored by speed; edge by agent identity)
        mk = marker_cycle[i % len(marker_cycle)]
        dot = ax.scatter([], [], s=DOT_SIZE, marker=mk, edgecolors="black", linewidths=0.8, zorder=6)
        dots[pid] = dot

    # Optional time text
    time_text = None
    if SHOW_TIME_TEXT:
        time_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor=BACKGROUND_COLOR, edgecolor="none", alpha=0.8),
            zorder=10
        )

    # Tail length in frames
    tail_frames = max(2, int(round(TRAIL_SEC / DT)))

    def init():
        for pid in pids:
            tail_collections[pid].set_segments([])
            tail_collections[pid].set_array(np.array([]))
            dots[pid].set_offsets(np.empty((0, 2)))
        if time_text is not None:
            time_text.set_text("")
        return list(tail_collections.values()) + list(dots.values()) + ([time_text] if time_text else [])

    def update(frame_idx):
        t_now = t_grid[frame_idx]
        f0 = max(0, frame_idx - tail_frames)

        for pid in pids:
            tr = tracks[pid]

            # Current position
            x = tr["x"][frame_idx]
            y = tr["y"][frame_idx]
            sp = tr["speed"][frame_idx]

            # If this agent is not active at this time, hide it
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(sp):
                dots[pid].set_offsets(np.empty((0, 2)))
                tail_collections[pid].set_segments([])
                tail_collections[pid].set_array(np.array([]))
                continue

            # Set dot position and color by speed
            dots[pid].set_offsets(np.array([[x, y]]))
            dots[pid].set_facecolor(cmap(norm(sp)))

            # Build tail segments (recent window)
            xw = tr["x"][f0:frame_idx+1]
            yw = tr["y"][f0:frame_idx+1]
            sw = tr["speed"][f0:frame_idx+1]

            # Drop NaNs inside the window (in case of gaps)
            ok = np.isfinite(xw) & np.isfinite(yw) & np.isfinite(sw)
            xw = xw[ok]; yw = yw[ok]; sw = sw[ok]

            if xw.size >= 2:
                segs = make_segments(xw, yw)
                tail_collections[pid].set_segments(segs)

                # One color per segment
                seg_vals = sw[:-1].copy()

                # Optional fade of older segments (alpha gradient)
                if TAIL_FADE and segs.shape[0] > 1:
                    # Create per-segment RGBA from colormap, then apply alpha ramp
                    rgba = cmap(norm(seg_vals))
                    alphas = np.linspace(0.15, TRAIL_ALPHA, num=rgba.shape[0])
                    rgba[:, 3] = alphas
                    tail_collections[pid].set_color(rgba)
                    tail_collections[pid].set_array(np.array([]))  # not used when colors are explicit
                else:
                    tail_collections[pid].set_array(seg_vals)       # use colormap

            else:
                tail_collections[pid].set_segments([])
                tail_collections[pid].set_array(np.array([]))

        if time_text is not None:
            time_text.set_text(f"t = {t_now:.2f} s")

        artists = list(tail_collections.values()) + list(dots.values())
        if time_text is not None:
            artists.append(time_text)
        return artists

    interval_ms = int(round(1000.0 / FPS))
    anim = FuncAnimation(
        fig, update, frames=len(t_grid),
        init_func=init, interval=interval_ms,
        blit=True
    )

    ax.set_title("Vehicle motion on Lanelet2 map (animated tails, colored by speed)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
