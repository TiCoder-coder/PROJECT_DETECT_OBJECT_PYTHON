import torch

_DEBUG_CONCAT_POINTS = False
import inspect
def mask_to_box(masks: torch.Tensor):

    # debug: will print only when environment variable DEBUG_MASK_TO_BOX=1
    try:
        if os.environ.get("DEBUG_MASK_TO_BOX","0") == "1":
            print("[DEBUG mask_to_box] called. dtype:", getattr(masks,"dtype",None), "shape:", getattr(masks,"shape",None))
    except Exception:
        pass

    # ensure boolean mask
    if not torch.is_tensor(masks):
        raise TypeError("mask_to_box expects a torch.Tensor")
    if masks.dtype != torch.bool:
        # threshold at 0.5 -> boolean
        masks = masks > 0.5

    # normalize shape to [B, H, W]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks_proc = masks[:, 0]
    elif masks.ndim == 3:
        masks_proc = masks
    else:
        # try to be tolerant
        masks_proc = masks.view(masks.shape[0], masks.shape[-2], masks.shape[-1])

    B, H, W = masks_proc.shape
    device = masks_proc.device

    # build coordinate grids
    xs = torch.arange(W, device=device)
    ys = torch.arange(H, device=device)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")  # grid_xs shape [W,H] with indexing xy => (x,y) axes
    # after meshgrid with indexing="xy", shapes are (W,H) so transpose to (H,W)
    # but torch.meshgrid with 1D xs,ys and indexing="xy" returns (H,W) shapes in recent torch; keep robust:
    if grid_xs.shape != (H, W):
        grid_xs = grid_xs.t()
        grid_ys = grid_ys.t()

    grid_xs = grid_xs.unsqueeze(0).expand(B, -1, -1)
    grid_ys = grid_ys.unsqueeze(0).expand(B, -1, -1)

    # for min: where(mask, coord, large_value)
    min_xs, _ = torch.min(torch.where(masks_proc, grid_xs, W).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks_proc, grid_ys, H).flatten(-2), dim=-1)
    # for max: where(mask, coord, small_value)
    max_xs, _ = torch.max(torch.where(masks_proc, grid_xs, torch.tensor(0, device=device)).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks_proc, grid_ys, torch.tensor(0, device=device)).flatten(-2), dim=-1)

    boxes = torch.stack([min_xs, min_ys, max_xs, max_ys], dim=1).float()
    return boxes

def concat_points(existing_points, new_points=None, new_labels=None):
    """
    Robust concat_points.

    Backwards-compatible: existing_points may be None, tuple/list (coords, labels),
    or a Tensor (coords). new_points/new_labels may be None.
    RETURN: None or dict with keys:
        { "point_coords": Tensor or None, "point_labels": Tensor or None }
    """
    if _DEBUG_CONCAT_POINTS:
        print(">> concat_points called with types:", type(existing_points), type(new_points), type(new_labels))

    # helper normalizers
    def _ensure_batch_dim_pts(pts):
        if pts is None:
            return None
        if not torch.is_tensor(pts):
            raise TypeError(f"new_points must be Tensor or None, got {type(pts)}")
        if pts.ndim == 2:
            return pts.unsqueeze(0)  # [N,2] -> [1,N,2]
        if pts.ndim == 3:
            return pts
        raise ValueError(f"Unsupported new_points ndim {pts.ndim}")

    def _ensure_batch_dim_lbls(lbl):
        if lbl is None:
            return None
        if not torch.is_tensor(lbl):
            raise TypeError(f"new_labels must be Tensor or None, got {type(lbl)}")
        if lbl.ndim == 1:
            return lbl.unsqueeze(0)  # [N] -> [1,N]
        if lbl.ndim == 2:
            return lbl
        raise ValueError(f"Unsupported new_labels ndim {lbl.ndim}")

    # normalize incoming new_points/new_labels
    try:
        new_points = _ensure_batch_dim_pts(new_points)
        new_labels = _ensure_batch_dim_lbls(new_labels)
    except Exception as e:
        if _DEBUG_CONCAT_POINTS:
            print(">> concat_points normalize new_* failed:", e)
        # if invalid new inputs, return existing in a normalized dict form
        if existing_points is None:
            return None
        # convert existing to dict form below

    # nothing new to add
    if new_points is None and new_labels is None:
        # convert existing to dict for caller compatibility
        if existing_points is None:
            return None
        # fall-through to parsing existing_points

    # Parse existing_points (tuple/list/tensor/ dict/None)
    existing_coords = None
    existing_labels = None
    if existing_points is None:
        existing_coords = None
        existing_labels = None
    elif isinstance(existing_points, dict):
        existing_coords = existing_points.get("point_coords", None)
        existing_labels = existing_points.get("point_labels", None)
    elif isinstance(existing_points, (list, tuple)):
        # try unpack
        if len(existing_points) >= 1:
            existing_coords = existing_points[0]
        if len(existing_points) >= 2:
            existing_labels = existing_points[1]
    elif torch.is_tensor(existing_points):
        existing_coords = existing_points
        existing_labels = None
    else:
        # unexpected type -> ignore it, use new as base
        if _DEBUG_CONCAT_POINTS:
            print(">> concat_points: unexpected existing_points type", type(existing_points))
        existing_coords = None
        existing_labels = None

    # Normalize existing coords/labels to have batch dim (like new_* did)
    def _norm_existing_coords(ec):
        if ec is None:
            return None
        if not torch.is_tensor(ec):
            return None
        if ec.ndim == 2:
            return ec.unsqueeze(0)
        if ec.ndim == 3:
            return ec
        return None

    def _norm_existing_labels(el):
        if el is None:
            return None
        if not torch.is_tensor(el):
            return None
        if el.ndim == 1:
            return el.unsqueeze(0)
        if el.ndim == 2:
            return el
        return None

    existing_coords = _norm_existing_coords(existing_coords)
    existing_labels = _norm_existing_labels(existing_labels)

    # Concat coordinates
    if existing_coords is None and new_points is None:
        coords = None
    elif existing_coords is None:
        coords = new_points
    elif new_points is None:
        coords = existing_coords
    else:
        if existing_coords.device != new_points.device:
            new_points = new_points.to(existing_coords.device)
        coords = torch.cat([existing_coords, new_points], dim=1)

    # Concat labels
    if existing_labels is None and new_labels is None:
        labels = None
    elif existing_labels is None:
        labels = new_labels
    elif new_labels is None:
        labels = existing_labels
    else:
        if existing_labels.device != new_labels.device:
            new_labels = new_labels.to(existing_labels.device)
        labels = torch.cat([existing_labels, new_labels], dim=1)

    # if both None, return None
    if coords is None and labels is None:
        return None

    # Return dict (this is what training code expects)
    out = {"point_coords": coords, "point_labels": labels}
    if _DEBUG_CONCAT_POINTS:
        print(">> concat_points returning coords:", None if coords is None else tuple(coords.shape),
              "labels:", None if labels is None else tuple(labels.shape))
    return out
