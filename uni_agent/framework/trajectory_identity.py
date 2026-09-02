"""Stable identifiers shared by TransferQueue and trajectory audit artifacts."""


def trajectory_tq_key(uid: str, session_index: int, trajectory_index: int) -> str:
    """Return one trajectory's TransferQueue and audit crosswalk key."""
    return f"{uid}_{session_index}_{trajectory_index}"
