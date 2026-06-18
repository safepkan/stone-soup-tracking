"""Caller metadata helpers for TOMHT-owned track metadata boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

TOMHT_OWNED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "track_id",
        "internal_track_id",
        "public_track_id",
        "node_id",
        "age",
        "hits",
        "missed_count",
        "last_det_key",
        "last_det_hit",
        "root_source",
        "birth_scan_index",
        "existence_log_odds",
        "existence_probability",
        "lifecycle_state",
        "publication_state",
    }
)


def validate_caller_metadata_keys(
    keys: Iterable[str],
    *,
    source_name: str,
) -> tuple[str, ...]:
    """Return validated caller metadata keys, rejecting TOMHT-owned names."""
    if isinstance(keys, str):
        raise ValueError(
            f"{source_name} must be an iterable of metadata key strings, "
            "not a string."
        )

    key_tuple = tuple(keys)
    non_string_keys = [key for key in key_tuple if not isinstance(key, str)]
    if non_string_keys:
        non_string_keys_str = ", ".join(repr(key) for key in non_string_keys)
        raise TypeError(
            f"{source_name} metadata keys must be strings; got: "
            f"{non_string_keys_str}."
        )

    duplicate_keys = sorted({key for key in key_tuple if key_tuple.count(key) > 1})
    if duplicate_keys:
        duplicate_keys_str = ", ".join(repr(key) for key in duplicate_keys)
        raise ValueError(
            f"{source_name} metadata keys must be unique; got duplicate key(s): "
            f"{duplicate_keys_str}."
        )

    protected_keys = sorted(set(key_tuple).intersection(TOMHT_OWNED_METADATA_KEYS))
    if protected_keys:
        protected_keys_str = ", ".join(repr(key) for key in protected_keys)
        raise ValueError(
            f"{source_name} cannot include TOMHT-owned metadata key(s): "
            f"{protected_keys_str}."
        )
    return key_tuple


def caller_metadata_from_mapping(
    metadata: Mapping[str, object],
    *,
    keys: Iterable[str],
) -> dict[str, object]:
    """Copy whitelisted caller metadata keys from a source metadata mapping."""
    key_tuple = validate_caller_metadata_keys(
        keys,
        source_name="caller metadata whitelist",
    )
    return {key: metadata[key] for key in key_tuple if key in metadata}
