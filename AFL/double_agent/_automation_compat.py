import datetime
import json
import pathlib
from typing import Any


class FallbackDriver:
    """Minimal AFL.automation Driver compatibility layer for unit tests."""

    TILED_RUN_DOCUMENTS_NODE = "run_documents"

    @staticmethod
    def unqueued(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    @staticmethod
    def queued(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def __init__(self, name: str = None, defaults: dict[str, Any] = None, overrides: dict[str, Any] = None, *args, **kwargs):
        self.name = name
        self.defaults = defaults or {}
        self.config = dict(self.defaults)
        if overrides:
            self.config.update(overrides)
        self.data = None
        self.app = None
        self.useful_links = None
        self._tiled_client = None

    def gather_defaults(self):
        return getattr(self, "defaults", {})

    def _read_tiled_config(self):
        """Read Tiled connection settings from ~/.afl/config.json."""
        config_path = pathlib.Path.home() / ".afl" / "config.json"

        if not config_path.exists():
            return {
                "status": "error",
                "message": "Config file not found at ~/.afl/config.json. Please create this file with tiled_server and tiled_api_key settings.",
            }

        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "status": "error",
                "message": f"Invalid JSON in config file: {exc}",
            }

        if not config_data:
            return {
                "status": "error",
                "message": "Config file is empty.",
            }

        datetime_key_format = "%y/%d/%m %H:%M:%S.%f"
        try:
            keys = sorted(
                config_data.keys(),
                key=lambda key: datetime.datetime.strptime(key, datetime_key_format),
                reverse=True,
            )
        except ValueError:
            keys = sorted(config_data.keys(), reverse=True)

        tiled_server = ""
        tiled_api_key = ""
        for key in keys:
            entry = config_data[key]
            if not isinstance(entry, dict):
                continue
            tiled_server = entry.get("tiled_server", "") or tiled_server
            tiled_api_key = entry.get("tiled_api_key", "") or tiled_api_key
            if tiled_server and tiled_api_key:
                break

        if not tiled_server:
            return {
                "status": "error",
                "message": "tiled_server not configured in ~/.afl/config.json. Please add a tiled_server URL to your config.",
            }

        if not tiled_api_key:
            return {
                "status": "error",
                "message": "tiled_api_key not configured in ~/.afl/config.json. Please add your Tiled API key to the config.",
            }

        return {
            "status": "success",
            "tiled_server": tiled_server,
            "tiled_api_key": tiled_api_key,
        }

    def _get_tiled_client(self):
        """Get or create a cached Tiled client."""
        if self._tiled_client is not None:
            return self._tiled_client

        config = self._read_tiled_config()
        if config["status"] == "error":
            return config

        try:
            from tiled.client import from_uri  # type: ignore
        except ModuleNotFoundError:
            return {
                "status": "error",
                "message": "Failed to connect to Tiled: tiled is not installed.",
            }

        try:
            self._tiled_client = from_uri(
                config["tiled_server"],
                api_key=config["tiled_api_key"],
                structure_clients="dask",
            )
            return self._tiled_client
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Failed to connect to Tiled: {exc}",
            }

    def _read_tiled_item(self, item: Any):
        """Read a Tiled item, disabling wide-table optimization when supported."""
        try:
            return item.read(optimize_wide_table=False)
        except TypeError as exc:
            message = str(exc)
            if "optimize_wide_table" in message or "unexpected keyword" in message:
                return item.read()
            raise

    def _normalize_run_document_entry_id(self, entry_id: Any):
        entry_id = str(entry_id or "").strip()
        prefix = f"{self.TILED_RUN_DOCUMENTS_NODE}/"
        if entry_id.startswith(prefix):
            entry_id = entry_id[len(prefix):]
        return entry_id.strip("/")

    def _walk_tiled_descendants(self, container: Any, prefix: str = ""):
        try:
            items = list(container.items())
        except Exception:
            return

        for key, item in items:
            key = str(key)
            path = f"{prefix}/{key}" if prefix else key
            yield path, item
            yield from self._walk_tiled_descendants(item, path)

    def _get_tiled_run_documents_container(self, create: bool = False):
        """Get the run_documents container, optionally creating it."""
        client = self._get_tiled_client()
        if isinstance(client, dict) and client.get("status") == "error":
            raise RuntimeError(client.get("message", "Failed to connect to Tiled."))

        try:
            return client[self.TILED_RUN_DOCUMENTS_NODE]
        except Exception:
            if not create:
                return None

        if not hasattr(client, "create_container"):
            raise RuntimeError("Tiled client does not support create_container.")

        client.create_container(
            key=self.TILED_RUN_DOCUMENTS_NODE,
            metadata={"type": self.TILED_RUN_DOCUMENTS_NODE},
        )
        return client[self.TILED_RUN_DOCUMENTS_NODE]

    def _get_tiled_run_document_item(self, entry_id: Any):
        normalized_id = self._normalize_run_document_entry_id(entry_id)
        if not normalized_id:
            raise KeyError(entry_id)

        container = self._get_tiled_run_documents_container(create=False)
        if container is None:
            raise KeyError(normalized_id)

        path_parts = [part for part in normalized_id.split("/") if part]
        if len(path_parts) > 1:
            item = container
            traversed = []
            for part in path_parts:
                item = item[part]
                traversed.append(part)
            return "/".join(traversed), item

        if normalized_id in container:
            return normalized_id, container[normalized_id]

        matches = [
            (path, item)
            for path, item in self._walk_tiled_descendants(container)
            if path.rsplit("/", 1)[-1] == normalized_id
        ]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(normalized_id)


def _fallback_plot_to_bytes(*args, **kwargs):
    raise RuntimeError("mpl_plot_to_bytes requires AFL-automation to be installed.")


def _fallback_xarray_to_bytes(*args, **kwargs):
    raise RuntimeError("xarray_to_bytes requires AFL-automation to be installed.")


try:
    from AFL.automation.APIServer.Driver import Driver  # type: ignore
    from AFL.automation.shared.utilities import mpl_plot_to_bytes, xarray_to_bytes
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("AFL.automation"):
        Driver = FallbackDriver  # type: ignore[assignment]
        mpl_plot_to_bytes = _fallback_plot_to_bytes
        xarray_to_bytes = _fallback_xarray_to_bytes
    else:
        raise

