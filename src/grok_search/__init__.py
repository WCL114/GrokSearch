__all__ = ["mcp"]


def __getattr__(name: str):
    """仅在调用包级 mcp 属性时加载 FastMCP 服务。"""
    if name == "mcp":
        from .server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["mcp"]
