# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def _safe_connect_server(server_name: str, server, max_retries: int = 3, connection_timeout: float = 10.0):
    """Context-managed safe MCP server connection with timeout"""
    provider = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"[MCP] Attempting to connect to server '{server_name}' (attempt {attempt + 1}/{max_retries}, timeout={connection_timeout}s)"
            )

            provider = server  # assume already created via Provider.from_process(...)
            # async context here ensures lifecycle is tracked
            # Wrap the connection with a timeout to prevent indefinite hanging
            try:
                async with asyncio.timeout(connection_timeout):
                    logger.info(f"[MCP] Entering context manager for server '{server_name}'...")
                    async with provider:
                        logger.info(f"[MCP] Server '{server_name}' connected successfully!")
                        try:
                            yield provider
                        except GeneratorExit:
                            # Handle proper cleanup on generator exit
                            logger.debug(f"[MCP] Server '{server_name}' generator being closed")
                            raise
                        return  # only yield once; exit after use
            except asyncio.TimeoutError:
                logger.error(f"[MCP] Timeout ({connection_timeout}s) while connecting to server '{server_name}'")
                raise

        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to MCP server {server_name} (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                raise
        except asyncio.CancelledError:
            # Handle cancellation during connection attempts
            logger.debug(f"MCP server {server_name} connection cancelled")
            raise
        except GeneratorExit:
            # Re-raise GeneratorExit to ensure proper cleanup
            raise
        except Exception as e:
            logger.error(f"Failed to connect MCP server {server_name} (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise

            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Handle cancellation during retry sleep
                logger.debug(f"MCP server {server_name} retry cancelled")
                raise


@asynccontextmanager
async def multiple_mcp_servers(mcp_servers: Dict[str, Any]):
    """Context manager for managing multiple MCP servers.

    Args:
        mcp_servers: Dictionary of MCP servers to manage

    Yields:
        Dictionary of connected MCP servers
    """
    connected_servers = {}
    stack = AsyncExitStack()

    try:
        logger.info(f"[MCP] Starting connection to {len(mcp_servers)} MCP server(s): {list(mcp_servers.keys())}")

        for server_name, server in mcp_servers.items():
            try:
                logger.info(f"[MCP] Connecting to server '{server_name}' (type: {type(server).__name__})...")
                cm = _safe_connect_server(server_name, server)
                connected_server = await stack.enter_async_context(cm)
                connected_servers[server_name] = connected_server
                logger.info(f"[MCP] Successfully added server '{server_name}' to connected servers")
            except Exception as e:
                logger.error(f"[MCP] Failed to start MCP server '{server_name}': {type(e).__name__}: {str(e)}")
                import traceback

                logger.debug(f"[MCP] Traceback for server '{server_name}':\n{traceback.format_exc()}")

        if not connected_servers:
            logger.warning("[MCP] No MCP servers were successfully connected")
        else:
            logger.info(f"[MCP] Total {len(connected_servers)} server(s) connected: {list(connected_servers.keys())}")

        yield connected_servers

    finally:
        logger.debug("Cleaning up all MCP servers via AsyncExitStack")
        try:
            await stack.aclose()
        except RuntimeError as e:
            if "Attempted to exit cancel scope in a different task than it was entered in" in str(e):
                # This is a known anyio issue that can be safely ignored during cleanup
                logger.debug("Suppressed cancel scope error during MCP server cleanup")
            else:
                raise
