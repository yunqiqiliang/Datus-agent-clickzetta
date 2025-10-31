# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic model file integration CLI commands for managing semantic model files.
This module provides commands for importing, synchronizing, and managing semantic model files
from various storage backends (ClickZetta Volume, Snowflake Stage, etc.).

Note: This handles semantic model FILES (YAML/JSON), not semantic VIEWs (database objects).
"""

import os
from typing import TYPE_CHECKING

from rich.box import SIMPLE_HEAD
from rich.table import Table
from rich.console import Console

from datus.utils.loggings import get_logger
from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class SemanticFileCommands:
    """Handler for semantic model file integration CLI commands."""

    def __init__(self, cli_instance: "DatusCLI"):
        """Initialize with reference to the main CLI instance."""
        self.cli = cli_instance
        self.console = Console()

    def cmd_list_semantic_files(self, args: str = ""):
        """List all available semantic model files."""
        _ = args  # Unused parameter
        try:
            integration = self._get_semantic_integration()
            if not integration:
                self.console.print("[red]Semantic model integration not available or not enabled[/red]")

                # Check if configuration is enabled but connection fails
                semantic_config = getattr(self.cli.agent_config, "external_semantic_files_config", {})
                if semantic_config.get("enabled", False):
                    self.console.print("[yellow]Check that your ClickZetta connection is properly configured[/yellow]")
                    self.console.print(
                        "[yellow]Verify that ClickZetta credentials in conf/agent.clickzetta.yml are correct[/yellow]"
                    )
                else:
                    self.console.print(
                        "[yellow]Hint: external_semantic_files.enabled is set to false in configuration[/yellow]"
                    )
                return

            # Get available models
            model_files = integration.list_available_models()

            if not model_files:
                self.console.print("[yellow]No semantic model files found[/yellow]")
                return

            # Create table
            table = Table(title="Available Semantic Model Files", box=SIMPLE_HEAD)
            table.add_column("File Name", style="cyan")
            table.add_column("Status", style="green")

            for model_file in model_files:
                # Simplified status check - just show as Available for external files
                # since these are external semantic model files from ClickZetta volume
                try:
                    info = integration.get_model_info(model_file)
                    model_format = info.get("format", "unknown")

                    if "content_error" in info:
                        # If there's an error reading the file content, show as Error
                        status = "[red]Error[/red]"
                    elif model_format and model_format != "unknown":
                        # If we can determine the format, it's available
                        status = "[green]Available[/green]"
                    else:
                        # Unknown format or couldn't parse
                        status = "[yellow]Unknown format[/yellow]"

                except Exception as e:
                    status = "[red]Error[/red]"

                table.add_row(model_file, status)

            self.console.print(table)

        except Exception as e:
            logger.error(f"Failed to list semantic models: {e}")
            self.console.print(f"[red]Error listing semantic models: {e}[/red]")

    def cmd_import_semantic_file(self, args: str = ""):
        """Import a specific semantic model file."""
        if not args.strip():
            self.console.print("[red]Usage: .import_semantic_file <model_file> [--force][/red]")
            return

        parts = args.strip().split()
        model_file = parts[0]
        force_update = "--force" in parts

        try:
            integration = self._get_semantic_integration()
            if not integration:
                self.console.print("[red]Semantic model integration not available or not enabled[/red]")
                return

            self.console.print(f"[cyan]Importing semantic model: {model_file}[/cyan]")
            result = integration.import_model(model_file, force_update=force_update)

            if result["status"] == "imported":
                self.console.print(
                    f"[green]✓ Successfully imported model: {result['model_name']} (format: {result['source_format']})[/green]"
                )
            elif result["status"] == "skipped":
                self.console.print(
                    f"[yellow]⚠ Model already exists: {result['model_name']} (use --force to update)[/yellow]"
                )
            else:
                self.console.print(f"[red]✗ Failed to import model: {result.get('error', 'Unknown error')}[/red]")

        except Exception as e:
            logger.error(f"Failed to import semantic model {model_file}: {e}")
            self.console.print(f"[red]Error importing semantic model: {e}[/red]")

    def cmd_sync_semantic_files(self, args: str = ""):
        """Synchronize (auto-import) all available semantic model files."""
        force_update = "--force" in args.strip()

        try:
            integration = self._get_semantic_integration()
            if not integration:
                self.console.print("[red]Semantic model integration not available or not enabled[/red]")
                return

            self.console.print("[cyan]Synchronizing semantic models...[/cyan]")
            result = integration.sync_models(force_update=force_update)

            if result["status"] == "success":
                imported = result.get("imported", [])
                skipped = result.get("skipped", [])
                failed = result.get("failed", [])
                total = result.get("total_files", 0)
                elapsed = result.get("elapsed_time", 0)

                self.console.print(f"[green]✓ Sync completed in {elapsed:.2f}s[/green]")
                self.console.print(f"  • Total files: {total}")
                self.console.print(f"  • Imported: {len(imported)}")
                self.console.print(f"  • Skipped: {len(skipped)}")
                self.console.print(f"  • Failed: {len(failed)}")

                if failed:
                    self.console.print("[red]Failed files:[/red]")
                    for error in failed:
                        self.console.print(f"  • {error['file']}: {error['error']}")

            else:
                error_msg = result.get("message", "Unknown error")
                self.console.print(f"[red]✗ Sync failed: {error_msg}[/red]")

        except Exception as e:
            logger.error(f"Failed to sync semantic models: {e}")
            self.console.print(f"[red]Error syncing semantic models: {e}[/red]")

    def cmd_semantic_file_info(self, args: str = ""):
        """Show information about a specific semantic model file."""
        if not args.strip():
            self.console.print("[red]Usage: .semantic_file_info <model_file>[/red]")
            return

        model_file = args.strip()

        try:
            integration = self._get_semantic_integration()
            if not integration:
                self.console.print("[red]Semantic model integration not available or not enabled[/red]")
                return

            info = integration.get_model_info(model_file)

            # Create info table
            table = Table(title=f"Semantic Model File Info: {model_file}", box=SIMPLE_HEAD)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("File Name", model_file)
            table.add_row("Model Name", info.get("model_name", "unknown"))
            table.add_row("Format", info.get("format", "unknown"))
            table.add_row("Description", info.get("description", "N/A"))
            table.add_row("Content Length", str(info.get("content_length", 0)))

            if "size" in info:
                table.add_row("File Size", f"{info['size']} bytes")
            if "modified_time" in info:
                table.add_row("Modified", info["modified_time"])

            self.console.print(table)

        except Exception as e:
            logger.error(f"Failed to get semantic model info for {model_file}: {e}")
            self.console.print(f"[red]Error getting model info: {e}[/red]")

    def cmd_semantic_file_config(self, args: str = ""):
        """Show current semantic model file integration configuration."""
        _ = args  # Unused parameter
        try:
            integration = self._get_semantic_integration()
            if not integration:
                self.console.print("[red]Semantic model integration not available or not enabled[/red]")
                return

            config = integration.get_config()

            # Create config table
            table = Table(title="Semantic Model File Integration Configuration", box=SIMPLE_HEAD)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Enabled", "✓" if config.get("enabled") else "✗")
            table.add_row("Storage Provider", config.get("storage_provider", "unknown"))
            table.add_row("Auto Import", "✓" if config.get("auto_import") else "✗")
            table.add_row("Sync on Startup", "✓" if config.get("sync_on_startup") else "✗")
            table.add_row("File Patterns", ", ".join(config.get("file_patterns", [])))

            provider_config = config.get("provider_config", {})
            if provider_config:
                table.add_row("", "")  # Separator
                table.add_row("[bold]Provider Config[/bold]", "")
                for key, value in provider_config.items():
                    table.add_row(f"  {key}", str(value))

            self.console.print(table)

        except Exception as e:
            logger.error(f"Failed to get semantic model config: {e}")
            self.console.print(f"[red]Error getting config: {e}[/red]")

    def _get_semantic_integration(self):
        """Get the semantic model integration from the current connector."""
        try:
            # Get current connector
            namespace = self.cli.agent_config.current_namespace
            connector = self.cli.db_manager.get_conn(namespace)

            # Check if it's a ClickZetta connector with semantic integration
            if isinstance(connector, ClickzettaConnector):
                return connector.semantic_integration
            else:
                logger.warning(
                    f"Semantic model integration only available for ClickZetta connectors, got: {type(connector)}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to get semantic integration: {e}")
            # Check if it's missing ClickZetta environment variables
            clickzetta_vars = [
                "CLICKZETTA_SERVICE",
                "CLICKZETTA_USERNAME",
                "CLICKZETTA_PASSWORD",
                "CLICKZETTA_INSTANCE",
                "CLICKZETTA_WORKSPACE",
                "CLICKZETTA_SCHEMA",
                "CLICKZETTA_VCLUSTER",
            ]
            missing_vars = [var for var in clickzetta_vars if not os.getenv(var)]
            if missing_vars:
                logger.error(f"Missing ClickZetta environment variables: {', '.join(missing_vars)}")
                logger.info("Please set the required ClickZetta environment variables to use semantic file integration")
            return None
