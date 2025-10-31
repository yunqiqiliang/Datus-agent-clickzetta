# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Universal semantic model integration service."""

from typing import Any, Dict, List, Optional, Tuple
import time
import json
from datetime import datetime

from .storage_adapter import SemanticModelStorageAdapter
from .format_converter import UniversalFormatConverter
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.storage.lancedb_conditions import eq, build_where, or_, like
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql.store import ReferenceSqlStorage

logger = get_logger(__name__)


class UniversalSemanticModelIntegration:
    """Universal semantic model integration service.

    Provides a unified interface for importing semantic models from various
    data platforms into the Datus Agent semantic model storage.
    """

    def __init__(self, connector: Any, config: Dict[str, Any]):
        """Initialize integration service.

        Args:
            connector: Database connector instance
            config: Semantic model configuration
        """
        self.connector = connector
        self.config = config
        self.storage_adapter = self._create_storage_adapter()
        self.format_converter = UniversalFormatConverter()

        # Initialize Datus Agent semantic model storage
        try:
            from datus.storage.metric.store import SemanticModelStorage
            from datus.storage.embedding_models import get_embedding_model

            # Use default storage path and embedding model if not specified
            db_path = config.get("storage_path", "data/semantic_models")
            embedding_model_name = config.get("embedding_model", "metric")
            embedding_model = get_embedding_model(embedding_model_name)

            self.semantic_storage = SemanticModelStorage(db_path, embedding_model)
            # Also prepare sibling storages for metrics and reference SQL
            self.metric_storage = MetricStorage(db_path, embedding_model)
            self.reference_sql_storage = ReferenceSqlStorage(db_path, embedding_model)
            logger.info("Initialized semantic model storage")
        except ImportError as exc:
            logger.error(f"Failed to import SemanticModelStorage: {exc}")
            self.semantic_storage = None
            self.metric_storage = None
            self.reference_sql_storage = None
        except Exception as exc:
            logger.warning(f"Failed to initialize SemanticModelStorage: {exc}")
            self.semantic_storage = None
            self.metric_storage = None
            self.reference_sql_storage = None

    def _create_storage_adapter(self) -> SemanticModelStorageAdapter:
        """Create appropriate storage adapter based on configuration.

        Returns:
            Storage adapter instance

        Raises:
            ValueError: If storage provider is not supported
        """
        provider = self.config.get("storage_provider", "volume")

        if provider == "volume":
            # ClickZetta Volume adapter
            if hasattr(self.connector, 'read_volume_file'):
                from ..adapters.clickzetta_adapter import ClickZettaVolumeAdapter
                return ClickZettaVolumeAdapter(self.connector, self.config)
            else:
                raise ValueError("Connector does not support volume operations")

        elif provider == "stage":
            # Snowflake Stage adapter (to be implemented)
            raise NotImplementedError("Snowflake stage adapter not yet implemented")

        elif provider == "s3":
            # S3 adapter (to be implemented)
            raise NotImplementedError("S3 adapter not yet implemented")

        elif provider == "local":
            # Local file adapter (to be implemented)
            raise NotImplementedError("Local file adapter not yet implemented")

        else:
            raise ValueError(f"Unsupported storage provider: {provider}")

    def is_enabled(self) -> bool:
        """Check if semantic model integration is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.config.get("enabled", False)

    def auto_import_models(self) -> Dict[str, Any]:
        """Automatically import all available semantic models.

        Returns:
            Dictionary with import results and statistics

        Raises:
            DatusException: If auto-import fails
        """
        if not self.is_enabled():
            logger.info("Semantic model integration is disabled, skipping auto-import")
            return {"status": "disabled", "imported": [], "skipped": [], "failed": []}

        if not self.semantic_storage:
            logger.error("Semantic model storage not available")
            return {"status": "error", "message": "Storage not available"}

        try:
            logger.info("Starting automatic semantic model import")
            start_time = time.time()

            # List all available models
            model_files = self.storage_adapter.list_models()
            if not model_files:
                logger.info("No semantic model files found")
                return {"status": "success", "imported": [], "skipped": [], "failed": []}

            logger.info(f"Found {len(model_files)} semantic model files")

            imported = []
            skipped = []
            failed = []

            for model_file in model_files:
                try:
                    result = self.import_model(model_file, force_update=False)
                    if result["status"] == "imported":
                        imported.append(model_file)
                    elif result["status"] == "skipped":
                        skipped.append(model_file)
                    else:
                        failed.append({"file": model_file, "error": result.get("error", "Unknown error")})

                except Exception as exc:
                    logger.warning(f"Failed to import {model_file}: {exc}")
                    failed.append({"file": model_file, "error": str(exc)})

            elapsed_time = time.time() - start_time

            logger.info(f"Auto-import completed in {elapsed_time:.2f}s: "
                       f"{len(imported)} imported, {len(skipped)} skipped, {len(failed)} failed")

            return {
                "status": "success",
                "imported": imported,
                "skipped": skipped,
                "failed": failed,
                "total_files": len(model_files),
                "elapsed_time": elapsed_time
            }

        except Exception as exc:
            logger.error(f"Auto-import failed: {exc}")
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": f"Semantic model auto-import failed: {exc}",
                    "sql": "semantic_model_auto_import"
                }
            ) from exc

    def import_model(self, model_file: str, force_update: bool = False) -> Dict[str, Any]:
        """Import a single semantic model.

        Args:
            model_file: Name of the model file to import
            force_update: Whether to force update if model already exists

        Returns:
            Dictionary with import result and details

        Raises:
            DatusException: If import fails
        """
        if not self.semantic_storage:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": "Semantic model storage not available",
                    "sql": f"import_model {model_file}"
                }
            )

        try:
            logger.info(f"Importing semantic model: {model_file}")

            # 1. Read source content
            source_content = self.storage_adapter.read_model(model_file)
            if not source_content.strip():
                return {"status": "failed", "error": "Empty model file"}

            # 2. Auto-detect format and convert
            source_format = self.format_converter.detect_format(source_content)
            logger.debug(f"Detected format: {source_format}")

            if source_format == "unknown":
                return {"status": "failed", "error": "Unknown or unsupported format"}

            metricflow_model = self.format_converter.convert_to_metricflow(source_content, source_format)

            # 3. Extract model name and check existence (support spec 'name')
            model_name = (
                metricflow_model.get("name")
                or metricflow_model.get("model")
                or model_file.replace(".yml", "").replace(".yaml", "")
            )

            if not force_update and self._model_exists(model_name):
                logger.info(f"Model {model_name} already exists, skipping")
                return {"status": "skipped", "model_name": model_name, "reason": "already_exists"}

            # 4. Save to Datus Agent storage
            self._save_model(model_name, metricflow_model, model_file)

            # 5. Extract and persist metrics and reference SQL defined in the model
            try:
                metrics_saved = self._extract_and_save_metrics(metricflow_model, model_name, force_update)
            except Exception as exc:
                logger.warning(f"Saving metrics failed for {model_name}: {exc}")
                metrics_saved = {"inserted": 0, "updated": 0}

            try:
                refsql_saved = self._extract_and_save_reference_sql(metricflow_model, model_name, model_file, force_update)
            except Exception as exc:
                logger.warning(f"Saving reference SQL failed for {model_name}: {exc}")
                refsql_saved = {"inserted": 0, "updated": 0}

            logger.info(f"Successfully imported semantic model: {model_name} (format: {source_format})")

            return {
                "status": "imported",
                "model_name": model_name,
                "source_format": source_format,
                "file": model_file,
                "force_update": force_update,
                "metrics_saved": metrics_saved,
                "reference_sql_saved": refsql_saved,
            }

        except Exception as exc:
            logger.error(f"Failed to import {model_file}: {exc}")
            return {
                "status": "failed",
                "file": model_file,
                "error": str(exc)
            }

    def _model_exists(self, model_name: str) -> bool:
        """Check whether a semantic model already exists in storage by name."""
        try:
            # Ensure underlying table is initialized
            self.semantic_storage._ensure_table_ready()
            where_clause = build_where(eq("semantic_model_name", model_name))
            return self.semantic_storage.table.count_rows(where_clause) > 0
        except Exception as exc:
            logger.warning(f"Failed to check existence for model {model_name}: {exc}")
            return False

    def _save_model(self, model_name: str, model: Dict[str, Any], source_path: str = "") -> None:
        """Persist a converted semantic model into the Datus storage.

        This maps the universal/metricflow model dictionary into the
        SemanticModelStorage schema and writes a single row.
        """
        # Prepare textual fields; convert complex structures to JSON strings
        description = model.get("description", "") or ""
        identifiers = model.get("identifiers", []) or []
        dimensions = model.get("dimensions", []) or []
        measures = model.get("measures", []) or []

        def to_json_str(val: Any) -> str:
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        # Build a simple embedding source text from dimension and identifier names
        def extract_names(items: Any) -> List[str]:
            names: List[str] = []
            if isinstance(items, dict):
                names = list(items.keys())
            elif isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and "name" in it:
                        names.append(str(it["name"]))
                    else:
                        names.append(str(it))
            return names

        embed_text_parts: List[str] = []
        embed_text_parts.extend(extract_names(identifiers))
        embed_text_parts.extend(extract_names(dimensions))
        embed_text_parts.extend(extract_names(measures))
        embed_text = " ".join([p for p in embed_text_parts if p]) or model_name

        # Compose the storage row according to SemanticModelStorage schema
        now_iso = datetime.utcnow().isoformat()
        row: Dict[str, Any] = {
            "id": model_name,
            "catalog_name": model.get("catalog", ""),
            "database_name": model.get("database", ""),
            "schema_name": model.get("schema", ""),
            "table_name": model.get("table", ""),
            "domain": model.get("domain", ""),
            "layer1": model.get("layer1", ""),
            "layer2": model.get("layer2", ""),
            "semantic_file_path": source_path or "",
            "semantic_model_name": model_name,
            "semantic_model_desc": description,
            "identifiers": to_json_str(identifiers),
            "dimensions": embed_text,  # also used as embedding source
            "measures": to_json_str(measures),
            "created_at": now_iso,
        }

        try:
            # If already exists (e.g., force update path), do an update; else insert
            if self._model_exists(model_name):
                where = eq("semantic_model_name", model_name)
                # Avoid updating the primary id; refresh textual fields only
                update_values = {k: v for k, v in row.items() if k not in ("id",)}
                self.semantic_storage.update(where=where, update_values=update_values)
            else:
                self.semantic_storage.store([row])
        except Exception as exc:
            # Re-raise to outer handler to report failure
            raise

    # --------------------
    # Metrics persistence
    # --------------------
    def _extract_and_save_metrics(
        self, model: Dict[str, Any], model_name: str, force_update: bool
    ) -> Dict[str, int]:
        """Extract metrics from the converted model and persist into MetricStorage.

        Returns summary dict: {inserted: N, updated: M}
        """
        if not self.metric_storage:
            return {"inserted": 0, "updated": 0}

        # Domain taxonomy (optional)
        domain = str(model.get("domain", "")).strip()
        layer1 = str(model.get("layer1", "")).strip()
        layer2 = str(model.get("layer2", "")).strip()

        now_iso = datetime.utcnow().isoformat()
        inserted = 0
        updated = 0

        # If forcing update, remove all existing metrics for this model to avoid stale duplicates
        if force_update:
            try:
                self.metric_storage._ensure_table_ready()
                # Remove both legacy rows (without semantic_model_name) and new rows
                # Legacy rows can be identified by id starting with "{model_name}:"
                cleanup_where = or_(
                    eq("semantic_model_name", model_name),
                    like("id", f"{model_name}:%"),
                )
                self.metric_storage.table.delete(build_where(cleanup_where))
            except Exception as exc:
                logger.warning(f"Failed to cleanup existing metrics for model {model_name}: {exc}")

        # Collect metrics from model-level and table-level
        # Each item: (name, metric_def, table_name|None)
        metric_items: List[Tuple[str, Dict[str, Any], Optional[str]]] = []

        # Model-level metrics: model["metrics"] (ClickZetta spec supports this)
        for m in model.get("metrics", []) or []:
            if isinstance(m, dict) and m.get("name"):
                metric_items.append((m["name"], m, None))

        # Table-level metrics under data_sources[*].table_metrics (converter path)
        for ds in model.get("data_sources", []) or []:
            src_table = ds.get("name") or ds.get("table") or None
            for tm in ds.get("table_metrics", []) or []:
                if isinstance(tm, dict) and tm.get("name"):
                    metric_items.append((tm["name"], tm, src_table))

        # Table-level metrics under tables[*].metrics (ClickZetta spec path)
        for tbl in model.get("tables", []) or []:
            tbl_name = tbl.get("name") or (tbl.get("base_table") or {}).get("table")
            for tm in tbl.get("metrics", []) or []:
                if isinstance(tm, dict) and tm.get("name"):
                    metric_items.append((tm["name"], tm, tbl_name))

        if not metric_items:
            return {"inserted": 0, "updated": 0}

        rows_to_insert: List[Dict[str, Any]] = []

        for name, m, tbl_name in metric_items:
            metric_id = f"{model_name}:{name}" if not tbl_name else f"{model_name}:{tbl_name}:{name}"
            llm_text_parts: List[str] = [name]
            desc = m.get("description") or ""
            # Support multiple expression keys per spec
            expr = m.get("expr") or m.get("expression") or m.get("sql") or ""
            if desc:
                llm_text_parts.append(str(desc))
            if expr:
                llm_text_parts.append(str(expr))
            llm_text = " \n".join([s for s in llm_text_parts if s])

            row = {
                "id": metric_id,
                "semantic_model_name": model_name,
                # Taxonomy: prefer meaningful grouping when not provided
                # Domain: use provided domain or fallback to model_name
                "domain": domain or model_name,
                # Layer1: table name for table-level metrics; else fallback to MODEL
                "layer1": (tbl_name or layer1 or "MODEL"),
                # Layer2: keep provided layer2 or leave empty (UI will show General)
                "layer2": layer2,
                "name": name,
                "llm_text": llm_text,
                "created_at": now_iso,
            }

            # Existence check by id
            try:
                self.metric_storage._ensure_table_ready()
                where_clause = build_where(eq("id", metric_id))
                exists = self.metric_storage.table.count_rows(where_clause) > 0
            except Exception:
                exists = False

            if exists and force_update:
                # Update textual fields and taxonomy
                try:
                    self.metric_storage.update(
                        where=eq("id", metric_id),
                        update_values={k: v for k, v in row.items() if k != "id"},
                    )
                    updated += 1
                except Exception as exc:
                    logger.warning(f"Failed to update metric {metric_id}: {exc}")
            elif not exists:
                rows_to_insert.append(row)
            else:
                # Exists and not force_update: skip
                pass

        if rows_to_insert:
            try:
                self.metric_storage.store_batch(rows_to_insert)
                inserted += len(rows_to_insert)
            except Exception as exc:
                logger.warning(f"Batch insert metrics failed: {exc}")

        return {"inserted": inserted, "updated": updated}

    # ---------------------------
    # Reference SQL persistence
    # ---------------------------
    def _extract_and_save_reference_sql(
        self, model: Dict[str, Any], model_name: str, source_path: str, force_update: bool
    ) -> Dict[str, int]:
        """Extract reference SQL (verified queries) and persist.

        Returns summary dict: {inserted: N, updated: M}
        """
        if not self.reference_sql_storage:
            return {"inserted": 0, "updated": 0}

        cz_ext = model.get("clickzetta_extensions", {}) or {}
        # Prefer spec top-level verified_queries; fallback to extensions.verified_queries
        verified_queries = (model.get("verified_queries") or cz_ext.get("verified_queries") or []) or []
        if not verified_queries:
            return {"inserted": 0, "updated": 0}

        domain = str(model.get("domain", "")).strip() or model_name
        # Group reference SQL under model unless explicitly categorized
        layer1 = str(model.get("layer1", "")).strip() or "MODEL"
        layer2 = str(model.get("layer2", "")).strip()

        # If forcing update, clean up existing entries for this model (legacy + new ids)
        if force_update:
            try:
                self.reference_sql_storage._ensure_table_ready()
                cleanup_where = like("id", f"{model_name}:%")
                self.reference_sql_storage.table.delete(build_where(cleanup_where))
            except Exception as exc:
                logger.warning(f"Failed to cleanup existing reference SQL for model {model_name}: {exc}")

        inserted = 0
        updated = 0
        rows_to_insert: List[Dict[str, Any]] = []

        for item in verified_queries:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("title")
            if not name:
                continue
            sql = item.get("sql") or item.get("query") or ""
            comment = item.get("comment") or ""
            # Build a reasonable summary from provided fields
            summary = item.get("summary") or item.get("description") or item.get("question") or name
            tags_val = item.get("tags")
            if isinstance(tags_val, list):
                tags = ",".join([str(t) for t in tags_val if t is not None])
            else:
                tags = str(tags_val) if tags_val is not None else ""

            ref_id = f"{model_name}:{name}"
            row = {
                "id": ref_id,
                "name": name,
                "sql": sql,
                "comment": comment,
                "summary": summary or name,
                "filepath": source_path or "",
                "domain": domain or model_name,
                "layer1": layer1 or "MODEL",
                "layer2": layer2,
                "tags": tags,
            }

            # Check existence by id
            try:
                self.reference_sql_storage._ensure_table_ready()
                where_clause = build_where(eq("id", ref_id))
                exists = self.reference_sql_storage.table.count_rows(where_clause) > 0
            except Exception:
                exists = False

            if exists and force_update:
                try:
                    self.reference_sql_storage.update(
                        where=eq("id", ref_id),
                        update_values={k: v for k, v in row.items() if k != "id"},
                    )
                    updated += 1
                except Exception as exc:
                    logger.warning(f"Failed to update reference SQL {ref_id}: {exc}")
            elif not exists:
                rows_to_insert.append(row)
            else:
                # Exists and not forcing update
                pass

        if rows_to_insert:
            try:
                self.reference_sql_storage.store_batch(rows_to_insert)
                inserted += len(rows_to_insert)
            except Exception as exc:
                logger.warning(f"Batch insert reference SQL failed: {exc}")

        return {"inserted": inserted, "updated": updated}

    def list_available_models(self) -> List[str]:
        """List all available semantic model files in storage.

        Returns:
            List of available model filenames

        Raises:
            DatusException: If listing fails
        """
        try:
            return self.storage_adapter.list_models()
        except Exception as exc:
            logger.error(f"Failed to list available models: {exc}")
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": f"Failed to list semantic models: {exc}",
                    "sql": "list_semantic_models"
                }
            ) from exc

    def get_model_info(self, model_file: str) -> Dict[str, Any]:
        """Get information about a specific semantic model file.

        Args:
            model_file: Name of the model file

        Returns:
            Dictionary with model information

        Raises:
            DatusException: If model info retrieval fails
        """
        try:
            # Get metadata from storage adapter
            metadata = self.storage_adapter.get_metadata(model_file)

            # Try to read content and detect format
            try:
                content = self.storage_adapter.read_model(model_file)
                format_type = self.format_converter.detect_format(content)

                # Try to extract basic info from content
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    model_name = data.get("name", "unknown") if isinstance(data, dict) else "unknown"
                    description = data.get("description", "") if isinstance(data, dict) else ""
                except:
                    model_name = "unknown"
                    description = ""

                metadata.update({
                    "format": format_type,
                    "model_name": model_name,
                    "description": description,
                    "content_length": len(content)
                })

            except Exception as content_exc:
                logger.warning(f"Failed to read content for {model_file}: {content_exc}")
                metadata.update({
                    "format": "unknown",
                    "content_error": str(content_exc)
                })

            return metadata

        except Exception as exc:
            logger.error(f"Failed to get model info for {model_file}: {exc}")
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": f"Failed to get model info: {exc}",
                    "sql": f"get_model_info {model_file}"
                }
            ) from exc

    def sync_models(self, force_update: bool = False) -> Dict[str, Any]:
        """Synchronize all semantic models (alias for auto_import_models).

        Args:
            force_update: Whether to force update existing models

        Returns:
            Dictionary with sync results
        """
        if force_update:
            # If force update, we need to import each model individually
            try:
                model_files = self.storage_adapter.list_models()
                results = {"status": "success", "imported": [], "skipped": [], "failed": []}

                for model_file in model_files:
                    result = self.import_model(model_file, force_update=True)
                    if result["status"] == "imported":
                        results["imported"].append(model_file)
                    elif result["status"] == "skipped":
                        results["skipped"].append(model_file)
                    else:
                        results["failed"].append({"file": model_file, "error": result.get("error")})

                results["total_files"] = len(model_files)
                return results

            except Exception as exc:
                logger.error(f"Force sync failed: {exc}")
                return {"status": "error", "message": str(exc)}
        else:
            return self.auto_import_models()

    def get_config(self) -> Dict[str, Any]:
        """Get current integration configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "enabled": self.is_enabled(),
            "storage_provider": self.config.get("storage_provider", "unknown"),
            "file_patterns": self.config.get("file_patterns", []),
            "auto_import": self.config.get("auto_import", False),
            "sync_on_startup": self.config.get("sync_on_startup", False),
            "provider_config": self.config.get("provider_config", {})
        }