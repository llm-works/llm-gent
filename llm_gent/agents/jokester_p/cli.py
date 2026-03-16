"""CLI tool for jokester-p agent."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_kelt.training import Factory as TrainFactory
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from llm_gent.core.memory.rating import BatchRatingService

from .storage import _validate_schema_name


if TYPE_CHECKING:
    from ...core.llm import LLMCaller


class JokesterCLI(Tool):
    """CLI commands for jokester-p agent.

    Provides agent-specific commands:
    - stats: Show joke rating statistics
    - rate: Rate unrated jokes in batches
    """

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="jokester-p", help_text="Jokester agent CLI commands")
        super().__init__(parent, config)
        self._pg: PG | None = None
        self._current_schema: str = "public"
        self._reference_schemas: list[str] = []
        self._train_factory: TrainFactory | None = None

    def configure(self) -> None:
        """Set up database connection and schema info."""
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        db_config = self.app.config.learn.db
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        self._pg = PG(self.lg, db_config)
        self._load_schema_config()
        self._load_train_factory()

    def _load_schema_config(self) -> None:
        """Load schema configuration from agent config."""
        agent_config = self._get_agent_config()
        if agent_config is None:
            return

        kelt_config = agent_config.get("kelt", {})
        schema_config = kelt_config.get("schema", {})
        self._current_schema = schema_config.get("name") or "public"

        reference_config = kelt_config.get("reference", {})
        if reference_config:
            schema_val = reference_config.get("schema")
            if isinstance(schema_val, list):
                self._reference_schemas = schema_val
            elif schema_val:
                self._reference_schemas = [schema_val]

    def _load_train_factory(self) -> None:
        """Load TrainFactory for adapter manifest lookups."""
        learn_config = getattr(self.app.config, "learn", None)
        if learn_config is None:
            return
        adapters_config = getattr(learn_config, "adapters", None)
        if adapters_config is None:
            return
        lora_config = getattr(adapters_config, "lora", None)
        if lora_config is None:
            return
        base_path = getattr(lora_config, "base_path", None)
        if base_path:
            self._train_factory = TrainFactory(self.lg, Path(base_path))

    def _get_adapter_parent(self, md5: str) -> str | None:
        """Look up parent adapter md5 from manifest."""
        if self._train_factory is None:
            return None
        manifest = self._train_factory.manifest.get_manifest(md5)
        if manifest and manifest.parent:
            return manifest.parent.md5
        return None

    def _schema_prefix(self, schema: str | None = None) -> str:
        """Get schema prefix for table names. Uses current schema if not specified."""
        s = schema or self._current_schema
        if s and s != "public":
            return f"{_validate_schema_name(s)}."
        return ""

    def _discover_schemas(self) -> list[str]:
        """Discover all schemas that have agent_jokester_training table."""
        assert self._pg is not None
        sql = text("""
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_name = 'agent_jokester_training'
            ORDER BY table_schema
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [row[0] for row in rows]

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        self._add_stats_args(subparsers)
        self._add_rate_args(subparsers)
        self._add_audit_args(subparsers)

    def _add_stats_args(self, subparsers: Any) -> None:
        """Add stats command arguments."""
        p = subparsers.add_parser("stats", help="Show joke rating statistics")
        p.add_argument(
            "--max-chars",
            type=int,
            default=140,
            help="Only count jokes under this character length (default: 140)",
        )
        p.add_argument(
            "--max-samples",
            type=int,
            help="Limit to first N samples per adapter (for fair comparison)",
        )
        p.add_argument(
            "--model",
            type=str,
            action="append",
            help="Filter by model (repeatable, case-insensitive, supports * wildcard)",
        )
        p.add_argument(
            "--schema",
            type=str,
            help="Filter to specific schema (default: show all schemas)",
        )
        p.add_argument(
            "--compact",
            action="store_true",
            help="Compact table view: models x training stage (base/sft/dpo)",
        )

    def _add_rate_args(self, subparsers: Any) -> None:
        """Add rate command arguments."""
        p = subparsers.add_parser("rate", help="Rate unrated jokes in batches")
        p.add_argument("--limit", type=int, help="Maximum jokes to rate (default: all unrated)")
        p.add_argument(
            "--batch-size", type=int, help="Jokes per API call (default: from config, typically 5)"
        )
        p.add_argument(
            "--dry-run", action="store_true", help="Show what would be rated without sending to LLM"
        )
        p.add_argument(
            "--schema", type=str, help="Rate jokes in specific schema (default: all schemas)"
        )

    def _add_audit_args(self, subparsers: Any) -> None:
        """Add audit command arguments."""
        p = subparsers.add_parser("audit", help="Audit data integrity (check for duplicates)")
        p.add_argument("--schema", type=str, help="Audit specific schema (default: all schemas)")
        p.add_argument(
            "--fix", action="store_true", help="Delete duplicate jokes (keeps first occurrence)"
        )
        p.add_argument(
            "--show-dups",
            type=int,
            default=0,
            metavar="N",
            help="Show top N duplicate jokes (default: 0)",
        )

    def run(self, **kwargs: Any) -> int:
        command = self.args.command

        if command == "stats":
            return self._cmd_stats()
        elif command == "rate":
            return self._cmd_rate()
        elif command == "audit":
            return self._cmd_audit()
        else:
            print("Usage: agent jokester-p <command>")
            print("Commands: stats, rate, audit")
            return 1

    def _cmd_stats(self) -> int:
        """Show joke rating statistics."""
        assert self._pg is not None

        max_chars = getattr(self.args, "max_chars", 140)
        max_samples = getattr(self.args, "max_samples", None)
        model = getattr(self.args, "model", None)
        schema_filter = getattr(self.args, "schema", None)
        compact = getattr(self.args, "compact", False)

        # Get schemas to query
        schemas = self._get_schemas_to_query(schema_filter)
        if not schemas:
            print("No schemas found with joke data.")
            return 0

        self._print_stats_header(max_chars, model, schema_filter, schemas, max_samples)

        if compact:
            self._print_compact_stats(schemas, max_chars, max_samples, model)
        else:
            self._print_stats_by_adapter(
                schemas, max_chars=max_chars, max_samples=max_samples, model=model
            )
            self._print_haiku_stats(max_chars=max_chars, max_samples=max_samples)

        return 0

    def _cmd_audit(self) -> int:
        """Audit data integrity - check for duplicate jokes."""
        assert self._pg is not None

        schema_filter = getattr(self.args, "schema", None)
        fix = getattr(self.args, "fix", False)
        show_dups = getattr(self.args, "show_dups", 0)

        schemas = self._get_schemas_to_query(schema_filter)
        if not schemas:
            print("No schemas found with joke data.")
            return 0

        print("\n=== Duplicate Audit ===\n")
        total_dups = 0
        has_issues = False

        for schema in schemas:
            dups = self._audit_schema(schema, show_dups)
            if dups:
                has_issues = True
                total_dups += sum(d["duplicates"] for d in dups)
                if fix:
                    self._fix_duplicates_in_schema(schema, dups)

        if has_issues:
            print(f"\n{'FIXED' if fix else 'FAILED'}: {total_dups} total duplicates found")
            return 0 if fix else 1
        else:
            print("✓ PASSED: No duplicates found")
            return 0

    def _audit_schema(self, schema: str, show_dups: int = 0) -> list[dict[str, Any]]:
        """Audit a single schema for duplicates. Returns list of duplicate info."""
        assert self._pg is not None
        context_key = self._get_context_key()

        rows = self._query_duplicate_stats(schema, context_key)
        if rows is None:
            return []

        results = self._print_audit_results(schema, rows)

        if show_dups > 0 and results:
            self._show_top_duplicates(schema, context_key, show_dups)

        return results

    def _query_duplicate_stats(self, schema: str, context_key: str) -> list[Any] | None:
        """Query duplicate statistics per model/adapter."""
        assert self._pg is not None
        prefix = self._schema_prefix(schema)
        sql = text(f"""
            SELECT
                t.base_model,
                CASE WHEN t.is_base_model THEN '(base)'
                     ELSE COALESCE(LEFT(t.adapter_info->>'md5', 4) || '..' ||
                          RIGHT(t.adapter_info->>'md5', 4), '(base)') END as adapter,
                COUNT(*) as total,
                COUNT(*) - COUNT(DISTINCT af.content) as duplicates
            FROM {prefix}agent_jokester_training t
            JOIN {prefix}atomic_facts af ON af.id = t.fact_id
            WHERE af.context_key = :context_key
            GROUP BY t.base_model,
                CASE WHEN t.is_base_model THEN '(base)'
                     ELSE COALESCE(LEFT(t.adapter_info->>'md5', 4) || '..' ||
                          RIGHT(t.adapter_info->>'md5', 4), '(base)') END
            ORDER BY t.base_model, adapter
        """)
        try:
            with self._pg.connect() as conn:
                return list(conn.execute(sql, {"context_key": context_key}).fetchall())
        except ProgrammingError:
            return None

    def _print_audit_results(self, schema: str, rows: list[Any]) -> list[dict[str, Any]]:
        """Print audit results and return list of entries with duplicates."""
        results = []
        print(f"{schema}")
        for row in rows:
            model, adapter, total, dups = row
            dup_pct = dups * 100 / total if total else 0
            if dups == 0:
                print(f"  ✅ {model} {adapter}: {total} jokes")
            else:
                status = "🔴" if dup_pct > 10 else "🟡" if dup_pct > 5 else "🟢"
                print(f"  {status} {model} {adapter}: {total} jokes, {dups} dups ({dup_pct:.1f}%)")
            if dups > 0:
                results.append(
                    {
                        "schema": schema,
                        "model": model,
                        "adapter": adapter,
                        "total": total,
                        "duplicates": dups,
                    }
                )
        return results

    def _show_top_duplicates(self, schema: str, context_key: str, limit: int) -> None:
        """Show the most common duplicate jokes."""
        assert self._pg is not None
        prefix = self._schema_prefix(schema)

        sql = text(f"""
            SELECT af.content, COUNT(*) as cnt
            FROM {prefix}atomic_facts af
            WHERE af.context_key = :context_key
            GROUP BY af.content
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT :limit
        """)

        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"context_key": context_key, "limit": limit}).fetchall()

        if rows:
            print(f"\n  Top duplicates in {schema}:")
            for content, cnt in rows:
                preview = content[:60] + "..." if len(content) > 60 else content
                print(f"    {cnt}x: {preview}")

    def _fix_duplicates_in_schema(self, schema: str, dups: list[dict[str, Any]]) -> None:
        """Delete duplicate jokes, keeping only the first occurrence."""
        assert self._pg is not None
        dup_ids = self._find_duplicate_fact_ids(schema)
        if not dup_ids:
            return
        self._delete_facts_by_ids(schema, dup_ids)
        print(f"  ✓ Deleted {len(dup_ids)} duplicate jokes from {schema}")

    def _find_duplicate_fact_ids(self, schema: str) -> list[int]:
        """Find fact IDs that are duplicates (excludes first occurrence)."""
        assert self._pg is not None
        context_key = self._get_context_key()
        prefix = self._schema_prefix(schema)
        sql = text(f"""
            WITH ranked AS (
                SELECT af.id, ROW_NUMBER() OVER (PARTITION BY af.content ORDER BY af.id) as rn
                FROM {prefix}atomic_facts af
                WHERE af.context_key = :context_key
            )
            SELECT id FROM ranked WHERE rn > 1
        """)
        with self._pg.connect() as conn:
            rows = conn.execute(sql, {"context_key": context_key}).fetchall()
        return [r[0] for r in rows]

    def _delete_facts_by_ids(self, schema: str, fact_ids: list[int]) -> None:
        """Delete facts and all dependent records by IDs."""
        assert self._pg is not None
        prefix = self._schema_prefix(schema)
        dependent_tables = [
            "agent_jokester_training",
            "agent_jokester_model_usage",
            "atomic_feedback_details",
            "atomic_solution_details",
            "embeddings",
        ]
        with self._pg.connect() as conn:
            for table in dependent_tables:
                try:
                    sql = text(f"DELETE FROM {prefix}{table} WHERE fact_id = ANY(:ids)")
                    conn.execute(sql, {"ids": fact_ids})
                except ProgrammingError:
                    pass  # Table might not exist
            conn.execute(
                text(f"DELETE FROM {prefix}atomic_facts WHERE id = ANY(:ids)"), {"ids": fact_ids}
            )
            conn.commit()

    def _get_schemas_to_query(self, schema_filter: str | None) -> list[str]:
        """Get list of schemas to query based on filter."""
        all_schemas = self._discover_schemas()
        if schema_filter:
            # Always include the requested schema (even if empty/no tables yet)
            schemas = [schema_filter]
            # Add reference schemas if configured and have data
            for ref_schema in self._reference_schemas:
                if ref_schema != schema_filter and ref_schema in all_schemas:
                    schemas.append(ref_schema)
            return schemas
        return all_schemas

    def _is_recently_active(self, last_created: Any, seconds: int = 300) -> bool:
        """Check if last_created timestamp is within the last N seconds."""
        if last_created is None:
            return False
        now = datetime.now(UTC)
        if last_created.tzinfo is None:
            last_created = last_created.replace(tzinfo=UTC)
        return bool((now - last_created) < timedelta(seconds=seconds))

    def _format_active_line(self, line: str, is_active: bool) -> str:
        """Wrap line in bold bright cyan if active."""
        if is_active:
            return f"\033[1;96m{line}\033[0m"  # Bold + bright cyan
        return line

    def _print_stats_header(
        self,
        max_chars: int,
        model_filter: str | None,
        schema_filter: str | None,
        schemas: list[str],
        max_samples: int | None = None,
    ) -> None:
        """Print stats header with filter info."""
        print("\n=== Jokester-p Stats ===\n")
        filters = [f"<{max_chars} chars"]
        if max_samples:
            filters.append(f"first {max_samples} samples")
        if model_filter:
            filters.append(f"model~'{model_filter}'")
        if schema_filter:
            filters.append(f"schema={schema_filter}")
        else:
            filters.append(f"schemas={','.join(schemas)}")
        print(f"Filters: {', '.join(filters)}")
        print()

    def _print_compact_stats(
        self,
        schemas: list[str],
        max_chars: int | None,
        max_samples: int | None,
        model: str | None,
    ) -> None:
        """Print compact table view: models x training stage."""
        all_stats = self._collect_adapter_stats(schemas, max_chars, max_samples, model)
        if not all_stats:
            print("No joke data found.")
            return

        # Group by schema (exclude reference schemas - they go in combined table)
        ref_set = set(self._reference_schemas)
        by_schema: dict[str, list[dict[str, Any]]] = {}
        for s in all_stats:
            by_schema.setdefault(s["schema"], []).append(s)

        # Print non-reference schemas first
        for schema, stats_list in by_schema.items():
            if schema not in ref_set:
                self._print_compact_table(schema, stats_list)

        # Print combined reference table at bottom
        ref_stats = self._collect_reference_stats(max_chars, max_samples)
        if ref_stats:
            self._print_reference_table(ref_stats)

    def _collect_reference_stats(
        self, max_chars: int | None, max_samples: int | None
    ) -> list[dict[str, Any]]:
        """Collect stats from all reference schemas."""
        ref_stats: list[dict[str, Any]] = []
        for ref_schema in self._reference_schemas:
            try:
                stats = self._query_haiku_stats(ref_schema, max_chars, max_samples)
                if stats["total"] > 0:
                    ref_stats.append(
                        {
                            "model": ref_schema,
                            "avg_stars": stats["avg_stars"],
                            "total": stats["total"],
                        }
                    )
            except Exception:
                continue
        return ref_stats

    def _print_compact_table(self, schema: str, stats_list: list[dict[str, Any]]) -> None:
        """Print a single compact table for a schema."""
        from rich.console import Console
        from rich.table import Table

        model_data, max_depth, active_cells = self._build_compact_model_data(stats_list)
        sorted_models = self._sort_models_by_size(model_data.keys())

        table = Table(title=schema, show_header=True, header_style="bold", title_justify="left")
        for m in sorted_models:
            table.add_column(m, justify="right")

        for depth in range(max_depth + 1):
            row = []
            for m in sorted_models:
                val = model_data[m].get(depth)
                if val:
                    cell = f"{val:.2f}"
                    if (m, depth) in active_cells:
                        cell = f"[bold bright_cyan]{cell}[/bold bright_cyan]"
                    row.append(cell)
                else:
                    row.append("--")
            table.add_row(*row)

        Console().print(table)

    def _print_reference_table(self, ref_stats: list[dict[str, Any]]) -> None:
        """Print simple reference table with schema names and averages."""
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title="haiku (reference)", show_header=True, header_style="bold", title_justify="left"
        )
        table.add_column("Schema", justify="left")
        table.add_column("Avg", justify="right")
        table.add_column("Samples", justify="right")

        for s in ref_stats:
            name = s.get("model", "unknown")
            avg = s.get("avg_stars", 0)
            total = s.get("total", 0)
            table.add_row(name, f"{avg:.2f}", str(total))

        Console().print(table)

    def _build_compact_model_data(
        self, stats_list: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[int, float | None]], int, set[tuple[str, int]]]:
        """Build model -> depth -> avg_stars mapping for compact view.

        Returns:
            Tuple of (model_data, max_depth, active_cells) where active_cells
            is a set of (size, depth) tuples that are recently active.
        """
        import re

        model_data: dict[str, dict[int, float | None]] = {}
        active_cells: set[tuple[str, int]] = set()
        max_depth = 0

        for s in stats_list:
            # Try both "model" (reference stats) and "model_name" (adapter stats)
            model_name = s.get("model") or s.get("model_name") or ""
            size_match = re.search(r"(\d+\.?\d*b)", model_name.lower())
            size = size_match.group(1) if size_match else model_name.split("-")[0]

            # Determine depth: 0=base, 1=first ft, 2=second ft, etc.
            depth = s.get("depth", 0)
            max_depth = max(max_depth, depth)

            # Track if this cell is active (recently updated - default 300s window)
            if self._is_recently_active(s.get("last_created")):
                active_cells.add((size, depth))

            model_data.setdefault(size, {})
            avg = s.get("avg_stars")
            if avg and (model_data[size].get(depth) is None or avg > model_data[size][depth]):
                model_data[size][depth] = avg

        return model_data, max_depth, active_cells

    def _sort_models_by_size(self, models: Any) -> list[str]:
        """Sort model names by numeric size."""
        import re

        def size_key(s: str) -> float:
            m = re.match(r"(\d+\.?\d*)", s)
            return float(m.group(1)) if m else 999

        return sorted(models, key=size_key)

    def _print_stats_by_adapter(
        self,
        schemas: list[str],
        max_chars: int | None = None,
        max_samples: int | None = None,
        model: str | None = None,
    ) -> None:
        """Print stats broken down by adapter, grouped by model and schema."""
        all_stats = self._collect_adapter_stats(schemas, max_chars, max_samples, model)
        if not all_stats:
            print("No joke data found.")
            return

        print("=== By Adapter ===\n")
        by_schema = self._group_stats_by_schema_model(all_stats)
        if not by_schema:
            return
        ref_set = set(self._reference_schemas)
        for schema, models in by_schema.items():
            if schema in ref_set:
                continue  # Reference schemas are shown separately
            print(schema)
            model_items = list(models.items())
            for i, (model_name, adapters) in enumerate(model_items):
                is_last_model = i == len(model_items) - 1
                model_prefix = "└── " if is_last_model else "├── "
                print(f"{model_prefix}{model_name}")
                self._print_adapter_tree(adapters, is_last_model)
        print()

    def _collect_adapter_stats(
        self, schemas: list[str], max_chars: int | None, max_samples: int | None, model: str | None
    ) -> list[dict[str, Any]]:
        """Collect and enrich adapter stats from all schemas."""
        all_stats: list[dict[str, Any]] = []
        for schema in schemas:
            schema_stats = self._get_stats_by_adapter_in_schema(
                schema, max_chars=max_chars, max_samples=max_samples, model=model
            )
            all_stats.extend(schema_stats)

        # Enrich with parent info and depth for tree display
        for stats in all_stats:
            md5 = stats.get("md5")
            if md5:
                stats["parent_md5"] = self._get_adapter_parent(md5)

        # Calculate depth for each adapter (base=0, first ft=1, etc.)
        self._calculate_adapter_depths(all_stats)
        return all_stats

    def _calculate_adapter_depths(self, all_stats: list[dict[str, Any]]) -> None:
        """Calculate depth for each adapter in the lineage."""
        md5_to_stats = {s.get("md5"): s for s in all_stats if s.get("md5")}

        for stats in all_stats:
            if stats.get("md5") is None:
                stats["depth"] = 0  # Base model
            else:
                depth = 1
                parent = stats.get("parent_md5")
                while parent and parent in md5_to_stats:
                    depth += 1
                    parent = md5_to_stats[parent].get("parent_md5")
                stats["depth"] = depth

    def _print_adapter_tree(self, adapters: list[dict[str, Any]], is_last_model: bool) -> None:
        """Print adapters as a tree based on parent-child relationships."""
        base_model_stats = [s for s in adapters if not s.get("md5")]
        adapter_stats = [s for s in adapters if s.get("md5")]
        by_md5, children = self._build_adapter_tree_maps(adapter_stats)

        # Print base model first, then adapter roots
        roots = children.get(None, [])
        total_items = len(base_model_stats) + len(roots)
        item_idx = 0

        for stats in base_model_stats:
            item_idx += 1
            self._print_base_model_stats(stats, is_last_model, item_idx == total_items)

        for root_md5 in roots:
            item_idx += 1
            self._print_adapter_node(
                root_md5, by_md5, children, is_last_model, item_idx == total_items, depth=0
            )

    def _build_adapter_tree_maps(
        self, adapter_stats: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str | None, list[str]]]:
        """Build lookup maps for adapter tree: by_md5 and children."""
        by_md5 = {s["md5"]: s for s in adapter_stats}
        children: dict[str | None, list[str]] = {None: []}
        for stats in adapter_stats:
            md5 = stats["md5"]
            parent = stats.get("parent_md5")
            # If parent not in this adapter list, treat as root
            if parent and parent not in by_md5:
                parent = None
            if parent not in children:
                children[parent] = []
            children[parent].append(md5)
        return by_md5, children

    def _print_base_model_stats(
        self, stats: dict[str, Any], is_last_model: bool, is_last: bool
    ) -> None:
        """Print base model stats (no adapter)."""
        model_indent = "    " if is_last_model else "│   "
        prefix = "└── " if is_last else "├── "
        tree_part = f"{model_indent}{prefix}(base model)"
        padded_tree = f"{tree_part:<24}"

        avg = f"{stats['avg_stars']:.2f}" if stats["avg_stars"] else "N/A"
        stars_str = self._format_star_distribution(stats.get("distribution", {}), stats["count"])
        line = f"{padded_tree} {stats['count']:5d} jokes  avg={avg}  {stars_str}"
        is_active = self._is_recently_active(stats.get("last_created"))
        print(self._format_active_line(line, is_active))

    def _print_adapter_node(
        self,
        md5: str,
        by_md5: dict[str, dict[str, Any]],
        children: dict[str | None, list[str]],
        is_last_model: bool,
        is_last: bool,
        depth: int,
    ) -> None:
        """Print a single adapter node and its children recursively."""
        stats = by_md5.get(md5)
        if not stats:
            return

        # Build indentation - align data columns by using fixed-width tree portion
        model_indent = "    " if is_last_model else "│   "
        depth_indent = "    " * depth
        prefix = "└── " if is_last else "├── "
        adapter_label = f"{md5[:2]}..{md5[-4:]}"

        # Calculate padding to align data columns (target: 24 chars for tree portion)
        tree_part = f"{model_indent}{depth_indent}{prefix}{adapter_label}"
        padded_tree = f"{tree_part:<24}"

        avg = f"{stats['avg_stars']:.2f}" if stats["avg_stars"] else "N/A"
        stars_str = self._format_star_distribution(stats.get("distribution", {}), stats["count"])
        line = f"{padded_tree} {stats['count']:5d} jokes  avg={avg}  {stars_str}"
        is_active = self._is_recently_active(stats.get("last_created"))
        print(self._format_active_line(line, is_active))

        # Print children
        child_list = children.get(md5, [])
        for j, child_md5 in enumerate(child_list):
            child_is_last = j == len(child_list) - 1
            self._print_adapter_node(
                child_md5, by_md5, children, is_last_model, child_is_last, depth + 1
            )

    def _group_stats_by_schema_model(
        self, adapter_stats: list[dict[str, Any]]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Group stats by schema -> model -> adapters."""
        by_schema: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for stats in adapter_stats:
            if stats["count"] == 0:
                continue
            schema = stats.get("schema") or "unknown"
            model_name = stats.get("model_name") or "unknown"
            if schema not in by_schema:
                by_schema[schema] = {}
            if model_name not in by_schema[schema]:
                by_schema[schema][model_name] = []
            by_schema[schema][model_name].append(stats)
        return by_schema

    def _format_star_distribution(self, dist: dict[int, int], total: int) -> str:
        """Format star distribution as: 5★:0(0.0%)  4★:18(0.2%)  3★:504(6.3%)  ..."""
        parts = []
        for stars in (5, 4, 3, 2, 1):
            count = dist.get(stars, 0)
            pct = count * 100 / total if total else 0
            parts.append(f"{stars}★:{count}({pct:.1f}%)")
        return "  ".join(parts)

    def _print_haiku_stats(
        self, max_chars: int | None = None, max_samples: int | None = None
    ) -> None:
        """Print stats for Haiku-generated jokes from reference schemas."""
        assert self._pg is not None
        if not self._reference_schemas:
            return

        chars_label = f", <{max_chars} chars" if max_chars else ""
        print(f"=== Haiku (Reference{chars_label}) ===\n")

        for ref_schema in self._reference_schemas:
            try:
                stats = self._query_haiku_stats(ref_schema, max_chars, max_samples)
            except ProgrammingError:
                continue  # Schema doesn't exist yet
            if stats["total"] == 0:
                continue
            avg_str = f"{float(stats['avg_stars']):.2f}" if stats["avg_stars"] else "N/A"
            stars_str = self._format_star_distribution(stats["dist"], stats["rated"])
            print(f"{ref_schema:20s}  {stats['total']:5d} jokes  avg={avg_str}  {stars_str}")
        print()

    def _get_latest_adapter_from_jokes(self, context_key: str) -> dict[str, Any] | None:
        """Get adapter info from most recent joke."""
        assert self._pg is not None
        prefix = self._schema_prefix()
        sql = text(f"""
            SELECT t.adapter_info
            FROM {prefix}agent_jokester_training t
            JOIN {prefix}atomic_facts af ON t.fact_id = af.id
            WHERE af.context_key = :context_key AND t.adapter_info IS NOT NULL
            ORDER BY af.id DESC
            LIMIT 1
        """)
        try:
            with self._pg.connect() as conn:
                row = conn.execute(sql, {"context_key": context_key}).fetchone()
            if not row or not row[0]:
                return None
            info = row[0]
            return {"name": info.get("actual") or info.get("requested"), "md5": info.get("md5")}
        except ProgrammingError:
            # Table doesn't exist yet (empty schema)
            return None

    def _get_stats_by_adapter_in_schema(
        self,
        schema: str,
        max_chars: int | None = None,
        max_samples: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get rating stats grouped by adapter md5 for a specific schema."""
        assert self._pg is not None
        context_key = self._get_context_key()
        prefix = self._schema_prefix(schema)
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)
        model_join, model_filter = self._build_model_filter(model, params, prefix)
        samples_filter = self._build_samples_filter(max_samples, params)

        sql = self._build_adapter_stats_sql(
            prefix, chars_join, model_join, chars_filter, model_filter, samples_filter
        )
        try:
            with self._pg.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return self._build_adapter_results(
                    conn, context_key, rows, params, max_chars, prefix, schema
                )
        except ProgrammingError:
            # Table doesn't exist yet (empty schema)
            return []

    def _build_samples_filter(
        self, max_samples: int | None, params: dict[str, Any] | None = None
    ) -> str:
        """Build WHERE clause for limiting samples per adapter."""
        if not max_samples:
            return ""
        if params is not None:
            params["max_samples"] = max_samples
        return "AND rn <= :max_samples"

    def _build_adapter_stats_sql(
        self,
        prefix: str,
        chars_join: str,
        model_join: str,
        chars_filter: str,
        model_filter: str,
        samples_filter: str = "",
    ) -> Any:
        """Build SQL for adapter stats aggregation."""
        # Use CTE with ROW_NUMBER to support max_samples filtering
        return text(f"""
            WITH ranked AS (
                SELECT af.id as fact_id,
                       t.adapter_info->>'md5' as adapter_md5,
                       t.base_model as model_name,
                       af.created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY t.base_model, t.adapter_info->>'md5'
                           ORDER BY af.created_at
                       ) as rn
                FROM {prefix}atomic_facts af
                JOIN {prefix}agent_jokester_training t ON t.fact_id = af.id
                {chars_join} {model_join}
                WHERE af.context_key = :context_key AND af.type = 'solution'
                  {chars_filter} {model_filter}
            )
            SELECT r.adapter_md5, r.model_name,
                   COUNT(*) as total, COUNT(afd.id) as rated,
                   AVG((afd.context->>'stars')::int) as avg_stars,
                   MAX(r.created_at) as last_created
            FROM ranked r
            LEFT JOIN {prefix}atomic_feedback_details afd ON r.fact_id = afd.fact_id
            WHERE 1=1 {samples_filter}
            GROUP BY r.model_name, r.adapter_md5
            ORDER BY MIN(r.fact_id)
        """)

    def _build_adapter_results(
        self,
        conn: Any,
        context_key: str,
        rows: list[Any],
        params: dict[str, Any],
        max_chars: int | None,
        prefix: str,
        schema: str,
    ) -> list[dict[str, Any]]:
        """Build result dicts with distribution for each adapter."""
        results = []
        for row in rows:
            md5, model_name, total, rated, avg_stars, last_created = row
            dist = self._query_adapter_distribution(
                conn, context_key, md5, model_name, params, max_chars, prefix
            )
            results.append(
                {
                    "md5": md5,
                    "model_name": model_name,
                    "schema": schema,
                    "count": total,
                    "rated": rated,
                    "avg_stars": avg_stars,
                    "distribution": dist,
                    "last_created": last_created,
                }
            )
        return results

    def _query_adapter_distribution(
        self,
        conn: Any,
        context_key: str,
        adapter_md5: str | None,
        base_model: str,
        base_params: dict[str, Any],
        max_chars: int | None = None,
        prefix: str = "",
    ) -> dict[int, int]:
        """Query star distribution for a specific adapter md5 and base model."""
        params = dict(base_params)
        params["base_model"] = base_model
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)

        if adapter_md5 is None:
            md5_filter = "AND t.adapter_info->>'md5' IS NULL"
        else:
            md5_filter = "AND t.adapter_info->>'md5' = :adapter_md5"
            params["adapter_md5"] = adapter_md5

        sql = text(f"""
            SELECT (afd.context->>'stars')::int as stars, COUNT(*) as cnt
            FROM {prefix}atomic_facts af
            JOIN {prefix}agent_jokester_training t ON t.fact_id = af.id
            {chars_join}
            JOIN {prefix}atomic_feedback_details afd ON af.id = afd.fact_id
            WHERE af.context_key = :context_key AND af.type = 'solution'
              AND t.base_model = :base_model
              {chars_filter} {md5_filter}
            GROUP BY (afd.context->>'stars')::int
        """)
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}

    def _get_latest_model(self, context_key: str) -> str | None:
        """Get base model name from most recent joke."""
        assert self._pg is not None
        prefix = self._schema_prefix()
        sql = text(f"""
            SELECT t.base_model
            FROM {prefix}agent_jokester_training t
            JOIN {prefix}atomic_facts af ON t.fact_id = af.id
            WHERE af.context_key = :context_key
            ORDER BY af.id DESC
            LIMIT 1
        """)
        try:
            with self._pg.connect() as conn:
                row = conn.execute(sql, {"context_key": context_key}).fetchone()
            return row[0] if row else None
        except ProgrammingError:
            # Table doesn't exist yet (empty schema)
            return None

    def _query_haiku_stats(
        self, schema: str, max_chars: int | None = None, max_samples: int | None = None
    ) -> dict[str, Any]:
        """Query stats for Haiku-generated jokes from a reference schema."""
        assert self._pg is not None
        prefix = self._schema_prefix(schema)
        context_key = self._get_context_key()
        params: dict[str, Any] = {"context_key": context_key}
        sql = self._build_haiku_stats_sql(prefix, max_chars, max_samples, params)

        with self._pg.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            total, rated, avg_stars = row[0], row[1], row[2]
            dist = self._query_haiku_distribution(conn, context_key, max_chars, max_samples, prefix)
        return {"total": total, "rated": rated, "avg_stars": avg_stars, "dist": dist}

    def _build_haiku_stats_sql(
        self, prefix: str, max_chars: int | None, max_samples: int | None, params: dict[str, Any]
    ) -> Any:
        """Build SQL for haiku stats aggregation with optional sample limiting."""
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)
        samples_limit = ""
        if max_samples:
            params["max_samples"] = max_samples
            samples_limit = "LIMIT :max_samples"

        return text(f"""
            WITH filtered AS (
                SELECT af.id as fact_id
                FROM {prefix}atomic_facts af
                JOIN {prefix}agent_jokester_model_usage u ON u.fact_id = af.id
                {chars_join}
                WHERE af.context_key = :context_key AND af.type = 'solution'
                  AND u.model_name LIKE '%haiku%' {chars_filter}
                ORDER BY af.created_at
                {samples_limit}
            )
            SELECT COUNT(*) as total, COUNT(afd.id) as rated,
                   AVG((afd.context->>'stars')::int) as avg_stars
            FROM filtered f
            LEFT JOIN {prefix}atomic_feedback_details afd ON f.fact_id = afd.fact_id
        """)

    def _query_haiku_distribution(
        self,
        conn: Any,
        context_key: str,
        max_chars: int | None,
        max_samples: int | None,
        prefix: str,
    ) -> dict[int, int]:
        """Query star distribution for Haiku jokes from reference schema."""
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)
        samples_limit = ""
        if max_samples:
            params["max_samples"] = max_samples
            samples_limit = "LIMIT :max_samples"

        sql = text(f"""
            WITH filtered AS (
                SELECT af.id as fact_id
                FROM {prefix}atomic_facts af
                JOIN {prefix}agent_jokester_model_usage u ON u.fact_id = af.id
                {chars_join}
                WHERE af.context_key = :context_key AND af.type = 'solution'
                  AND u.model_name LIKE '%haiku%' {chars_filter}
                ORDER BY af.created_at
                {samples_limit}
            )
            SELECT (afd.context->>'stars')::int as stars, COUNT(*) as cnt
            FROM filtered f
            JOIN {prefix}atomic_feedback_details afd ON f.fact_id = afd.fact_id
            GROUP BY (afd.context->>'stars')::int
        """)
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}

    def _build_chars_filter(
        self, max_chars: int | None, params: dict[str, Any] | None = None, prefix: str = ""
    ) -> tuple[str, str]:
        """Build JOIN and WHERE clause for character length filtering.

        If params dict provided, adds max_chars to it automatically.
        """
        if not max_chars:
            return "", ""
        if params is not None:
            params["max_chars"] = max_chars
        chars_join = f"JOIN {prefix}atomic_solution_details asd ON asd.fact_id = af.id"
        chars_filter = "AND length(asd.answer_text) < :max_chars"
        return chars_join, chars_filter

    def _build_model_filter(
        self,
        model: list[str] | str | None,
        params: dict[str, Any] | None = None,
        prefix: str = "",
    ) -> tuple[str, str]:
        """Build JOIN and WHERE clause for model filtering.

        Supports glob-style patterns (e.g., "qwen*14b") with case-insensitive matching.
        Can accept a single model or list of models (OR logic).
        If params dict provided, adds model_pattern_N to it automatically.
        """
        if not model:
            return "", ""

        # Normalize to list
        models = [model] if isinstance(model, str) else model

        if params is not None:
            for i, m in enumerate(models):
                # Normalize to lowercase and convert glob * to SQL %
                pattern = m.lower().replace("*", "%")
                # Wrap with % if no wildcards present (substring match)
                if "%" not in pattern:
                    pattern = f"%{pattern}%"
                params[f"model_pattern_{i}"] = pattern

        model_join = f"JOIN {prefix}agent_jokester_model_usage mu ON mu.fact_id = af.id"
        conditions = [f"LOWER(mu.model_name) LIKE :model_pattern_{i}" for i in range(len(models))]
        model_filter = f"AND ({' OR '.join(conditions)})"
        return model_join, model_filter

    def _cmd_rate(self) -> int:
        """Rate unrated jokes in batches."""
        assert self._pg is not None

        rate_config = self._load_rate_config()
        if rate_config is None:
            return 1

        prompt_template, batch_size, provider, model, max_chars = rate_config
        schemas = self._get_schemas_to_query(getattr(self.args, "schema", None))
        if not schemas:
            print("No schemas found with joke data.")
            return 0

        llm_caller = self._create_llm_caller(dry_run=getattr(self.args, "dry_run", False))
        if llm_caller is None:
            print("Error: Could not create LLM caller")
            return 1

        try:
            total = self._rate_all_schemas(
                schemas, llm_caller, prompt_template, batch_size, provider, model, max_chars
            )
            if total == 0:
                print("No unrated jokes found.")
            return 0
        finally:
            llm_caller.close()

    def _rate_all_schemas(
        self,
        schemas: list[str],
        llm_caller: LLMCaller,
        prompt_template: str,
        batch_size: int,
        provider: str,
        model: str,
        max_chars: int | None,
    ) -> int:
        """Rate jokes across all schemas. Returns total count rated."""
        total = 0
        for schema in schemas:
            total += self._rate_schema(
                schema, llm_caller, prompt_template, batch_size, provider, model, max_chars
            )
        return total

    def _rate_schema(
        self,
        schema: str,
        llm_caller: LLMCaller,
        prompt_template: str,
        batch_size: int,
        provider: str,
        model: str,
        max_chars: int | None,
    ) -> int:
        """Rate jokes in a specific schema. Returns count of rated jokes."""
        assert self._pg is not None
        service = BatchRatingService(
            lg=self.lg,
            pg=self._pg,
            llm_caller=llm_caller,
            prompt_template=prompt_template,
            context_key=self._get_context_key(),
            batch_size=batch_size,
            provider=provider,
            model=model,
            schema=schema,
            max_chars=max_chars,
        )
        unrated = service.get_unrated(self.args.limit)
        if not unrated:
            return 0

        print(f"\n=== {schema} ===")
        print(f"Unrated jokes: ~{len(unrated)}")
        return self._run_rating(service, batch_size)

    def _load_rate_config(self) -> tuple[str, int, str, str, int | None] | None:
        """Load rating config. Returns (prompt, batch_size, provider, model, max_chars) or None."""
        agent_config = self._get_agent_config()
        if agent_config is None:
            print("Error: Could not load agent config")
            return None

        rating_config = agent_config.get("rating", {})
        prompt_template = self._get_prompt_template(rating_config)
        config_batch_size = rating_config.get("batch_size", 5)
        batch_size = self.args.batch_size or config_batch_size

        # Extract provider and model from first enabled provider
        provider, model = self._get_provider_settings(rating_config)

        # Get max_chars from target config
        max_chars = agent_config.get("target", {}).get("max_chars")

        return (prompt_template, batch_size, provider, model, max_chars)

    def _run_rating(self, service: BatchRatingService, batch_size: int) -> int:
        """Execute rating and print results. Returns count of rated jokes."""
        total_rated = 0
        batch_count = 0

        for batch in service.rate_batches(limit=self.args.limit, batch_size=batch_size):
            batch_count += 1
            for r in batch:
                total_rated += 1
                clamped = max(0, min(5, r.score))
                stars = "★" * clamped + "☆" * (5 - clamped)
                joke = r.content[:90] + "..." if len(r.content) > 90 else r.content
                print(f"{r.id}  ({r.score})  {stars}  {joke}")

        if total_rated > 0:
            print(f"\n✓ Rated: {total_rated} ({batch_count} batches)")

        return total_rated

    def _get_context_key(self) -> str:
        """Get context_key from agent identity config."""
        from llm_gent.core.agent import Identity

        agent_config = self._get_agent_config()
        if agent_config is None:
            raise RuntimeError("Could not load agent config")

        # Identity can be at top level or under kelt.identity
        identity_config = agent_config.get("identity")
        if not identity_config:
            kelt_config = agent_config.get("kelt", {})
            identity_config = kelt_config.get("identity", {})

        identity = Identity.from_config(identity_config)
        return identity.context_key

    def _get_agent_config(self) -> dict[str, Any] | None:
        """Get jokester-p agent config."""
        if not hasattr(self.app, "config") or not hasattr(self.app.config, "agents"):
            return None
        config = self.app.config.agents.get("jokester-p")
        return dict(config) if config else None

    def _get_prompt_template(self, rating_config: dict[str, Any]) -> str:
        """Extract prompt template from rating config."""
        models = rating_config.get("models", {})
        atomic = models.get("atomic", {})
        solution = atomic.get("solution", {})
        return str(solution.get("prompt", "Rate this content 1-5 stars."))

    def _get_provider_settings(self, rating_config: dict[str, Any]) -> tuple[str, str]:
        """Extract provider name and model from first enabled provider.

        Returns:
            Tuple of (provider, model). Provider is used for both routing and identification.
        """
        providers = rating_config.get("providers", {})
        for name, config in providers.items():
            if not config.get("enabled", True):
                continue
            backend_config = config.get("backend", {})
            model = backend_config.get("model", "auto")
            return (name, model)
        return ("local", "auto")

    def _create_llm_caller(self, dry_run: bool = False) -> LLMCaller | None:
        """Create LLM caller from config.

        Args:
            dry_run: If True, caller will log but not send requests.
        """
        from llm_infer.client import Factory as LLMClientFactory

        from ...core.llm import LLMCaller

        llm_config = self.app.config.get("llm", {})
        try:
            router = LLMClientFactory(self.lg).from_config(llm_config)
            return LLMCaller(self.lg, router, dry_run=dry_run)
        except Exception as e:
            self.lg.error("failed to create LLM caller", extra={"exception": e})
            return None

    def _get_rating_stats(
        self,
        exclude_haiku: bool = False,
        max_chars: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Get rating statistics using lateral subquery for latest feedback per fact."""
        assert self._pg is not None
        prefix = self._schema_prefix()
        context_key = self._get_context_key()
        params: dict[str, Any] = {"context_key": context_key}

        sql = self._build_rating_stats_sql(prefix, exclude_haiku, max_chars, model, params)
        try:
            with self._pg.connect() as conn:
                result = conn.execute(sql, params).fetchone()
            return self._parse_rating_stats_result(result)
        except ProgrammingError:
            # Table doesn't exist yet (empty schema)
            return {"total": 0, "rated": 0, "unrated": 0, "avg_stars": None}

    def _build_rating_stats_sql(
        self,
        prefix: str,
        exclude_haiku: bool,
        max_chars: int | None,
        model: str | None,
        params: dict[str, Any],
    ) -> Any:
        """Build SQL for rating stats query."""
        haiku_filter = self._build_haiku_filter(exclude_haiku, prefix)
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)
        model_join, model_filter = self._build_model_filter(model, params, prefix)

        return text(f"""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE afd.id IS NOT NULL) as rated,
                   COUNT(*) FILTER (WHERE afd.id IS NULL) as unrated,
                   AVG((afd.context->>'stars')::int) as avg_stars
            FROM {prefix}atomic_facts af
            {chars_join} {model_join}
            LEFT JOIN LATERAL (
                SELECT afd2.id, afd2.context FROM {prefix}atomic_feedback_details afd2
                WHERE afd2.fact_id = af.id ORDER BY afd2.id DESC LIMIT 1
            ) afd ON true
            WHERE af.context_key = :context_key AND af.type = 'solution'
              {haiku_filter} {chars_filter} {model_filter}
        """)

    def _parse_rating_stats_result(self, result: Any) -> dict[str, Any]:
        """Parse rating stats query result into dict."""
        return {
            "total": result[0],
            "rated": result[1],
            "unrated": result[2],
            "avg_stars": float(result[3]) if result[3] else None,
        }

    def _build_haiku_filter(self, exclude_haiku: bool, prefix: str = "") -> str:
        """Build WHERE clause to exclude Haiku jokes."""
        if not exclude_haiku:
            return ""
        return f"""AND NOT EXISTS (
            SELECT 1 FROM {prefix}agent_jokester_model_usage u
            WHERE u.fact_id = af.id AND u.model_name LIKE '%haiku%'
        )"""

    def _get_rating_distribution(
        self,
        exclude_haiku: bool = False,
        max_chars: int | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get rating distribution using DISTINCT ON for latest rating per fact."""
        assert self._pg is not None
        prefix = self._schema_prefix()
        context_key = self._get_context_key()
        haiku_filter = self._build_haiku_filter(exclude_haiku, prefix)
        params: dict[str, Any] = {"context_key": context_key}
        chars_join, chars_filter = self._build_chars_filter(max_chars, params, prefix)
        model_join, model_filter = self._build_model_filter(model, params, prefix)

        sql = text(f"""
            WITH latest_ratings AS (
                SELECT DISTINCT ON (af.id) (afd.context->>'stars')::int as stars
                FROM {prefix}atomic_facts af
                JOIN {prefix}atomic_feedback_details afd ON af.id = afd.fact_id
                {chars_join}
                {model_join}
                WHERE af.context_key = :context_key AND af.type = 'solution'
                  {haiku_filter} {chars_filter} {model_filter}
                ORDER BY af.id, afd.id DESC
            )
            SELECT stars, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
            FROM latest_ratings GROUP BY stars ORDER BY stars DESC
        """)
        try:
            with self._pg.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [{"stars": r[0], "count": r[1], "pct": float(r[2])} for r in rows]
        except ProgrammingError:
            # Table doesn't exist yet (empty schema)
            return []
