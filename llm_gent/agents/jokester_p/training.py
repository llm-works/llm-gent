"""Training data provider for jokester-p agent."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from appinfra.db.pg import PG

from .pairing import PairingService, StarFilter


if TYPE_CHECKING:
    from appinfra.log import Logger


class JokesterTrainingProvider:
    """Provides training data for jokester-p agent."""

    def __init__(self, lg: Logger, pg: PG, context_key: str) -> None:
        self._lg = lg
        self._pg = pg
        self._context_key = context_key

    def add_args(self, parser: argparse.ArgumentParser, method: str = "sft") -> None:
        """Add jokester-specific training arguments."""
        self._add_common_args(parser)
        if method == "sft":
            self._add_sft_args(parser)
        elif method == "dpo":
            self._add_dpo_args(parser)
        elif method == "prompt":
            # Prompt tuning uses same data as SFT
            self._add_sft_args(parser)

    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common training arguments (SFT and DPO)."""
        parser.add_argument(
            "--max-chars",
            type=int,
            default=140,
            help="Maximum joke length in characters (default: 140)",
        )
        parser.add_argument(
            "--filter-model",
            type=str,
            help="Filter jokes by source model (substring match)",
        )
        parser.add_argument(
            "--filter-schema",
            type=str,
            help="Schema to get training data from (default: agent's kelt.schema)",
        )
        parser.add_argument(
            "--max",
            type=int,
            dest="max_examples",
            help="Maximum examples to include",
        )

    def _add_sft_args(self, parser: argparse.ArgumentParser) -> None:
        """Add SFT-specific training arguments."""
        parser.add_argument(
            "--min-stars",
            type=int,
            default=4,
            help="Minimum star rating (default: 4)",
        )

    def _add_dpo_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO-specific training arguments."""
        self._add_dpo_pairing_args(parser)
        self._add_dpo_star_filter_args(parser)
        self._add_dpo_length_balance_args(parser)

    def _add_dpo_pairing_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO pairing count arguments."""
        parser.add_argument("--margin", type=int, help="Minimum star difference (default: 2)")
        parser.add_argument("--min-pairs", type=int, help="Minimum pairs to generate")
        parser.add_argument("--max-pairs", type=int, help="Maximum pairs to include")
        parser.add_argument("--no-reuse", action="store_true", help="1:1 pairing only")

    def _add_dpo_star_filter_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO star filter arguments."""
        parser.add_argument("--chosen-stars", type=str, help="Stars for chosen (e.g., 4 or >=4)")
        parser.add_argument(
            "--rejected-stars", type=str, help="Stars for rejected (e.g., 2 or <=2)"
        )

    def _add_dpo_length_balance_args(self, parser: argparse.ArgumentParser) -> None:
        """Add DPO length balancing arguments."""
        parser.add_argument(
            "--length-balance",
            action="store_true",
            help="Enable multi-pass length-balanced pairing to prevent length reward hacking",
        )
        parser.add_argument(
            "--length-epsilons",
            type=str,
            help="Custom length epsilons (comma-separated, e.g., '5,15,30'). Implies --length-balance",
        )

    def get_sft_examples(self, args: argparse.Namespace) -> list[dict[str, str]]:
        """Get SFT training examples from high-rated jokes."""
        filter_schema = getattr(args, "filter_schema", None)
        service = PairingService(self._lg, self._pg, self._context_key, schema=filter_schema)
        filter_model = getattr(args, "filter_model", None)
        all_jokes = service.get_rated_jokes(
            max_chars=args.max_chars,
            model=filter_model,
        )

        # Filter by minimum stars
        min_stars = getattr(args, "min_stars", 4)
        filtered = [j for j in all_jokes if j.score >= min_stars]

        # Sort by stars descending
        filtered.sort(key=lambda j: j.score, reverse=True)

        # Apply max cap
        max_examples = getattr(args, "max_examples", None)
        if max_examples is not None and len(filtered) > max_examples:
            filtered = filtered[:max_examples]

        # Convert to SFT format
        return [{"instruction": "Tell me a joke.", "output": j.content} for j in filtered]

    def get_dpo_pairs(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        """Get DPO preference pairs."""
        filter_schema = getattr(args, "filter_schema", None)
        service = PairingService(self._lg, self._pg, self._context_key, schema=filter_schema)

        # Parse length balancing options
        length_epsilons = self._parse_length_epsilons(args)

        result = service.create_pairs(
            margin=getattr(args, "margin", None) or 2,
            max_chars=args.max_chars,
            model=getattr(args, "filter_model", None),
            min_pairs=getattr(args, "min_pairs", None),
            max_pairs=getattr(args, "max_pairs", None),
            no_reuse=getattr(args, "no_reuse", False),
            chosen_stars=StarFilter.parse(getattr(args, "chosen_stars", None)),
            rejected_stars=StarFilter.parse(getattr(args, "rejected_stars", None)),
            length_epsilons=length_epsilons,
        )

        return self._format_dpo_pairs(result.pairs)

    def _parse_length_epsilons(self, args: argparse.Namespace) -> list[int | None] | None:
        """Parse length epsilon options from CLI args.

        Returns:
            List of epsilons for multi-pass pairing, or None if disabled.
        """
        # Check for custom epsilons first
        custom = getattr(args, "length_epsilons", None)
        if custom:
            epsilons: list[int | None] = []
            for part in custom.split(","):
                part = part.strip()
                if part.lower() in ("none", ""):
                    epsilons.append(None)
                else:
                    epsilons.append(int(part))
            return epsilons

        # Check for default length balance flag
        if getattr(args, "length_balance", False):
            # Default schedule: tight -> medium -> loose -> unconstrained
            return [5, 15, 30, None]

        return None

    def _format_dpo_pairs(self, pairs: list[Any]) -> list[dict[str, Any]]:
        """Convert preference pairs to DPO format."""
        # Compute summary before conversion (needs raw pair scores/lengths)
        self._last_summary = self._compute_summary(pairs)
        return [
            {
                "prompt": "Tell me a joke.",
                "chosen": p.chosen.content,
                "rejected": p.rejected.content,
            }
            for p in pairs
        ]

    def _compute_summary(self, pairs: list[Any]) -> list[str]:
        """Compute summary lines from raw pairs."""
        if not pairs:
            return []

        pair_counts = self._collect_pair_counts(pairs)
        chosen_lens = [len(p.chosen.content) for p in pairs]
        rejected_lens = [len(p.rejected.content) for p in pairs]

        lines = self._format_pair_dist(pair_counts, len(pairs))

        # Weighted average margin
        total_margin = sum((c - r) * cnt for (c, r), cnt in pair_counts.items())
        avg_margin = total_margin / len(pairs)
        lines.append(f"Avg margin: {avg_margin:.1f}★")

        c_avg = sum(chosen_lens) / len(chosen_lens)
        r_avg = sum(rejected_lens) / len(rejected_lens)
        lines.append(
            f"Avg len: chosen={c_avg:.1f}  rejected={r_avg:.1f}  diff={c_avg - r_avg:+.1f}"
        )

        return lines

    def get_dpo_summary(self) -> list[str] | None:
        """Get formatted summary lines for DPO pairs.

        Returns pair distribution (e.g., 5★ - 2★: 100) and length stats.
        Must be called after get_dpo_pairs().
        """
        return getattr(self, "_last_summary", None) or None

    def _collect_pair_counts(self, pairs: list[Any]) -> dict[tuple[int, int], int]:
        """Collect counts for each (chosen_stars, rejected_stars) combination."""
        counts: dict[tuple[int, int], int] = {}
        for p in pairs:
            key = (p.chosen.score, p.rejected.score)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _format_pair_dist(self, counts: dict[tuple[int, int], int], total: int) -> list[str]:
        """Format pair distribution as lines."""
        lines = []
        # Sort by chosen desc, then rejected desc
        for c, r in sorted(counts.keys(), key=lambda x: (-x[0], -x[1])):
            count = counts[(c, r)]
            pct = 100 * count / total if total else 0
            lines.append(f"{c}★ - {r}★: {count}({pct:.0f}%)")
        return lines

    def get_context_key(self) -> str:
        """Get context key for this agent."""
        return self._context_key

    def get_description(self, method: str, count: int) -> str:
        """Get description for the training manifest."""
        descriptions = {
            "sft": f"SFT from {count} high-rated jokes",
            "dpo": f"DPO from {count} preference pairs",
            "prompt": f"Prompt tuning from {count} high-rated jokes",
        }
        return descriptions.get(method, f"{method.upper()} training with {count} examples")
