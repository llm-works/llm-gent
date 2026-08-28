# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Rate tool - rate agent content for training."""

import argparse
import json
from datetime import datetime
from typing import Any, cast

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_kelt.core import Database

from ...core.memory.rating import (
    AtomicFactsBackend,
    ConfigParser,
    ProviderType,
    Request,
    Service,
)


class RateTool(Tool):
    """Rate unrated content from agents for training."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="rate", help_text="Rate unrated content from agents for training")
        super().__init__(parent, config)
        self._db: Database | None = None
        self._backend: AtomicFactsBackend | None = None

    def configure(self) -> None:
        """Set up database connection and backend."""
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        db_config = self.app.config.learn.db
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        # Create database connection and backend
        pg = PG(self.lg, db_config)
        self._db = Database(self.lg, pg)
        self._backend = AtomicFactsBackend(self.lg, self._db)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "agent_name",
            help="Name of the agent to rate content from",
        )
        parser.add_argument(
            "--cont",
            action="store_true",
            help="Continuous mode - rate all unrated items",
        )
        parser.add_argument(
            "--category",
            help="Filter by category (e.g., 'joke')",
        )
        parser.add_argument(
            "--type",
            help="Filter by fact type (e.g., 'solution')",
        )
        parser.add_argument(
            "--auto",
            action="store_true",
            help="Automated mode - use LLM for rating (requires RatingTrait configuration)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Maximum number of items to rate in auto mode (0 = unlimited, default: 10)",
        )

    def run(self, **kwargs: Any) -> int:
        agent_name = self.args.agent_name
        context_key = self._resolve_context_key(agent_name)

        self.lg.info(
            "starting rating session",
            extra={"agent": agent_name, "context_key": context_key},
        )

        if self.args.auto:
            return self._auto_mode(context_key)
        if self.args.cont:
            return self._continuous_mode(context_key)
        return self._single_rating(context_key)

    def _resolve_context_key(self, agent_name: str) -> str:
        """Resolve context_key from agent config.

        Args:
            agent_name: Agent name from command line

        Returns:
            Resolved context key from config, or falls back to agent_name
        """
        config = self._load_agent_config(agent_name)
        if config is None:
            return agent_name

        context_key = self._extract_context_key(config, agent_name)
        return context_key if context_key else agent_name

    def _load_agent_config(self, agent_name: str) -> DotDict | None:
        """Load agent config from app config."""
        if not hasattr(self.app.config, "agents"):
            self.lg.warning(
                "no agents in config, using agent name as context_key",
                extra={"agent": agent_name},
            )
            return None

        agents = self.app.config.agents
        if agent_name not in agents:
            self.lg.warning(
                "agent config not found, using agent name as context_key",
                extra={"agent": agent_name, "available_agents": list(agents.keys())},
            )
            return None

        return cast(DotDict, agents[agent_name])

    def _extract_context_key(self, config: DotDict, agent_name: str) -> str | None:
        """Extract context_key from agent config."""
        if config and "identity" in config:
            identity = config["identity"]
            if isinstance(identity, dict) and "context_key" in identity:
                context_key = identity["context_key"]
                if isinstance(context_key, str):
                    self.lg.info(
                        "resolved context_key from agent config",
                        extra={"agent": agent_name, "context_key": context_key},
                    )
                    return context_key

        self.lg.info(
            "no context_key in agent config, using agent name",
            extra={"agent": agent_name},
        )
        return None

    def _continuous_mode(self, context_key: str) -> int:
        """Rate multiple items in continuous mode."""
        assert self._backend is not None
        count = 0
        while True:
            fact = self._get_next_unrated(context_key)
            if fact is None:
                if count == 0:
                    print("\nNo unrated facts found.")
                else:
                    print(f"\n✓ Rated {count} items. No more unrated facts.")
                return 0

            self._display_fact(fact)
            result = self._get_rating_input()

            if result is None:  # quit
                print(f"\n✓ Rated {count} items. Exiting.")
                return 0

            if result == "skip":
                continue

            assert isinstance(result, tuple), "result should be tuple after skip check"
            signal, strength, stars = result
            stars_visual = self._format_stars(stars)
            print(f"{stars_visual} ", end="")
            self._save_manual_feedback(fact["id"], signal, strength, stars)
            count += 1
            print(f"✓ Saved ({count} rated)")

    def _single_rating(self, context_key: str) -> int:
        """Rate a single item."""
        fact = self._get_next_unrated(context_key)
        if fact is None:
            print("No unrated facts found.")
            return 0

        self._display_fact(fact)
        result = self._get_rating_input()

        if result is None or result == "skip":
            print("Skipped.")
            return 0

        assert isinstance(result, tuple), "result should be tuple after None/skip check"
        signal, strength, stars = result
        stars_visual = self._format_stars(stars)
        print(f"{stars_visual} ✓ Saved")
        self._save_manual_feedback(fact["id"], signal, strength, stars)
        return 0

    def _get_next_unrated(self, context_key: str) -> dict[str, Any] | None:
        """Get next unrated fact matching filters.

        Args:
            context_key: Context key to filter by

        Returns:
            dict with fact details or None if no unrated facts
        """
        assert self._backend is not None
        for fact in self._backend.unrated_facts(
            context_key,
            fact_type=self.args.type,
            category=self.args.category,
            limit=1,
        ):
            return fact
        return None

    def _display_fact(self, fact: dict[str, Any]) -> None:
        """Display fact in generic format."""
        print("\n" + "=" * 80)
        print(
            f"Type: {fact['type']} | Category: {fact['category'] or 'N/A'} | "
            f"Source: {fact['source']} | ID: {fact['id']}"
        )

        # Format datetime
        created_at = fact["created_at"]
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_at_str = str(created_at)
        print(f"Created: {created_at_str}")
        print("-" * 80)

        # Display content
        content = fact["content"]
        # Try to parse as JSON for pretty printing
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))
        except (json.JSONDecodeError, TypeError):
            # Plain text content
            print(content)

        print("=" * 80)

    def _get_rating_input(self) -> tuple[str, float, int] | str | None:
        """Get rating input from user.

        Returns:
            tuple[str, float, int]: (signal, strength, stars) for positive/negative/dismiss
            str: "skip" to skip this item without recording
            None: to quit
        """
        prompt = "\nRating (1-5 stars, g/👍=★★★★★, b/👎=★☆☆☆☆, s/skip, q/quit): "

        while True:
            try:
                user_input = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if not user_input:
                continue

            # Handle special commands
            if user_input in ("q", "quit"):
                return None
            if user_input in ("s", "skip"):
                return "skip"
            if user_input in ("g", "good", "👍"):
                return ("positive", 1.0, 5)
            if user_input in ("b", "bad", "👎"):
                return ("negative", 1.0, 1)

            # Try star rating
            result = self._parse_star_rating(user_input)
            if result:
                return result

    def _parse_star_rating(self, user_input: str) -> tuple[str, float, int] | None:
        """Parse numeric star rating (1-5)."""
        try:
            stars = int(user_input)
            if stars == 1:
                return ("negative", 1.0, 1)
            elif stars == 2:
                return ("negative", 0.5, 2)
            elif stars == 3:
                return ("dismiss", 1.0, 3)
            elif stars == 4:
                return ("positive", 0.5, 4)
            elif stars == 5:
                return ("positive", 1.0, 5)
            else:
                print("Stars must be 1-5")
                return None
        except ValueError:
            print("Invalid input. Use: 1-5 (stars), g/👍 (5★), b/👎 (1★), s/skip, q/quit")
            return None

    def _format_stars(self, stars: int) -> str:
        """Format star rating visually.

        Args:
            stars: Number of stars (0-5)

        Returns:
            Star visualization (e.g., "★★★☆☆")
        """
        filled = "★" * stars
        empty = "☆" * (5 - stars)
        return filled + empty

    def _auto_mode(self, context_key: str) -> int:
        """Automated rating mode using LLM via Service.

        Args:
            context_key: Context key for filtering facts.

        Returns:
            Exit code (0 = success).
        """
        assert self._backend is not None

        # Load and validate rating config
        result = self._load_rating_config()
        if result is None:
            return 1
        provider, criteria_config, fact_type = result

        # Determine how many facts to rate
        to_rate = self._determine_rating_count(context_key, fact_type)
        if to_rate == 0:
            print("No unrated facts found.")
            return 0

        # Create LLM client and run rating
        return self._run_auto_rating(provider, criteria_config, context_key, to_rate, fact_type)

    def _load_rating_config(self) -> tuple[Any, Any, str] | None:
        """Load and validate rating configuration.

        Returns:
            Tuple of (provider, criteria_config, fact_type) or None on error.
        """
        config = self._load_agent_config(self.args.agent_name)
        if config is None:
            print(f"Error: Agent config not found for '{self.args.agent_name}'")
            return None

        if "rating" not in config:
            print(f"Error: No rating configuration found in {self.args.agent_name} config")
            print("Add a 'rating:' section to the agent's YAML config")
            return None

        # Parse rating config using ConfigParser
        rating_config = config.get("rating", {})
        parser = ConfigParser(self.lg)
        providers = parser.parse_providers(rating_config.get("providers", []))
        criteria_map = parser.parse_criteria(rating_config.get("models", {}))

        # Validate we have what we need
        enabled_providers = [p for p in providers if p.enabled]
        if not enabled_providers:
            print("Error: No enabled rating providers configured")
            return None

        fact_type = self.args.type or "solution"
        if fact_type not in criteria_map:
            print(f"Error: No criteria configured for fact type: {fact_type}")
            return None

        return enabled_providers[0], criteria_map[fact_type], fact_type

    def _run_auto_rating(
        self, provider: Any, criteria_config: Any, context_key: str, to_rate: int, fact_type: str
    ) -> int:
        """Create LLM client and run auto rating."""
        from llm_infer.client import Factory as LLMClientFactory

        from ...core.llm import LLMCaller

        llm_config = self.app.config.get("llm", {})
        router = LLMClientFactory(self.lg).from_config(llm_config)
        caller = LLMCaller(self.lg, router)

        try:
            service = Service(self.lg, caller)
            total_rated = self._rate_with_service(
                service, provider, criteria_config, context_key, to_rate, fact_type
            )
            print(f"\n✓ Rated {total_rated} items using automated LLM rating")
            return 0
        finally:
            caller.close()

    def _determine_rating_count(self, context_key: str, fact_type: str) -> int:
        """Determine how many facts to rate based on limit and available unrated facts."""
        assert self._backend is not None
        unrated_count = self._backend.get_unrated_count(
            context_key, fact_type=fact_type, category=self.args.category
        )
        if unrated_count == 0:
            return 0

        limit: int = self.args.limit
        if limit == 0:
            print(f"Found {unrated_count} unrated facts. Rating all...")
            return unrated_count
        else:
            to_rate = min(limit, unrated_count)
            print(f"Found {unrated_count} unrated facts. Rating {to_rate}...")
            return to_rate

    def _rate_with_service(
        self,
        service: Service,
        provider: Any,
        criteria_config: Any,
        context_key: str,
        to_rate: int,
        fact_type: str,
    ) -> int:
        """Rate facts using Service and save with backend.

        Args:
            service: Rating service instance.
            provider: Provider configuration.
            criteria_config: Criteria configuration for the fact type.
            context_key: Context key for filtering.
            to_rate: Number of facts to rate.
            fact_type: Resolved fact type for filtering.

        Returns:
            Total number of facts rated.
        """
        assert self._backend is not None
        batch_size = 10
        total_rated = 0
        remaining = to_rate

        while remaining > 0:
            batch_limit = min(batch_size, remaining)
            facts = list(
                self._backend.unrated_facts(
                    context_key,
                    fact_type=fact_type,
                    category=self.args.category,
                    limit=batch_limit,
                )
            )
            if not facts:
                break

            for fact in facts:
                if self._rate_and_save_fact(service, provider, criteria_config, fact):
                    total_rated += 1

            remaining -= len(facts)

            print(f"Rated {total_rated}/{to_rate}...")

        return total_rated

    def _build_rating_request(
        self, provider: Any, criteria_config: Any, fact: dict[str, Any]
    ) -> Request:
        """Build rating request for a fact."""
        backend_type = provider.backend.get("type", "unknown")
        return Request(
            fact=fact["id"],
            content=fact["content"],
            prompt_template=criteria_config.prompt,
            criteria=criteria_config.criteria,
            model=provider.model,
            provider=f"llm_{provider.model}_{backend_type}",
            temperature=0.3,
        )

    def _rate_and_save_fact(
        self, service: Service, provider: Any, criteria_config: Any, fact: dict[str, Any]
    ) -> bool:
        """Rate a single fact and save. Returns True on success."""
        assert self._backend is not None
        try:
            request = self._build_rating_request(provider, criteria_config, fact)
            result = service.rate_content(request)
            self._backend.save_rating(result, source="cli_rate_tool")

            stars_visual = self._format_stars(result.stars)
            snippet = fact["content"][:60]
            ellipsis = "..." if len(fact["content"]) > 60 else ""
            print(f"{stars_visual} Fact {fact['id']}: {snippet}{ellipsis}")
            return True
        except Exception as e:
            self.lg.warning("rating failed", extra={"exception": e, "fact_id": fact["id"]})
            return False

    def _save_manual_feedback(self, fact_id: int, signal: str, strength: float, stars: int) -> None:
        """Save manual feedback from user input.

        Uses the Result model with ProviderType.MANUAL for consistency.

        Args:
            fact_id: ID of the fact being rated.
            signal: "positive", "negative", or "dismiss".
            strength: Confidence [0.0-1.0].
            stars: Star rating (1-5).
        """
        assert self._backend is not None
        from ...core.memory.rating import Result

        result = Result(
            fact=fact_id,
            signal=signal,  # type: ignore[arg-type]
            strength=strength,
            stars=stars,
            criteria_scores={},
            reasoning="Manual rating via CLI",
            provider_type=ProviderType.MANUAL,
            model=None,
            provider="cli_rate_tool",
        )
        self._backend.save_rating(result, source="cli_rate_tool")
