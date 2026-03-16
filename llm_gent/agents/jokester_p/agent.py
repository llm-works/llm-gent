"""Programmatic joke-teller agent with guaranteed novelty checking.

This agent demonstrates the difference between prompt-based and programmatic agents:
- Prompt-based: Relies on LLM to follow instructions (probabilistic)
- Programmatic: Code enforces constraints (deterministic)

For joke-telling, the key constraint is "never repeat" across thousands of jokes,
which cannot be reliably enforced through prompts alone due to context window limits.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger
from appinfra.time import Ticker, TickerMode, since
from llm_infer.client import BackendUnavailableError
from llm_infer.client.types import AdapterInfo

from ...core.agent import Agent, ExecutionResult
from ...core.llm.backend import StructuredOutputError
from ...core.traits.builtin.learn import LearnTrait
from .generate import GenerationAttempt, JokeGenerator
from .rating import BatchRater
from .storage import Storage


class JokesterAgent(Agent):
    """Programmatic agent that tells jokes with guaranteed novelty checking.

    Unlike prompt-based agents that rely on LLM instructions, this agent uses
    code to enforce the "never repeat" constraint across thousands of jokes.

    Flow:
        1. Recall recent jokes for style context
        2. LLM generates candidate joke
        3. Code validates novelty via embedding similarity (guaranteed check)
        4. If too similar, retry with feedback (code-controlled loop)
        5. Save novel joke to memory
    """

    def __init__(
        self,
        lg: Logger,
        config: DotDict,
    ) -> None:
        """Initialize joke-teller agent.

        Args:
            lg: Logger instance.
            config: Agent configuration with keys:
                - identity.name: Agent name (required)
                - max_retries: Maximum attempts to generate novel joke (default: 3)
                - similarity_threshold: Min similarity for duplicate (0.0-1.0, default: 0.85)
                - denylist: List of words/phrases to filter (case-insensitive, default: [])
                - parallel: Number of parallel LLM calls (default: 1)
        """
        super().__init__(lg, config)
        self._jokes_generated_this_session = 0
        self._recent_results: deque[ExecutionResult] = deque(maxlen=100)
        self._storage: Storage | None = None
        self._generator: JokeGenerator | None = None
        self._rater: BatchRater | None = None
        self._last_joke_time: float = time.monotonic()
        self._adapter_target_reached: bool = False
        self._parallel: int = config.get("parallel", 1)
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start agent and traits."""
        from .factory import Factory

        self._start_traits()

        try:
            components = Factory.create_components(self)
            self._generator = components.generator
            self._storage = components.storage
            self._rater = components.rater
        except Exception:
            self._stop_traits()
            raise

        # Create persistent event loop for async operations
        self._loop = asyncio.new_event_loop()
        self._started = True
        self._lg.info("agent started", extra={"agent": self.name})

    def stop(self) -> None:
        """Stop agent and traits."""
        if not self._started:
            return
        self._stop_traits()
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        self._storage = None
        self._generator = None
        self._rater = None
        self._started = False
        self._lg.info("agent stopped", extra={"agent": self.name})

    def run(self) -> ExecutionResult:
        """Run agent loop respecting schedule.interval until target reached.

        Called by core runner on serve. Uses Ticker for precise scheduling.
        Exits when target is reached or shutdown is signaled.

        Returns:
            ExecutionResult with final status.
        """
        interval = self._get_schedule_interval()
        self._lg.info(
            "starting run loop",
            extra={"interval": interval, "target": self.config.get("target", {})},
        )
        ticker = self._create_ticker(interval)
        last_result: ExecutionResult | None = None

        while not self._adapter_target_reached:
            if not self._wait_for_tick(ticker):
                continue
            self._cycle_count += 1
            last_result = self._run_generation_cycle() or last_result

        return last_result or ExecutionResult(
            success=True, content="Target reached", iterations=self._cycle_count
        )

    def _create_ticker(self, interval: float) -> Ticker | None:
        """Create ticker for scheduled execution (None for continuous)."""
        if interval <= 0:
            return None
        return Ticker(self._lg, secs=interval, mode=TickerMode.FLEX, initial=True)

    def _wait_for_tick(self, ticker: Ticker | None) -> bool:
        """Wait for ticker if scheduled. Returns True if cycle should run."""
        if ticker is None:
            return True
        wait_time = ticker.time_until_next_tick()
        if wait_time > 0:
            time.sleep(wait_time)
        return ticker.try_tick()

    def _run_generation_cycle(self) -> ExecutionResult | None:
        """Run one generation cycle with error handling."""
        try:
            return self._execute_generation(continuous=True)
        except (StructuredOutputError, BackendUnavailableError) as e:
            self._lg.warning("generation failed", extra={"error": str(e)})
        except Exception as e:
            self._lg.warning("generation failed", extra={"exception": e})
        return None

    def run_once(self) -> ExecutionResult:
        """Generate a single joke and return.

        Called by core runner on RPC request. Generates one joke
        and returns immediately.

        Returns:
            ExecutionResult with the joke or error.
        """
        self._cycle_count += 1
        try:
            return self._execute_generation(continuous=False)
        except (StructuredOutputError, BackendUnavailableError) as e:
            self._lg.warning("joke generation failed", extra={"error": str(e)})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)
        except Exception as e:
            self._lg.warning("joke generation failed", extra={"exception": e})
            return ExecutionResult(success=False, content=f"Error: {e}", iterations=1)

    def _get_schedule_interval(self) -> float:
        """Get schedule interval from config (default 0 for continuous)."""
        schedule_config = self.config.get("schedule", {})
        return float(schedule_config.get("interval", 0))

    def _execute_generation(self, continuous: bool) -> ExecutionResult:
        """Execute joke generation.

        Args:
            continuous: If True, run until target. If False, stop after one save.
        """
        assert self._generator is not None
        assert self._storage is not None

        if self._adapter_target_reached:
            return ExecutionResult(
                success=True,
                content="Adapter target already reached",
                iterations=0,
            )

        if self._parallel > 1:
            return self._execute_parallel(continuous)

        return self._execute_serial()

    def _execute_serial(self) -> ExecutionResult:
        """Execute single joke generation (serial mode, one joke)."""
        assert self._generator is not None

        learn_trait = self.require_trait(LearnTrait)
        recent_jokes = self._get_recent_jokes(learn_trait, limit=5)

        attempt = self._generator.generate(recent_jokes)

        if not attempt.success:
            return ExecutionResult(
                success=False,
                content=f"Failed to generate novel joke after {attempt.cumulative_attempts} attempts",
                iterations=attempt.cumulative_attempts,
            )

        return self._complete_cycle(attempt)

    def _execute_parallel(self, continuous: bool) -> ExecutionResult:
        """Execute parallel joke generation.

        Args:
            continuous: If True, run until target. If False, stop after one save.
        """
        assert self._generator is not None
        assert self._loop is not None

        learn_trait = self.require_trait(LearnTrait)
        recent_jokes = self._get_recent_jokes(learn_trait, limit=5)

        result = self._loop.run_until_complete(self._run_parallel_async(recent_jokes, continuous))
        return result

    async def _run_parallel_async(self, context: list[str], continuous: bool) -> ExecutionResult:
        """Run worker pool for parallel generation.

        Args:
            context: Recent jokes for style inspiration.
            continuous: If True, run until target. If False, stop after one save.
        """
        assert self._generator is not None
        self._lg.debug(
            "starting worker pool",
            extra={"workers": self._parallel, "mode": "continuous" if continuous else "single"},
        )
        pending = self._spawn_initial_workers(context)
        saved_count, total_attempts, last_result = 0, 0, None

        while pending and self._should_continue_parallel(continuous, saved_count):
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                total_attempts += 1
                result = self._process_worker_task(task)
                if result is not None:
                    last_result, saved_count = result, saved_count + 1
                if self._should_continue_parallel(continuous, saved_count):
                    pending.add(asyncio.create_task(self._generator.generate_async(context)))

        return await self._finalize_parallel_run(pending, saved_count, total_attempts, last_result)

    def _spawn_initial_workers(self, context: list[str]) -> set[asyncio.Task[GenerationAttempt]]:
        """Spawn initial worker pool."""
        assert self._generator is not None
        return {
            asyncio.create_task(self._generator.generate_async(context))
            for _ in range(self._parallel)
        }

    def _should_continue_parallel(self, continuous: bool, saved_count: int) -> bool:
        """Check if parallel generation should continue."""
        if self._adapter_target_reached:
            return False
        return continuous or saved_count == 0

    def _process_worker_task(self, task: asyncio.Task[GenerationAttempt]) -> ExecutionResult | None:
        """Process a completed worker task, returning result if saved."""
        try:
            attempt = task.result()
        except Exception as e:
            self._lg.debug("worker failed", extra={"exception": e})
            return None

        if not attempt.success:
            return None
        return self._try_save_candidate(attempt)

    async def _finalize_parallel_run(
        self,
        pending: set[asyncio.Task[GenerationAttempt]],
        saved_count: int,
        total_attempts: int,
        last_result: ExecutionResult | None,
    ) -> ExecutionResult:
        """Finalize parallel run: cancel remaining workers and return result."""
        for task in pending:
            task.cancel()

        # Await cancelled tasks to ensure cleanup runs
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        self._lg.debug(
            "worker pool stopped",
            extra={"saved": saved_count, "attempts": total_attempts},
        )

        if saved_count == 0:
            return ExecutionResult(
                success=False,
                content=f"No novel jokes from {total_attempts} attempts",
                iterations=total_attempts,
            )

        assert last_result is not None
        return last_result

    def _try_save_candidate(self, attempt: GenerationAttempt) -> ExecutionResult | None:
        """Try to save a candidate, re-checking novelty for race conditions."""
        assert self._generator is not None
        assert attempt.joke is not None

        # Re-check novelty to handle race condition between parallel workers
        recheck = self._generator.check_novelty(attempt.joke, attempt.model_name, attempt.adapter)
        if not recheck.success:
            self._lg.debug(
                "parallel candidate rejected (race)",
                extra={"similarity": recheck.max_similarity, "joke": attempt.joke.text[:50]},
            )
            return None

        return self._complete_cycle(attempt)

    def _complete_cycle(self, attempt: GenerationAttempt) -> ExecutionResult:
        """Complete joke generation cycle with save and logging."""
        assert attempt.joke is not None
        assert self._storage is not None
        assert self._generator is not None

        if self._is_target_reached(attempt.adapter, attempt.model_name):
            self._adapter_target_reached = True
            return ExecutionResult(
                success=True, content="Adapter target reached - stopping generation", iterations=0
            )

        fact_id, schema = self._storage.save_joke(
            joke=attempt.joke,
            model_name=attempt.model_name,
            attempts=attempt.cumulative_attempts,
            adapter=attempt.adapter,
        )
        self._jokes_generated_this_session += 1
        self._generator.record_success(attempt.joke.text, fact_id)

        result = self._build_success_result(attempt, fact_id)
        if self._rater:
            self._rater.queue(fact_id, attempt.joke.text, schema)
        return result

    def _build_success_result(self, attempt: GenerationAttempt, fact_id: int) -> ExecutionResult:
        """Build success result and log the new joke."""
        assert attempt.joke is not None
        result = ExecutionResult(
            success=True,
            content=f"{attempt.joke.text}\n(Style: {attempt.joke.style})",
            iterations=attempt.cumulative_attempts,
        )
        self._recent_results.append(result)
        self._lg.info("found new joke", extra=self._build_joke_log_extra(attempt, fact_id))
        self._last_joke_time = time.monotonic()
        if attempt.similar_joke:
            self._lg.debug("closest existing joke", extra={"joke": attempt.similar_joke})
        return result

    def _is_target_reached(self, adapter: AdapterInfo | None, model_name: str) -> bool:
        """Check if adapter (or base model) has reached its target count."""
        target_config = self.config.get("target", {})
        adapter_target = target_config.get("adapter")
        if not adapter_target:
            return False

        max_chars = target_config.get("max_chars")
        assert self._storage is not None

        # Base model case - filter by specific model name
        if adapter is None or not adapter.actual:
            count = self._storage.get_base_model_count(model_name=model_name, max_chars=max_chars)
            if count >= adapter_target:
                self._lg.info(
                    "base model target reached",
                    extra={"model": model_name, "count": count, "target": adapter_target},
                )
                return True
            return False

        # Adapter case
        schema = self._storage._resolve_schema(adapter)
        count = self._storage.get_adapter_count(adapter.actual, schema=schema, max_chars=max_chars)

        if count >= adapter_target:
            self._lg.info(
                "adapter target reached",
                extra={"adapter": adapter.actual, "count": count, "target": adapter_target},
            )
            return True
        return False

    def _build_joke_log_extra(self, attempt: GenerationAttempt, fact_id: int) -> dict[str, Any]:
        """Build log extra dict for joke generation."""
        assert attempt.joke is not None
        return {
            "after": since(self._last_joke_time),
            "fact": fact_id,
            "agent": self.name,
            "joke": attempt.joke.text,
            "chars": len(attempt.joke.text),
            "model": attempt.model_name,
            "style": attempt.joke.style,
            "closest": round(attempt.max_similarity, 2),
            "attempts": {"run": attempt.run_attempts, "cumulative": attempt.cumulative_attempts},
            "sess_count": self._jokes_generated_this_session,
        }

    def ask(self, question: str) -> str:
        """Answer question (not supported)."""
        return "Jokester agent does not support questions. Use run_once() to get a joke."

    def _get_recent_jokes(self, learn_trait: LearnTrait, limit: int) -> list[str]:
        """Get recent jokes from in-memory history for style inspiration."""
        if self._generator is None:
            return []
        return self._generator.get_recent_jokes(limit)

    def record_feedback(self, message: str) -> None:
        """Record feedback about a joke.

        Args:
            message: Feedback message from user.
        """
        # Log at debug level to avoid exposing PII in production logs
        self._lg.debug("feedback received", extra={"feedback": message})

    def get_recent_results(self, limit: int = 10) -> list[ExecutionResult]:
        """Get recent execution results.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of recent execution results.
        """
        return list(self._recent_results)[-limit:]
