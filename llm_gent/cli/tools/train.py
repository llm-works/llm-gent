"""Train tool - create training manifests for agents."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Literal, Protocol

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from llm_kelt.training import Factory as TrainFactory


class AdapterNotFoundError(Exception):
    """Raised when a specified adapter cannot be found."""

    def __init__(self, md5: str) -> None:
        self.md5 = md5
        super().__init__(f"Adapter not found: {md5}")


class TrainingDataProvider(Protocol):
    """Protocol for agents to provide training data."""

    def add_args(self, parser: argparse.ArgumentParser, method: str = "sft") -> None:
        """Add agent-specific arguments to the parser."""
        ...

    def get_sft_examples(self, args: argparse.Namespace) -> list[dict[str, str]]:
        """Get SFT training examples.

        Returns:
            List of {"instruction": str, "output": str} dicts.
        """
        ...

    def get_dpo_pairs(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        """Get DPO preference pairs.

        Returns:
            List of {"chosen": str, "rejected": str, "prompt": str} dicts.
        """
        ...

    def get_context_key(self) -> str:
        """Get context key for this agent."""
        ...

    def get_description(self, method: str, count: int) -> str:
        """Get description for the training manifest."""
        ...


class TrainTool(Tool):
    """Create training manifests for agents."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="train", help_text="Create training manifests for agents")
        super().__init__(parent, config)
        self._pg: PG | None = None

    def configure(self) -> None:
        """Set up database connection."""
        if not hasattr(self.app, "config"):
            raise RuntimeError("App does not have config")

        learn_config = getattr(self.app.config, "learn", None)
        if learn_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        db_config = getattr(learn_config, "db", None)
        if db_config is None:
            raise RuntimeError("Database configuration not found in learn.db")

        self._pg = PG(self.lg, db_config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        # Top-level args (apply to all methods)
        self._add_common_args(parser)
        self._add_training_args(parser)

        subparsers = parser.add_subparsers(dest="method", help="Training method")

        # SFT subcommand
        sft_parser = subparsers.add_parser("sft", help="Supervised fine-tuning")
        self._add_agent_subcommand(sft_parser)

        # DPO subcommand
        dpo_parser = subparsers.add_parser("dpo", help="Direct preference optimization")
        self._add_agent_subcommand(dpo_parser)

        # Prompt tuning subcommand
        prompt_parser = subparsers.add_parser(
            "prompt", help="Prompt tuning (stable for large models)"
        )
        self._add_prompt_args(prompt_parser)
        self._add_agent_subcommand(prompt_parser)

    def _add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common training arguments."""
        parser.add_argument(
            "--model",
            type=str,
            help="Base model for training (default: from llm-kelt config)",
        )
        parser.add_argument(
            "--adapter",
            type=str,
            help="Name for the trained adapter (default: <agent>-<method>)",
        )
        parser.add_argument(
            "--from-adapter",
            type=str,
            help="Existing adapter (md5) to train on top of",
        )
        parser.add_argument(
            "--schema",
            type=str,
            help="Schema for this adapter's data (default: from agent's kelt.schema)",
        )
        parser.add_argument(
            "--registry-path",
            type=str,
            help="Path to adapter registry (default: from config or ~/.llm-kelt/adapters)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without saving",
        )

    def _add_training_args(self, parser: argparse.ArgumentParser) -> None:
        """Add training hyperparameter arguments."""
        self._add_lora_args(parser)
        self._add_optimizer_args(parser)

    def _add_lora_args(self, parser: argparse.ArgumentParser) -> None:
        """Add LoRA configuration arguments."""
        parser.add_argument("--lora-r", type=int, help="LoRA rank (default: 16)")
        parser.add_argument("--lora-alpha", type=int, help="LoRA alpha (default: 2x rank)")
        parser.add_argument("--lora-dropout", type=float, help="LoRA dropout (default: 0.05)")
        parser.add_argument(
            "--rslora",
            action="store_true",
            help="Enable rsLoRA scaling (recommended for 32B+ models)",
        )

    def _add_optimizer_args(self, parser: argparse.ArgumentParser) -> None:
        """Add optimizer/training arguments."""
        parser.add_argument("--lr", type=float, dest="learning_rate", help="Learning rate")
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, help="Batch size")
        parser.add_argument(
            "--neftune-alpha",
            type=float,
            help="NEFTune noise alpha (recommended: 5 for <5K samples)",
        )

    def _add_prompt_args(self, parser: argparse.ArgumentParser) -> None:
        """Add prompt tuning specific arguments."""
        parser.add_argument(
            "--num-tokens",
            type=int,
            default=20,
            help="Number of virtual tokens (default: 20, range: 8-50)",
        )
        parser.add_argument(
            "--init-text",
            type=str,
            help="Text to initialize virtual tokens (e.g., 'You are a witty comedian.')",
        )
        parser.add_argument(
            "--init-random",
            action="store_true",
            help="Use random initialization instead of text",
        )

    def _add_agent_subcommand(self, parser: argparse.ArgumentParser) -> None:
        """Add 'agent' subcommand that takes agent name and args."""
        subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand")
        agent_parser = subparsers.add_parser("agent", help="Train a specific agent")
        agent_parser.add_argument(
            "agent_name",
            metavar="name",
            help="Agent name (e.g., jokester-p)",
        )
        # Remaining args will be parsed by agent's data provider
        agent_parser.add_argument(
            "agent_args",
            nargs=argparse.REMAINDER,
            help="Agent-specific arguments",
        )

    def run(self, **kwargs: Any) -> int:
        method = self._validate_method()
        if method is None:
            return 1

        provider = self._get_validated_provider(method)
        if provider is None:
            return 1

        try:
            if method == "sft":
                return self._run_sft(provider)
            elif method == "dpo":
                return self._run_dpo(provider)
            elif method == "prompt":
                return self._run_prompt(provider)
            else:
                print(f"Unknown method: {method}")
                return 1
        except AdapterNotFoundError as e:
            print(f"Error: {e}")
            return 1

    def _validate_method(self) -> str | None:
        """Validate method and subcommand args."""
        method = self.args.method
        if method is None:
            print("Usage: train <sft|dpo|prompt> [options] agent <name> [agent-options]")
            return None

        subcommand = getattr(self.args, "subcommand", None)
        if subcommand != "agent":
            print(f"Usage: train {method} [options] agent <name> [agent-options]")
            return None

        return str(method)

    def _get_validated_provider(self, method: str) -> TrainingDataProvider | None:
        """Get provider and parse agent-specific args."""
        agent_name = self.args.agent_name
        provider = self._get_data_provider(agent_name)
        if provider is None:
            return None

        # Parse agent-specific args and merge into self.args
        agent_parser = argparse.ArgumentParser(prog=f"train {method} agent {agent_name}")
        provider.add_args(agent_parser, method=method)
        agent_args = agent_parser.parse_args(self.args.agent_args)
        for key, value in vars(agent_args).items():
            setattr(self.args, key, value)

        return provider

    def _get_data_provider(self, agent_name: str) -> TrainingDataProvider | None:
        """Get training data provider for an agent."""
        # Load agent config
        if not hasattr(self.app.config, "agents"):
            print("Error: No agents configured")
            return None

        agents = self.app.config.agents
        if agent_name not in agents:
            print(f"Error: Agent '{agent_name}' not found")
            print(f"Available agents: {', '.join(agents.keys())}")
            return None

        agent_config = agents[agent_name]

        # Get the agent's training data provider
        # For now, support jokester-p directly; later make this pluggable
        if agent_name == "jokester-p":
            return self._create_jokester_provider(agent_config)

        print(f"Error: Agent '{agent_name}' does not support training")
        return None

    def _create_jokester_provider(self, config: DotDict) -> TrainingDataProvider:
        """Create training data provider for jokester-p agent."""
        from llm_gent.agents.jokester_p.training import JokesterTrainingProvider

        assert self._pg is not None
        context_key = self._resolve_context_key(config)
        return JokesterTrainingProvider(self.lg, self._pg, context_key)

    def _resolve_context_key(self, config: DotDict) -> str:
        """Resolve context key from agent config (kelt.identity.name)."""
        kelt_config = config.get("kelt", {})
        identity = kelt_config.get("identity", {})
        if isinstance(identity, dict) and "name" in identity:
            return str(identity["name"])
        return "unknown"

    def _run_sft(self, provider: TrainingDataProvider) -> int:
        """Run SFT training flow."""
        examples = provider.get_sft_examples(self.args)
        if not examples:
            print("\nNo examples available for SFT training.")
            return 0

        self._print_sft_summary(examples, provider)

        if self.args.dry_run:
            print("\nDry run complete. Use without --dry-run to submit manifest.")
            return 0

        factory = self._get_training_factory()
        return self._submit_manifest(factory, provider, "sft", examples)

    def _run_dpo(self, provider: TrainingDataProvider) -> int:
        """Run DPO training flow."""
        pairs = provider.get_dpo_pairs(self.args)
        if not pairs:
            print("\nNo pairs available for DPO training.")
            return 0

        self._print_dpo_summary(pairs, provider)

        if self.args.dry_run:
            print("\nDry run complete. Use without --dry-run to submit manifest.")
            return 0

        factory = self._get_training_factory()
        return self._submit_manifest(factory, provider, "dpo", pairs)

    def _run_prompt(self, provider: TrainingDataProvider) -> int:
        """Run prompt tuning flow.

        Prompt tuning uses same data format as SFT but trains only virtual tokens.
        Stable for large models (32B+) where LoRA often fails with NaN gradients.
        """
        examples = provider.get_sft_examples(self.args)
        if not examples:
            print("\nNo examples available for prompt tuning.")
            return 0

        self._print_prompt_summary(examples, provider)

        if self.args.dry_run:
            print("\nDry run complete. Use without --dry-run to submit manifest.")
            return 0

        factory = self._get_training_factory()
        return self._submit_manifest(factory, provider, "prompt", examples)

    def _get_training_factory(self) -> TrainFactory:
        """Get training factory - llm-kelt handles defaults."""
        path = self._get_registry_path()
        return TrainFactory(self.lg, Path(path))

    def _get_registry_path(self) -> str:
        """Get adapter registry path from args, config, or default."""
        if self.args.registry_path:
            return str(self.args.registry_path)

        try:
            adapters_config = self.app.config.learn.get("adapters", {})
            lora_config = adapters_config.get("lora", {})
            if base_path := lora_config.get("base_path"):
                return os.path.expanduser(str(base_path))
        except (AttributeError, KeyError):
            pass

        return os.environ.get("ADAPTER_REGISTRY_PATH", os.path.expanduser("~/.llm-kelt/adapters"))

    def _get_adapter_name(self, provider: TrainingDataProvider, method: str) -> str:
        """Get adapter name from args or generate default."""
        if self.args.adapter:
            return str(self.args.adapter)
        context_key = provider.get_context_key()
        return f"{context_key}-{method}"

    def _get_lora_overrides(self) -> dict[str, Any] | None:
        """Get LoRA config overrides from command line args."""
        overrides: dict[str, Any] = {}
        if getattr(self.args, "lora_r", None) is not None:
            overrides["r"] = self.args.lora_r
        if getattr(self.args, "lora_alpha", None) is not None:
            overrides["lora_alpha"] = self.args.lora_alpha
        if getattr(self.args, "lora_dropout", None) is not None:
            overrides["lora_dropout"] = self.args.lora_dropout
        if getattr(self.args, "rslora", False):
            overrides["use_rslora"] = True
        return overrides if overrides else None

    def _get_training_overrides(self) -> dict[str, Any] | None:
        """Get training config overrides from command line args."""
        overrides: dict[str, Any] = {}
        if getattr(self.args, "learning_rate", None) is not None:
            overrides["learning_rate"] = self.args.learning_rate
        if getattr(self.args, "epochs", None) is not None:
            overrides["num_epochs"] = self.args.epochs
        if getattr(self.args, "batch_size", None) is not None:
            overrides["batch_size"] = self.args.batch_size
        if getattr(self.args, "neftune_alpha", None) is not None:
            overrides["neftune_noise_alpha"] = self.args.neftune_alpha
        return overrides if overrides else None

    def _submit_manifest(
        self,
        factory: TrainFactory,
        provider: TrainingDataProvider,
        method: Literal["sft", "dpo", "prompt"],
        data: list[dict[str, Any]],
    ) -> int:
        """Create and submit training manifest."""
        adapter_name = self._get_adapter_name(provider, method)
        if not self._handle_existing_manifest(factory, adapter_name):
            return 1

        manifest = self._create_manifest(factory, provider, adapter_name, method, data)
        result = factory.manifest.submit(manifest)
        path = Path(result.location) if getattr(result, "location", None) else None
        self._print_submit_result(manifest, method, len(data), path)
        return 0

    def _handle_existing_manifest(self, factory: TrainFactory, adapter_name: str) -> bool:
        """Check for existing manifest and prompt for replacement. Returns False to abort."""
        existing = factory.manifest.get_pending(adapter_name)
        if existing:
            if not self._confirm_replace(adapter_name):
                print("Aborted.")
                return False
            factory.manifest.remove_pending(adapter_name)
        return True

    def _create_manifest(
        self,
        factory: TrainFactory,
        provider: TrainingDataProvider,
        adapter_name: str,
        method: Literal["sft", "dpo", "prompt"],
        data: list[dict[str, Any]],
    ) -> Any:
        """Create training manifest with all overrides."""
        config = self._build_config_overrides()
        return factory.manifest.create(
            adapter=adapter_name,
            method=method,
            model=self.args.model,
            parent=self._resolve_parent_adapter(factory),
            data=data,
            context_key=provider.get_context_key(),
            schema_name=self._resolve_schema(),
            description=provider.get_description(method, len(data)),
            config=config,
        )

    def _resolve_schema(self) -> str | None:
        """Resolve schema from --schema arg or agent's kelt.schema config."""
        if schema := getattr(self.args, "schema", None):
            return str(schema)

        # Fall back to agent's kelt.schema
        agent_name = getattr(self.args, "agent_name", None)
        if not agent_name:
            return None

        agent_config = self.app.config.agents.get(agent_name)
        if not agent_config:
            return None

        kelt_config = agent_config.get("kelt", {})
        schema_config = kelt_config.get("schema", {})
        schema_name = schema_config.get("name")
        return str(schema_name) if schema_name else None

    def _build_config_overrides(self) -> dict[str, Any] | None:
        """Build config overrides from lora, prompt, and training args."""
        config: dict[str, Any] = {}
        if lora := self._get_lora_overrides():
            config["lora"] = lora
        if prompt := self._get_prompt_overrides():
            config.update(prompt)  # Prompt args go at top level
        if training := self._get_training_overrides():
            config.update(training)  # Training args go at top level of config
        return config if config else None

    def _get_prompt_overrides(self) -> dict[str, Any] | None:
        """Get prompt tuning config overrides from command line args."""
        # Only relevant for prompt method
        if getattr(self.args, "method", None) != "prompt":
            return None

        overrides: dict[str, Any] = {}
        num_tokens = getattr(self.args, "num_tokens", None)
        if num_tokens is not None:
            overrides["num_virtual_tokens"] = num_tokens

        init_random = getattr(self.args, "init_random", False)
        if init_random:
            overrides["prompt_tuning_init"] = "RANDOM"
        else:
            overrides["prompt_tuning_init"] = "TEXT"
            init_text = getattr(self.args, "init_text", None)
            if init_text:
                overrides["prompt_tuning_init_text"] = init_text

        return overrides if overrides else None

    def _confirm_replace(self, adapter_name: str) -> bool:
        """Confirm replacement of existing manifest."""
        print(f"\nPending manifest already exists for '{adapter_name}'")
        try:
            response = input("Replace it? [y/N] ").strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _resolve_parent_adapter(self, factory: TrainFactory) -> Any:
        """Resolve parent adapter from --from-adapter arg.

        Supports abbreviated md5 format (e.g., "40..9d18").

        Raises:
            AdapterNotFoundError: If the specified adapter doesn't exist.
            ValueError: If abbreviated md5 is ambiguous.
        """
        from_adapter = getattr(self.args, "from_adapter", None)
        if not from_adapter:
            return None

        # Expand abbreviated md5 if needed
        md5 = self._expand_adapter_md5(from_adapter, factory)

        parent = factory.manifest.find_adapter(md5)
        if parent is None:
            raise AdapterNotFoundError(from_adapter)
        return parent

    def _expand_adapter_md5(self, value: str, factory: TrainFactory) -> str:
        """Expand abbreviated md5 (XX..XXXX) to full md5."""
        match = re.match(r"^([0-9a-f]{2})\.\.([0-9a-f]{4})$", value, re.IGNORECASE)
        if not match:
            return value  # Already full md5 or name

        prefix, suffix = match.groups()

        # Search completed manifests for matching md5
        matches: list[str] = []
        for manifest in factory.manifest.list_completed():
            if manifest.output and manifest.output.adapter:
                md5 = manifest.output.adapter.md5
                if md5 and md5.startswith(prefix) and md5.endswith(suffix):
                    matches.append(md5)

        if len(matches) == 0:
            raise AdapterNotFoundError(f"{prefix}..{suffix}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous adapter '{prefix}..{suffix}' matches: {matches}")
        return matches[0]

    def _print_sft_summary(
        self, examples: list[dict[str, str]], provider: TrainingDataProvider
    ) -> None:
        """Print SFT training summary."""
        print(f"\n=== SFT Training: {provider.get_context_key()} ===")
        print(f"Examples: {len(examples)}")
        print("\nSample:")
        for i, ex in enumerate(examples[:3], 1):
            output = ex.get("output", "")[:60]
            suffix = "..." if len(ex.get("output", "")) > 60 else ""
            print(f"  {i}. {output}{suffix}")
        if len(examples) > 3:
            print(f"  ... and {len(examples) - 3} more")

    def _print_dpo_summary(
        self, pairs: list[dict[str, Any]], provider: TrainingDataProvider
    ) -> None:
        """Print DPO training summary."""
        print(f"\n=== DPO Training: {provider.get_context_key()} ===")
        print(f"Pairs: {len(pairs)}")

        if hasattr(provider, "get_dpo_summary"):
            for line in provider.get_dpo_summary() or []:
                print(f"  {line}")

    def _print_prompt_summary(
        self, examples: list[dict[str, str]], provider: TrainingDataProvider
    ) -> None:
        """Print prompt tuning summary."""
        num_tokens = getattr(self.args, "num_tokens", 20)
        init_text = getattr(self.args, "init_text", None)
        init_mode = "RANDOM" if getattr(self.args, "init_random", False) else "TEXT"

        print(f"\n=== Prompt Tuning: {provider.get_context_key()} ===")
        print(f"Examples: {len(examples)}")
        print(f"Virtual tokens: {num_tokens}")
        print(f"Init mode: {init_mode}")
        if init_text:
            print(f"Init text: {init_text[:60]}{'...' if len(init_text) > 60 else ''}")
        print("\nSample:")
        for i, ex in enumerate(examples[:3], 1):
            output = ex.get("output", "")[:60]
            suffix = "..." if len(ex.get("output", "")) > 60 else ""
            print(f"  {i}. {output}{suffix}")
        if len(examples) > 3:
            print(f"  ... and {len(examples) - 3} more")

    def _print_submit_result(
        self, manifest: Any, method: str, count: int, path: Path | None
    ) -> None:
        """Print manifest submission result."""
        print(f"\n✓ Submitted {method.upper()} manifest")
        print(f"  Adapter:   {manifest.adapter}")
        model = manifest.training.get("requested_model") or "(from config)"
        print(f"  Model:     {model}")
        print(f"  Records:   {count}")
        print(f"  Path:      {path or '(unknown)'}")
