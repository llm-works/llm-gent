# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

local := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_gent

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# SPDX header enforcement
INFRA_DEV_CQ_SPDX := true

# Run every example script as a `make check` subcheck (mirrors `make examples.check`)
INFRA_DEV_CHECK_EXAMPLES := true

# PyTest and Docstring coverage thresholds
INFRA_PYTEST_COVERAGE_THRESHOLD := 70
INFRA_DEV_DOCSTRING_THRESHOLD := 95

# Coverage counts both unit and integration tests (integration exercises the
# alembic-backed Postgres store; unit-only coverage would understate real
# coverage of llm_gent/flow/stores and llm_gent/schema).
INFRA_PYTEST_COVERAGE_MARKERS := unit or integration

# PostgreSQL configuration (drives Makefile.pg targets and alembic CLI)
INFRA_PG_CONFIG_FILE := pg.yaml
INFRA_PG_DATABASES := main unittest

# Point the alembic CLI at etc/pg.yaml so `make migrate` works without
# additional env setup. env.py reads dbs.<LLM_GENT_DB_KEY>.url (default main).
export LLM_GENT_CONFIG := $(local)etc/pg.yaml

# Include framework
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.pg
include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.install
include $(infra)/make/Makefile.clean

# -----------------------------------------------------------------------------
# Alembic migrations (llm-gent framework-level schema)
# -----------------------------------------------------------------------------

ALEMBIC := $(PYTHON) -m alembic -c llm_gent/migrations/alembic.ini
.PHONY: migrate migrate.status migrate.history migrate.new migrate.test

migrate: ## Run llm-gent schema migrations to head (uses LLM_GENT_DB_KEY, default 'main')
	$(ALEMBIC) upgrade head

migrate.status: ## Show current migration revision on the target database
	$(ALEMBIC) current

migrate.history: ## Show the migration revision history
	$(ALEMBIC) history

migrate.new: ## Create a new revision. Usage: make migrate.new m="add foo table"
	$(ALEMBIC) revision -m "$(m)"

migrate.test: ## Run migrations against the unittest database (LLM_GENT_DB_KEY=unittest)
	LLM_GENT_DB_KEY=unittest $(ALEMBIC) upgrade head
