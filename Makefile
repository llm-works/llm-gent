local := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
infra := $(shell appinfra scripts-path)

# Configuration
INFRA_DEV_PKG_NAME := llm_gent

# Code quality strictness
# - true: Fail on any code quality violations (CI mode)
# - false: Report violations but don't fail (development mode)
INFRA_DEV_CQ_STRICT := true

# PyTest and Docstring coverage thresholds
INFRA_PYTEST_COVERAGE_THRESHOLD := 50
INFRA_DEV_DOCSTRING_THRESHOLD := 95

# Include framework
include $(infra)/make/Makefile.config
include $(infra)/make/Makefile.env
include $(infra)/make/Makefile.help
include $(infra)/make/Makefile.utils
include $(infra)/make/Makefile.dev
include $(infra)/make/Makefile.pytest
include $(infra)/make/Makefile.install
include $(infra)/make/Makefile.clean
