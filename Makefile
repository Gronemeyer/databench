# The version comes from git tags (setuptools-scm); a release is a tag.

.PHONY: version release

version:
	@python -c "from databench.provenance import _databench_version as v; print(v())"

release:
	@test -n "$(V)" || (echo "usage: make release V=0.3.1"; exit 1)
	@test -z "$$(git status --porcelain --untracked-files=no)" || (echo "commit or stash changes first"; exit 1)
	git tag -a v$(V) -m "databench v$(V)"
	git push origin v$(V)
