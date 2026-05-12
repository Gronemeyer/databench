release:
	@test -n "$(V)" || (echo "usage: make release V=1.3.0"; exit 1)
	python -m build
	git tag v$(V)
	git push --tags
