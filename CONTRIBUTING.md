# Contributing to LEAD

## Thank You! 🙏

Thank you for your interest in contributing to LEAD! We appreciate your enthusiasm and support for advancing end-to-end autonomous driving research. Contributions are welcome and encouraged!

## How to Contribute

### Report Issues
Found a bug? Please [open an issue](https://github.com/autonomousvision/lead/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, CUDA version, etc.)

### Improve Documentation
Documentation contributions are always welcome! This includes:
- Fixing typos or unclear explanations
- Adding examples or tutorials
- Improving setup instructions
- Clarifying FAQ entries

### Feature Requests and Enhancements
We welcome feature contributions! To ensure quality and maintainability:
- **Include tests**: All new features should include appropriate unit tests in the `tests/` directory
- **Verify scripts run**: Ensure all modified scripts execute successfully
- **Keep PRs focused**: Smaller, well-scoped pull requests are easier to review and merge
- **Follow existing patterns**: Match the coding style and architecture of the existing codebase

### Code Quality Standards
- Install pre-commit hooks locally before starting development: `pre-commit install`
- Use type hints with `jaxtyping` for tensor operations
- Add appropriate `beartype` decorators for runtime validation
- Include docstrings for all public functions
- Run existing tests before submitting: `pytest tests/`

## Contact

For questions or discussions:
- Open an issue for bugs or feature requests
- See the maintainers listed in [README.md](README.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*This document will be updated once the repository stabilizes. Thank you for your patience!*
