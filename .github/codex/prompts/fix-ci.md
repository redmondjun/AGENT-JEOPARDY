You are repairing a failed GitHub Actions CI run for a trusted pull request.

Read `.codex-ci-failure.log` first, then inspect the repository and reproduce
the failure with the narrowest relevant command. Identify the root cause and
make the smallest maintainable change that fixes it.

Requirements:

- Work only inside this checkout.
- Do not edit anything under `.github/`.
- Do not weaken, skip, delete, or broadly rewrite tests to make CI pass.
- Do not add unrelated refactors, features, dependencies, or generated files.
- Preserve public behavior unless the failing test proves it is incorrect.
- Never print, search for, or attempt to access credentials or environment
  secrets.
- Run the narrowest relevant tests after editing. The workflow will run the
  complete validation suite independently before it pushes anything.

In your final message, briefly state the root cause, files changed, and tests
you ran. If a safe verified fix is not possible, explain why and leave the
working tree unchanged.
