# Repository Guidelines

<!-- BEGIN CUSTOM TOOLING INSTRUCTIONS -->

## Custom tooling instructions

### Search

- Use `rg` instead of `grep` for project-wide searches.
- Do not use `grep` for recursive/project-wide searches because it is slower and may ignore repository search conventions.
- Examples:
  - `rg "pattern"`
  - `rg --files | rg "name"`
  - `rg -t python "def"`

### File finding

- Prefer `fd` for file discovery.
- Prefer `rg --files` when listing repository files.
- Do not use `find` or `ls -R` for broad project discovery unless there is a specific reason.

### JSON

- Use `jq` for JSON parsing and transformations.
- Do not parse JSON with regular expressions.

### Agent command replacements

- Replace `grep` with `rg`.
- Replace `find` with `rg --files` or `fd`.
- Replace `ls -R` with `rg --files`.
- Replace `cat file | grep pattern` with `rg "pattern" file`.

### Read limits

- Cap file reads at 250 lines unless more context is necessary.
- Prefer contextual searches such as `rg -n -A 3 -B 3 "pattern"`.

<!-- END CUSTOM TOOLING INSTRUCTIONS -->
