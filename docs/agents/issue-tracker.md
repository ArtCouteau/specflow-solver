# Issue tracker: GitHub

Issues live in GitHub Issues: `ArtCouteau/specflow-solver`. Use the `gh` CLI; pass `--repo ArtCouteau/specflow-solver` outside the clone.

## Commands

- Create: `gh issue create --title "..." --body-file <file> --label <label>`
- Read: `gh issue view <n> --comments`
- List: `gh issue list --state open --json number,title,labels`
- Comment: `gh issue comment <n> --body-file <file>`
- Close: comment first, then `gh issue close <n>`
- Label: `gh issue edit <n> --add-label <label>`
- Sub-issue: `gh api repos/ArtCouteau/specflow-solver/issues/<parent>/sub_issues -F sub_issue_id=$(gh api repos/ArtCouteau/specflow-solver/issues/<child> --jq .id)`

## Labels

Only two labels are used:

- `ready-for-agent`: fully specified; an agent can implement it without asking.
- `ready-for-human`: needs a person (a decision, a review, hardware or manual work).

## Specs and tickets

- A **spec** is one issue, labelled `ready-for-agent`.
- A **ticket** is one issue per vertical slice, created as a sub-issue of its spec, with `## What to build`, `## Acceptance criteria` and `## Blocked by` sections.
- Publish tickets blockers first, so every `#<n>` reference exists.
- After publishing, report the issue numbers and URLs to the user.
