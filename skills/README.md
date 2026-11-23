# Skills

Task guides written as plain Markdown, versioned alongside the code they
describe. Each is a directory containing `SKILL.md` and, where the detail would
crowd it out, a `reference/` folder.

| Skill | Use when |
|---|---|
| [`multianimal-lp3d-dataset`](multianimal-lp3d-dataset/SKILL.md) | A new multi-animal multiview dataset arrives, or you need to build, verify, configure or train its cropped / per-animal / SAM-recoloured variants, or map predictions back to the combined format |

## Using them

**Any assistant or person:** read `SKILL.md`, follow the references it names.
Nothing depends on a particular runtime.

**Claude Code:** `.claude/skills/<name>` is a symlink to the directory here, so
skills are discovered automatically. Add a new one with:

```bash
ln -s /teamspace/studios/this_studio/lp3d-analysis/skills/<name> \
      /teamspace/studios/this_studio/.claude/skills/<name>
```

## Writing one

The YAML frontmatter (`name`, `description`) is what Claude Code matches against
a request, so the description should say both *what* the skill covers and *when*
to reach for it. Everything below the frontmatter is ordinary Markdown.

Keep `SKILL.md` to the workflow — the steps, in order, with the commands. Push
formats, catalogues and rationale into `reference/`, so the entry point stays
short enough to read before starting rather than during.
